import os
import math
import random
import torch
import torch.nn as nn
import sentencepiece as spm

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# SHELLYQT+ QWEN-INSPIRED TRAINER
# ==========================================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(BASE, "data/train.txt")
TOKENIZER = os.path.join(BASE, "tokenizer/shelly.model")
CHECKPOINTS = os.path.join(BASE, "checkpoints")

os.makedirs(CHECKPOINTS, exist_ok=True)

torch.set_num_threads(2)
torch.manual_seed(42)
random.seed(42)

DEVICE = "cpu"

# Model configuration
VOCAB_SIZE = 2000
DIM = 384
LAYERS = 6
HEADS = 8
HIDDEN = 1024
CONTEXT = 256

BATCH_SIZE = 2
EPOCHS = 20
PATIENCE = 3
LR = 0.0001


# ==========================================
# DATA PARSER
# ==========================================

def clean_line(line):
    line = line.strip().lstrip("\ufeff")
    return line.rstrip("~").strip()


def load_pairs():

    with open(DATA, "r", encoding="utf-8") as f:
        lines = f.readlines()

    pairs = []
    plain_runs = []

    current_run = []
    pending_user = None

    def flush_run():
        nonlocal current_run

        if current_run:
            plain_runs.append(current_run)
            current_run = []

    for raw in lines:

        line = clean_line(raw)

        if not line:
            flush_run()
            continue

        if line.startswith("#"):
            flush_run()
            continue

        lower = line.lower()

        if lower.startswith("you:"):
            flush_run()
            pending_user = line[4:].strip()

        elif lower.startswith("shelly:"):
            flush_run()

            response = line[7:].strip()

            if pending_user and response:
                pairs.append((pending_user, response))

            pending_user = None

        else:
            current_run.append(line)

    flush_run()

    # Parse unlabeled conversations.
    def is_prompt(text):

        text = text.lower().strip()

        if not text:
            return False

        if "?" in text:
            return True

        greetings = {
            "hi", "hello", "hey", "bye",
            "lol", "bruh", "thanks",
            "thank you", "wow", "okay",
            "yes", "no"
        }

        if text in greetings:
            return True

        starts = (
            "what ", "who ", "where ",
            "when ", "why ", "how ",
            "do you ", "are you ",
            "can you ", "is ",
            "tell me ", "i am ",
            "im ", "i like ",
            "i have ", "i made "
        )

        return text.startswith(starts)

    for run in plain_runs:

        for i in range(len(run) - 1):

            prompt = run[i]
            response = run[i + 1]

            if is_prompt(prompt):
                pairs.append((prompt, response))

    # Remove duplicates.
    unique = []
    seen = set()

    for prompt, response in pairs:

        key = (prompt.lower(), response.lower())

        if key not in seen:
            seen.add(key)
            unique.append((prompt, response))

    return unique


# ==========================================
# TOKENIZER
# ==========================================

print("Loading tokenizer...")

sp = spm.SentencePieceProcessor(
    model_file=TOKENIZER
)

VOCAB_SIZE = sp.get_piece_size()

PAD = sp.pad_id()
EOS = sp.eos_id()

print("Vocabulary:", VOCAB_SIZE)


# ==========================================
# DATASET
# ==========================================

print("\nReading conversations...")

pairs = load_pairs()

print("Found conversations:", len(pairs))

if len(pairs) < 10:
    raise RuntimeError("Not enough conversation pairs!")


random.shuffle(pairs)

split = max(1, int(len(pairs) * 0.9))

train_pairs = pairs[:split]
val_pairs = pairs[split:]

if not val_pairs:
    raise RuntimeError("Not enough validation examples!")


class ChatDataset(Dataset):

    def __init__(self, examples):

        self.samples = []

        for prompt, response in examples:

            prefix = sp.encode(
                f"User: {prompt}\nShelly:",
                out_type=int
            )

            answer = sp.encode(
                " " + response,
                out_type=int
            )

            answer.append(EOS)

            if len(prefix) >= CONTEXT - 1:
                prefix = prefix[-(CONTEXT - 2):]

            available = CONTEXT - len(prefix)

            if len(answer) > available:
                answer = answer[:available]
                answer[-1] = EOS

            sequence = prefix + answer

            x = sequence[:-1]
            y = sequence[1:]

            # Only learn the assistant's response.
            masked = len(prefix) - 1
            y[:masked] = [-100] * masked

            self.samples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate(batch):

    max_len = max(len(x) for x, y in batch)

    xs = []
    ys = []

    for x, y in batch:

        padding = max_len - len(x)

        xs.append(torch.cat([
            x,
            torch.full((padding,), PAD)
        ]))

        ys.append(torch.cat([
            y,
            torch.full((padding,), -100)
        ]))

    return torch.stack(xs), torch.stack(ys)


train_data = ChatDataset(train_pairs)
val_data = ChatDataset(val_pairs)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    collate_fn=collate
)


# ==========================================
# ROTARY POSITION EMBEDDINGS
# ==========================================

def apply_rope(x):

    length = x.shape[2]
    head_dim = x.shape[-1]

    half = head_dim // 2

    positions = torch.arange(
        length,
        device=x.device,
        dtype=torch.float32
    )

    frequencies = 1.0 / (
        10000 ** (
            torch.arange(
                half,
                device=x.device,
                dtype=torch.float32
            ) / half
        )
    )

    angles = positions[:, None] * frequencies[None, :]

    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    x = x.reshape(
        *x.shape[:-1],
        half,
        2
    )

    even = x[..., 0]
    odd = x[..., 1]

    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos

    return torch.stack(
        (rotated_even, rotated_odd),
        dim=-1
    ).flatten(-2)


# ==========================================
# RMS NORMALIZATION
# ==========================================

class RMSNorm(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

        self.eps = 1e-6

    def forward(self, x):

        norm = torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )

        return x * norm * self.weight


# ==========================================
# CAUSAL ATTENTION
# ==========================================

class Attention(nn.Module):

    def __init__(self):

        super().__init__()

        self.head_dim = DIM // HEADS

        self.qkv = nn.Linear(
            DIM,
            DIM * 3,
            bias=False
        )

        self.output = nn.Linear(
            DIM,
            DIM,
            bias=False
        )

    def forward(self, x):

        batch, length, _ = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        def split_heads(t):

            return t.view(
                batch,
                length,
                HEADS,
                self.head_dim
            ).transpose(1, 2)

        q = apply_rope(split_heads(q))
        k = apply_rope(split_heads(k))
        v = split_heads(v)

        result = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        result = result.transpose(1, 2).contiguous()

        result = result.view(batch, length, DIM)

        return self.output(result)


# ==========================================
# SWIGLU
# ==========================================

class SwiGLU(nn.Module):

    def __init__(self):

        super().__init__()

        self.gate = nn.Linear(DIM, HIDDEN, bias=False)
        self.up = nn.Linear(DIM, HIDDEN, bias=False)
        self.down = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x):

        return self.down(
            nn.functional.silu(self.gate(x)) * self.up(x)
        )


# ==========================================
# TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.norm1 = RMSNorm(DIM)
        self.norm2 = RMSNorm(DIM)

        self.attention = Attention()
        self.mlp = SwiGLU()

    def forward(self, x):

        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


# ==========================================
# SHELLY BRAIN
# ==========================================

class ShellyBrain(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            DIM
        )

        self.layers = nn.ModuleList([
            TransformerBlock()
            for _ in range(LAYERS)
        ])

        self.norm = RMSNorm(DIM)

        self.output = nn.Linear(
            DIM,
            VOCAB_SIZE,
            bias=False
        )

        self.output.weight = self.embedding.weight

    def forward(self, x):

        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        return self.output(x)


# ==========================================
# TRAINING
# ==========================================

model = ShellyBrain().to(DEVICE)

parameters = sum(
    p.numel() for p in model.parameters()
)

print("\nShellyQT+ MLM")
print("======================")
print("Parameters:", f"{parameters:,}")
print("Layers:", LAYERS)
print("Embedding:", DIM)
print("Attention heads:", HEADS)
print("Context:", CONTEXT)
print("Training pairs:", len(train_data))
print("Validation pairs:", len(val_data))
print("CPU threads:", torch.get_num_threads())


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.05
)

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)

best_loss = float("inf")
bad_epochs = 0


def evaluate():

    model.eval()

    total = 0
    tokens = 0

    with torch.no_grad():

        for x, y in val_loader:

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )

            count = (y != -100).sum().item()

            total += loss.item() * count
            tokens += count

    model.train()

    return total / max(tokens, 1)


print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total = 0
    tokens = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for x, y in progress:

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, VOCAB_SIZE),
            y.reshape(-1)
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        count = (y != -100).sum().item()

        total += loss.item() * count
        tokens += count

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = total / max(tokens, 1)
    val_loss = evaluate()

    print(
        f"\nEpoch {epoch + 1}: "
        f"train={train_loss:.4f} "
        f"validation={val_loss:.4f}"
    )

    if math.isfinite(val_loss) and val_loss < best_loss:

        best_loss = val_loss
        bad_epochs = 0

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "validation_loss": val_loss,
                "config": {
                    "vocab_size": VOCAB_SIZE,
                    "dim": DIM,
                    "layers": LAYERS,
                    "heads": HEADS,
                    "hidden": HIDDEN,
                    "context": CONTEXT
                }
            },
            os.path.join(
                CHECKPOINTS,
                "best_chat.pt"
            )
        )

        print("Saved best checkpoint!")

    else:

        bad_epochs += 1

        print(
            "No improvement:",
            bad_epochs,
            "/",
            PATIENCE
        )

    if bad_epochs >= PATIENCE:

        print("\nEarly stopping!")
        break


print("\nTraining complete!")
print("Best validation loss:", best_loss)
print("Saved: checkpoints/best_chat.pt")
