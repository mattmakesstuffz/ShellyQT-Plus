import os
import math
import random
import torch
import torch.nn as nn
import sentencepiece as spm

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= CONFIG =================

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(BASE, "data", "train.txt")
TOKENIZER = os.path.join(BASE, "tokenizer", "shelly.model")
CHECKPOINTS = os.path.join(BASE, "checkpoints")

DEVICE = torch.device("cpu")

torch.set_num_threads(2)
torch.manual_seed(42)
random.seed(42)

EMBED_DIM = 128
LAYERS = 2
HEADS = 4
CONTEXT = 128

BATCH_SIZE = 8
EPOCHS = 20
PATIENCE = 3
LEARNING_RATE = 0.0001

os.makedirs(CHECKPOINTS, exist_ok=True)


# ================= PARSER =================

def read_blocks(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    blocks = []
    current = []

    for line in lines:
        line = line.strip()

        if not line:
            if current:
                blocks.append(current)
                current = []
            continue

        if line.startswith("#"):
            continue

        # The ~ character is just a terminator.
        line = line.rstrip("~").strip()

        if line:
            current.append(line)

    if current:
        blocks.append(current)

    return blocks


def is_prompt(text):
    text = text.lower().strip()
    words = text.split()

    if not words or len(words) > 12:
        return False

    if "?" in text:
        return True

    starts = (
        "what ", "who ", "where ", "when ",
        "why ", "how ", "do you ", "are you ",
        "can you ", "is ", "tell me ",
        "say ", "make ", "repeat ",
        "i am ", "im ", "i made ",
        "my code ", "i found ",
        "i like ", "i have "
    )

    if text.startswith(starts):
        return True

    greetings = {
        "hello", "hi", "hey", "bye",
        "good morning", "good afternoon",
        "good night", "lol", "bruh",
        "banana", "chicken banana",
        "thanks", "thank you", "wow",
        "okay", "yes", "no"
    }

    return text in greetings


def is_definition(text):
    text = text.lower().strip()

    starts = (
        "a ", "an ", "the ",
        "having ", "moving ",
        "with ", "a feeling ",
        "the study "
    )

    return text.startswith(starts)


def load_pairs(path):
    blocks = read_blocks(path)

    pairs = []
    pending_user = None
    plain_blocks = []

    # First: read explicitly labeled conversations.
    for block in blocks:
        for line in block:
            lower = line.lower()

            if lower.startswith("you:"):
                pending_user = line[4:].strip()

            elif lower.startswith("shelly:"):
                response = line[7:].strip()

                if pending_user and response:
                    pairs.append((pending_user, response))

                pending_user = None

            else:
                plain_blocks.append(" ".join(block))
                break

    # Second: identify unlabeled question-answer pairs.
    i = 0

    while i < len(plain_blocks) - 1:
        prompt = plain_blocks[i].strip()
        response = plain_blocks[i + 1].strip()

        valid = is_prompt(prompt)

        # Also recognize dictionary-style entries.
        if len(prompt.split()) <= 3 and is_definition(response):
            valid = True

        if valid and response and not is_prompt(response):
            pairs.append((prompt, response))
            i += 2
        else:
            i += 1

    # Remove duplicates.
    unique = []
    seen = set()

    for prompt, response in pairs:
        key = (prompt.lower(), response.lower())

        if key not in seen:
            seen.add(key)
            unique.append((prompt, response))

    return unique


# ================= TOKENIZER =================

print("Loading SentencePiece...")

sp = spm.SentencePieceProcessor(
    model_file=TOKENIZER
)

VOCAB_SIZE = sp.get_piece_size()
PAD_ID = sp.pad_id()
EOS_ID = sp.eos_id()

print("Vocabulary:", VOCAB_SIZE)


# ================= DATASET =================

print("\nReading conversations...")

pairs = load_pairs(DATA)

print("Conversation pairs found:", len(pairs))

if len(pairs) < 10:
    raise RuntimeError(
        "Not enough conversation pairs were found. "
        "Check the dataset parser before training."
    )


class ChatDataset(Dataset):

    def __init__(self, examples):
        self.examples = []

        for prompt, response in examples:

            prefix = sp.encode(
                f"User: {prompt}\nShelly:",
                out_type=int
            )

            answer = sp.encode(
                " " + response,
                out_type=int
            )

            answer.append(EOS_ID)

            # Preserve room for the answer.
            if len(prefix) >= CONTEXT - 1:
                prefix = prefix[-(CONTEXT - 2):]

            available = CONTEXT - len(prefix)

            if len(answer) > available:
                answer = answer[:available]
                answer[-1] = EOS_ID

            sequence = prefix + answer

            x = sequence[:-1]
            y = sequence[1:]

            # Only calculate loss on Shelly's answer.
            masked = max(0, len(prefix) - 1)

            y[:masked] = [-100] * masked

            self.examples.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


# ================= TRAIN / VALIDATION SPLIT =================

random.shuffle(pairs)

split = max(1, int(len(pairs) * 0.9))

train_pairs = pairs[:split]
val_pairs = pairs[split:]

if not val_pairs:
    raise RuntimeError("Not enough examples for validation.")

train_data = ChatDataset(train_pairs)
val_data = ChatDataset(val_pairs)


def collate(batch):
    max_length = max(len(x) for x, y in batch)

    xs = []
    ys = []

    for x, y in batch:
        padding = max_length - len(x)

        xs.append(
            torch.cat([
                x,
                torch.full((padding,), PAD_ID)
            ])
        )

        ys.append(
            torch.cat([
                y,
                torch.full((padding,), -100)
            ])
        )

    return torch.stack(xs), torch.stack(ys)


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


# ================= TRANSFORMER =================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.attention = nn.MultiheadAttention(
            EMBED_DIM,
            HEADS,
            batch_first=True,
            dropout=0.1
        )

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.feedforward = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM),
            nn.Dropout(0.1)
        )

    def forward(self, x, mask):

        normalized = self.norm1(x)

        attention, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=mask,
            need_weights=False
        )

        x = x + attention
        x = x + self.feedforward(self.norm2(x))

        return x


class ShellyBrain(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM
        )

        self.position = nn.Embedding(
            CONTEXT,
            EMBED_DIM
        )

        self.blocks = nn.ModuleList([
            TransformerBlock()
            for _ in range(LAYERS)
        ])

        self.norm = nn.LayerNorm(EMBED_DIM)

        self.output = nn.Linear(
            EMBED_DIM,
            VOCAB_SIZE,
            bias=False
        )

        self.output.weight = self.embedding.weight

        mask = torch.triu(
            torch.ones(CONTEXT, CONTEXT),
            diagonal=1
        ).bool()

        self.register_buffer("mask", mask)

    def forward(self, x):

        batch, length = x.shape

        positions = torch.arange(
            length,
            device=x.device
        )

        x = self.embedding(x) + self.position(positions)

        mask = self.mask[:length, :length]

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        return self.output(x)


# ================= TRAINING =================

model = ShellyBrain().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.05
)

criterion = nn.CrossEntropyLoss(
    ignore_index=-100
)

best_loss = float("inf")
bad_epochs = 0

parameters = sum(
    p.numel() for p in model.parameters()
)

print("\nShellyQT+ Chatter Brain")
print("=======================")
print("Parameters:", f"{parameters:,}")
print("Training pairs:", len(train_data))
print("Validation pairs:", len(val_data))
print("Maximum epochs:", EPOCHS)
print("CPU threads:", torch.get_num_threads())


def evaluate():

    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():

        for x, y in val_loader:

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )

            count = (y != -100).sum().item()

            total_loss += loss.item() * count
            total_tokens += count

    model.train()

    return total_loss / max(total_tokens, 1)


print("\nStarting chatter training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    total_tokens = 0

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

        total_loss += loss.item() * count
        total_tokens += count

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = total_loss / max(total_tokens, 1)
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
                    "embed_dim": EMBED_DIM,
                    "layers": LAYERS,
                    "heads": HEADS,
                    "context": CONTEXT
                }
            },
            os.path.join(CHECKPOINTS, "best_chat.pt")
        )

        print("Saved new best chatter checkpoint!")

    else:
        bad_epochs += 1
        print("No improvement:", bad_epochs, "/", PATIENCE)

    if bad_epochs >= PATIENCE:
        print("\nEarly stopping! No more useful improvement.")
        break


print("\nChatter training finished!")
print("Best validation loss:", best_loss)
print("Checkpoint:", os.path.join(CHECKPOINTS, "best_chat.pt"))
