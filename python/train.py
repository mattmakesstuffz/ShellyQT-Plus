import os
import math
import random
import torch
import torch.nn as nn
import sentencepiece as spm

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(BASE, "data", "train.txt")
TOKENIZER = os.path.join(BASE, "tokenizer", "shelly.model")
CHECKPOINTS = os.path.join(BASE, "checkpoints")

DEVICE = torch.device("cpu")

torch.set_num_threads(2)
torch.manual_seed(42)
random.seed(42)

VOCAB_SIZE = 2000
EMBED_DIM = 128
LAYERS = 3
HEADS = 4
CONTEXT = 128
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0003

os.makedirs(CHECKPOINTS, exist_ok=True)

# ---------------- TOKENIZER ----------------

print("Loading SentencePiece...")

sp = spm.SentencePieceProcessor(
    model_file=TOKENIZER
)

VOCAB_SIZE = sp.get_piece_size()

# ---------------- DATASET ----------------

print("Loading training data...")

with open(DATA, "r", encoding="utf-8") as f:
    text = f.read()

ids = sp.encode(text, out_type=int)

ids = torch.tensor(ids, dtype=torch.long)

if len(ids) <= CONTEXT:
    raise ValueError("Dataset is too small!")

# Predict the next token.
class TextDataset(Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens) - CONTEXT

    def __getitem__(self, index):
        x = self.tokens[index:index + CONTEXT]
        y = self.tokens[index + 1:index + CONTEXT + 1]
        return x, y

# Split sequentially to reduce train/validation leakage.
split = int(len(ids) * 0.9)

train_data = TextDataset(ids[:split])
val_data = TextDataset(ids[split - CONTEXT:])

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE
)

# ---------------- TRANSFORMER ----------------

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = nn.LayerNorm(EMBED_DIM)

        self.attention = nn.MultiheadAttention(
            EMBED_DIM,
            HEADS,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(EMBED_DIM)

        self.feedforward = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM * 4),
            nn.GELU(),
            nn.Linear(EMBED_DIM * 4, EMBED_DIM)
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


# ---------------- TRAINING ----------------

model = ShellyBrain().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

criterion = nn.CrossEntropyLoss()

best_loss = float("inf")

parameters = sum(
    p.numel() for p in model.parameters()
)

print("\nShellyQT+ Brain")
print("================")
print("Parameters:", f"{parameters:,}")
print("Vocabulary:", VOCAB_SIZE)
print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))
print("Device:", DEVICE)

def evaluate():
    model.eval()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                y.reshape(-1)
            )

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    model.train()

    return total_loss / max(total_tokens, 1)


print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for x, y in progress:
        x, y = x.to(DEVICE), y.to(DEVICE)

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

        total_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    scheduler.step()

    average = total_loss / max(len(train_loader), 1)
    validation = evaluate()

    print(
        f"Epoch {epoch + 1}: "
        f"train={average:.4f} "
        f"validation={validation:.4f}"
    )

    if math.isfinite(validation) and validation < best_loss:
        best_loss = validation

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "validation_loss": validation,
                "config": {
                    "vocab_size": VOCAB_SIZE,
                    "embed_dim": EMBED_DIM,
                    "layers": LAYERS,
                    "heads": HEADS,
                    "context": CONTEXT
                }
            },
            os.path.join(CHECKPOINTS, "best.pt")
        )

        print("Saved new best checkpoint!")

print("\nTraining finished!")
print("Best validation loss:", best_loss)
