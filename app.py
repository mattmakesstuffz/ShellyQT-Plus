import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

from flask import Flask, request, jsonify, render_template

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

torch.set_num_threads(2)

TOKENIZER = spm.SentencePieceProcessor(
    model_file=os.path.join(BASE, "tokenizer/shelly.model")
)

VOCAB_SIZE = TOKENIZER.get_piece_size()
DIM = 512
LAYERS = 8
HEADS = 8
HIDDEN = 1280
CONTEXT = 256
DEVICE = "cpu"

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = 1e-6

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def apply_rope(x):
    length = x.shape[2]
    half = x.shape[-1] // 2
    positions = torch.arange(length, device=x.device, dtype=torch.float32)
    frequencies = 1.0 / (10000 ** (
        torch.arange(half, device=x.device, dtype=torch.float32) / half
    ))
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    x = x.reshape(*x.shape[:-1], half, 2)
    even, odd = x[..., 0], x[..., 1]
    return torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1
    ).flatten(-2)

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = DIM // HEADS
        self.qkv = nn.Linear(DIM, DIM * 3, bias=False)
        self.output = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(t):
            return t.view(batch, length, HEADS, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        q, k = apply_rope(q), apply_rope(k)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(batch, length, DIM)
        return self.output(out)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(DIM, HIDDEN, bias=False)
        self.up = nn.Linear(DIM, HIDDEN, bias=False)
        self.down = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(DIM)
        self.norm2 = RMSNorm(DIM)
        self.attention = Attention()
        self.mlp = SwiGLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class ShellyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, DIM)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(LAYERS)])
        self.norm = RMSNorm(DIM)
        self.output = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))

model = ShellyBrain().to(DEVICE)

checkpoint_path = os.path.join(BASE, "checkpoints/best_chat_25m.pt")

if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

print("Loading ShellyQT+...")
checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint["model"])
model.eval()
print("ShellyQT+ loaded!")

def generate(prompt):
    prefix = TOKENIZER.encode(f"User: {prompt}\nShelly:")
    ids = prefix[-(CONTEXT - 1):]
    generated = []

    with torch.inference_mode():
        for _ in range(60):
            x = torch.tensor([ids[-CONTEXT:]], dtype=torch.long)
            logits = model(x)[0, -1].float().clone()

            # Repetition penalty
            for token_id in set(generated[-48:]):
                if logits[token_id] > 0:
                    logits[token_id] /= 1.25
                else:
                    logits[token_id] *= 1.25

            logits /= 0.8

            # Top-k sampling
            values, indices = torch.topk(logits, min(40, logits.numel()))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(0, indices, values)

            next_id = torch.multinomial(F.softmax(filtered, dim=-1), 1).item()

            if next_id == TOKENIZER.eos_id():
                break

            generated.append(next_id)
            ids.append(next_id)

    return TOKENIZER.decode(generated).strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    message = request.json.get("message", "").strip()

    if not message:
        return jsonify({"reply": "Type something first!"}), 400

    if len(message) > 1000:
        return jsonify({"reply": "That message is too long."}), 400

    try:
        return jsonify({"reply": generate(message)})
    except Exception as exc:
        app.logger.exception("Chat generation failed")
        return jsonify({"reply": "Shelly encountered an error."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
