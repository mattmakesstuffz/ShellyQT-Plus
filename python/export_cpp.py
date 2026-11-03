import torch
import struct
from pathlib import Path

CHECKPOINT = Path("checkpoints/best_chat_25m.pt")
OUTPUT = Path("weights/shelly_25m.bin")

print("Loading ShellyQT+ 25M checkpoint...")

if not CHECKPOINT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

state = checkpoint["model"]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("wb") as f:
    f.write(b"SHELLYQ1")

    tensors = [
        (name, tensor.detach().cpu().float().contiguous().numpy())
        for name, tensor in state.items()
    ]

    f.write(struct.pack("<I", len(tensors)))

    for name, array in tensors:
        name_bytes = name.encode("utf-8")

        f.write(struct.pack("<I", len(name_bytes)))
        f.write(name_bytes)

        f.write(struct.pack("<I", array.ndim))

        for dimension in array.shape:
            f.write(struct.pack("<I", dimension))

        f.write(array.astype("<f4", copy=False).tobytes())

        print(f"Exported: {name} {array.shape}")

print("\nExport complete!")
print(f"Output: {OUTPUT}")
print(f"Tensor count: {len(tensors)}")
print(f"Size: {OUTPUT.stat().st_size / 1024 / 1024:.2f} MB")
