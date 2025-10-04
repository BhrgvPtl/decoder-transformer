# scripts/train.py
# Train a tiny decoder-only Transformer on a toy corpus and save a checkpoint.

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Make the src/ package importable regardless of where we run this script ---
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decoder_transformer.model import DecoderOnlyTransformer  # noqa: E402


# ----------------------------
# Tiny corpus and tokenization
# ----------------------------
CORPUS = [
    "<SOS> I Love Transformers <EOS>",
    "<SOS> Transformers are awesome <EOS>",
    "<SOS> I Love attention mechanisms <EOS>",
    "<SOS> Self attention learns token relations <EOS>",
]

TOKEN_RE = re.compile(r"<EOS>|<SOS>|[\w]+|[^\w\s]")


def tokenize(s: str) -> List[str]:
    return TOKEN_RE.findall(s)


# Build vocab
tokens_all: List[str] = []
for line in CORPUS:
    tokens_all.extend(tokenize(line))

SPECIALS = ["<PAD>", "<SOS>", "<EOS>"]
for sp in SPECIALS:
    if sp not in tokens_all:
        tokens_all.append(sp)

itos = sorted(set(tokens_all))
stoi = {t: i for i, t in enumerate(itos)}
PAD_ID, SOS_ID, EOS_ID = stoi["<PAD>"], stoi["<SOS>"], stoi["<EOS>"]
VOCAB_SIZE = len(itos)

# Build a simple repeated training stream
encoded_stream: List[int] = []
for _ in range(32):
    for line in CORPUS:
        encoded_stream.extend(stoi[t] for t in tokenize(line))
ids = torch.tensor(encoded_stream, dtype=torch.long)


# ----------------------------
# Dataset
# ----------------------------
class LMWindowedDataset(Dataset):
    """Overlapping (x, y) windows for next-token prediction."""
    def __init__(self, ids: torch.Tensor, block_size: int):
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.ids) - self.block_size)

    def __getitem__(self, i):
        chunk = self.ids[i : i + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def main():
    # --- Config ---
    BLOCK = 24
    BATCH = 16
    N_EMBD = 64
    N_HEADS = 4
    N_LAYERS = 2
    DROPOUT = 0.1
    LR = 3e-4
    ITERS = 300
    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1337)

    # --- Data ---
    ds = LMWindowedDataset(ids, BLOCK)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True)

    # --- Model ---
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        block_size=BLOCK,
        pad_id=PAD_ID,
    ).to(DEV)

    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    def train_epoch() -> float:
        model.train()
        total = 0.0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            opt.step()
            total += loss.item()
        return total / len(dl)

    # --- Train ---
    for it in range(1, ITERS + 1):
        loss = train_epoch()
        if it % 50 == 0 or it == ITERS:
            print(f"iter {it:4d} | train loss {loss:.4f}")

    # --- Save checkpoint ---
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "meta": {"vocab": itos, "pad_id": PAD_ID, "block": BLOCK},
        },
        ckpt_path,
    )
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    main()
