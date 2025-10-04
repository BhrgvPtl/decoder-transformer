# scripts/generate.py
# Load a trained checkpoint and generate text with temperature/top-k/top-p sampling.

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# --- Make the src/ package importable regardless of where we run this script ---
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decoder_transformer.model import DecoderOnlyTransformer  # noqa: E402
from decoder_transformer.sampling import sampler  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "model.pt"))
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    args = p.parse_args()

    DEV = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=DEV)
    itos = ckpt["meta"]["vocab"]
    stoi = {t: i for i, t in enumerate(itos)}
    pad_id = ckpt["meta"]["pad_id"]
    block = ckpt["meta"]["block"]
    vocab_size = len(itos)

    # Rebuild model and load weights
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        n_embd=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        block_size=block,
        pad_id=pad_id,
    ).to(DEV)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Start from <SOS>
    start = torch.tensor([[stoi["<SOS>"]]], dtype=torch.long, device=DEV)

    # Build a sampling step function and generate
    step = sampler(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    out = model.generate(start, max_new_tokens=args.max_new_tokens, step_fn=step)[0].tolist()

    print("Tokens:", out)
    print("Decoded:", " ".join(itos[i] for i in out))


if __name__ == "__main__":
    main()
