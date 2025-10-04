import argparse
import torch
from src.decoder_transformer.model import DecoderOnlyTransformer
from src.decoder_transformer.sampling import sampler
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--top_p", type=float, default=None)
    args = p.parse_args()

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=DEV)
    itos = ckpt["meta"]["vocab"]; stoi = {t:i for i,t in enumerate(itos)}
    pad_id = ckpt["meta"]["pad_id"]; block = ckpt["meta"]["block"]
    vocab_size = len(itos)

    model = DecoderOnlyTransformer(vocab_size, 64, 4, 2, 0.1, block, pad_id).to(DEV)
    model.load_state_dict(ckpt["state_dict"]); model.eval()

    start = torch.tensor([[stoi["<SOS>"]]], dtype=torch.long, device=DEV)
    step = sampler(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    out = model.generate(start, args.max_new_tokens, step_fn=step)[0].tolist()

    print("Tokens:", out)
    print("Decoded:", " ".join(itos[i] for i in out))

if __name__ == "__main__":
    main()
