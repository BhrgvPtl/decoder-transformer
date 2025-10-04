# Decoder-Only Transformer (from scratch) with Temperature / Top-K / Top-P Sampling

A tiny, educational decoder-only Transformer implemented in PyTorch. Trains on a toy corpus and demonstrates generation with temperature, top-k, and nucleus (top-p) sampling.

[![Open In Colab](https://colab.research.google.com/drive/1TacSObyPawVmIbUkykn6UNEgiWLEo_Ha?usp=sharing)

## Quick Start (Local)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train.py
python scripts/generate.py --checkpoint checkpoints/model.pt --max_new_tokens 20 --temperature 0.9 --top_p 0.95
