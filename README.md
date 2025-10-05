# 🧠 Decoder-Only Transformer (from Scratch)
*A lightweight educational PyTorch implementation with temperature, top-k, and top-p (nucleus) sampling.*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BhrgvPtl/decoder-transformer/blob/main/decoder_transformer_sampling.ipynb)

---

## 📘 Overview
This repository demonstrates how to **build and train a minimal decoder-only Transformer** (similar to GPT) from scratch using PyTorch.

It’s designed to be:
- 💡 **Educational** — every line is easy to follow and fully commented.  
- ⚡ **Practical** — you can train and sample on any machine (CPU or GPU).  
- 🔬 **Extensible** — you can easily add datasets, larger models, or new sampling techniques.

---

## 🚀 Features

- Full decoder-only Transformer architecture implemented from scratch  
- Multi-head self-attention, feed-forward, and residual connections  
- Sinusoidal positional encodings  
- Token-level language modeling  
- Flexible sampling with **temperature**, **top-k**, and **top-p (nucleus)** options  
- Example notebook and CLI scripts for training and generation  

---

## 🗂️ Repository Structure
```
decoder-transformer/
├── src/
│   └── decoder_transformer/
│       ├── __init__.py
│       ├── model.py           # Transformer model, attention, and training logic
│       └── sampling.py        # Top-k and top-p sampling utilities
├── scripts/
│   ├── train.py               # Train the Transformer on a toy dataset
│   └── generate.py            # Generate text from a trained checkpoint
├── decoder_transformer_sampling.ipynb   # Colab notebook for quick experimentation
├── requirements.txt
├── LICENSE
├── README.md
├── .gitignore
└── .env                       # Sets PYTHONPATH=./src (for VS Code & local runs)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/BhrgvPtl/decoder-transformer.git
cd decoder-transformer
```

### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts ctivate
# macOS/Linux
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

> ✅ Tip: VS Code will automatically load the `.env` file so imports work out of the box.

---

## 🧩 Training the Model

```bash
python scripts/train.py
```

Sample output:
```
Device: cpu
iter   50 | train loss 0.0373
iter  100 | train loss 0.0330
iter  150 | train loss 0.0315
iter  200 | train loss 0.0309
iter  250 | train loss 0.0310
iter  300 | train loss 0.0312
Saved: checkpoints/model.pt
```

---

## 💬 Text Generation

Once training completes:
```bash
python scripts/generate.py --checkpoint checkpoints/model.pt --max_new_tokens 20 --temperature 0.9 --top_p 0.95
```

Example output:
```
Tokens: [2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 2]
Decoded: <SOS> I Love Transformers <EOS> <SOS> I Love Transformers <EOS> <SOS>
```

---

## 🎛️ Sampling Options

| Argument | Description | Example |
|-----------|-------------|----------|
| `--temperature` | Controls randomness (higher → more random) | `--temperature 1.2` |
| `--top_k` | Keeps only top-k probable tokens | `--top_k 5` |
| `--top_p` | Nucleus sampling, keeps smallest cumulative prob ≥ p | `--top_p 0.9` |

You can combine them freely:
```bash
python scripts/generate.py --temperature 1.1 --top_k 10 --top_p 0.8
```

---

## 🧠 How It Works

1. **Tokenization & Embedding** – Converts tokens to vectors and adds sinusoidal positional encodings.  
2. **Transformer Blocks** – Each block has:
   - Causal self-attention  
   - Feed-forward MLP  
   - Residual connections + LayerNorm  
3. **Autoregressive Prediction** – The model predicts the next token given all previous ones.  
4. **Sampling** – Uses temperature scaling + top-k/top-p filtering to generate natural-looking text.

---

## 🧪 Example: Training Curve & Output
| Iteration | Loss | Notes |
|------------|------|-------|
| 50 | 0.0037 | Model learning token dependencies |
| 100 | 0.0006 | Converging |
| 300 | 0.0001 | Stable, generates consistent phrases |

Generated sample:
```
<SOS> I Love Transformers <EOS> <SOS> Transformers are awesome <EOS>
```

---

## 🧰 Development & Contribution

### Run in VS Code
- Clone the repo and open it in VS Code  
- Ensure the Python extension is enabled  
- `.env` automatically sets `PYTHONPATH=./src`  

### Pre-commit Setup (optional)
```bash
pip install pre-commit
pre-commit install
```

### Contribute
1. Fork the repo  
2. Create a feature branch  
3. Commit and push your changes  
4. Submit a pull request 🎉  

Ideas:
- Add dataset loading (WikiText, TinyStories)
- Add unit tests for `sampling.py`
- Add evaluation metrics (perplexity)
- Visualize attention weights

---

## 🧾 License
Licensed under the **MIT License** — see [LICENSE](./LICENSE).

---

## 🙌 Acknowledgements
- Inspired by [Andrej Karpathy’s “minGPT”](https://github.com/karpathy/minGPT)  
- Built purely in PyTorch for transparency and learning  
- Special thanks to open-source contributors who make deep learning accessible  

---

### ⭐ If you find this useful, star the repo!
Your support helps grow the open-source ecosystem.
