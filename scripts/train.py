import re
import torch
from torch.utils.data import Dataset, DataLoader
from src.decoder_transformer.model import DecoderOnlyTransformer
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Tiny corpus
CORPUS = [
    "<SOS> I Love Transformers <EOS>",
    "<SOS> Transformers are awesome <EOS>",
    "<SOS> I Love attention mechanisms <EOS>",
    "<SOS> Self attention learns token relations <EOS>",
]
def tok(s): return re.findall(r"<EOS>|<SOS>|[\w]+|[^\w\s]", s)

tokens = []
for line in CORPUS: tokens += tok(line)
for sp in ["<PAD>", "<SOS>", "<EOS>"]:
    if sp not in tokens: tokens.append(sp)
itos = sorted(set(tokens)); stoi = {t:i for i,t in enumerate(itos)}
PAD_ID, SOS_ID, EOS_ID = stoi["<PAD>"], stoi["<SOS>"], stoi["<EOS>"]
VOCAB_SIZE = len(itos)

stream = []
for _ in range(32):
    for line in CORPUS:
        stream += [stoi[t] for t in tok(line)]
ids = torch.tensor(stream, dtype=torch.long)

BLOCK = 24; BATCH = 16; N_EMBD=64; N_HEADS=4; N_LAYERS=2; DROPOUT=0.1
LR = 3e-4; ITERS=300; DEV = "cuda" if torch.cuda.is_available() else "cpu"

class DS(Dataset):
    def __len__(self): return max(0, len(ids)-BLOCK)
    def __getitem__(self, i):
        c = ids[i:i+BLOCK+1]; return c[:-1], c[1:]

loader = DataLoader(DS(), batch_size=BATCH, shuffle=True, drop_last=True)
model = DecoderOnlyTransformer(VOCAB_SIZE, N_EMBD, N_HEADS, N_LAYERS, DROPOUT, BLOCK, pad_id=PAD_ID).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR)

def train_epoch():
    model.train(); tot=0
    for x,y in loader:
        x,y=x.to(DEV),y.to(DEV)
        opt.zero_grad(set_to_none=True)
        _,loss = model(x,y)
        loss.backward(); opt.step()
        tot += loss.item()
    return tot/len(loader)

for it in range(1, ITERS+1):
    loss = train_epoch()
    if it%50==0 or it==ITERS: print(f"iter {it:4d} | train loss {loss:.4f}")

# Save
ckpt = "checkpoints/model.pt"
import os; os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "state_dict": model.state_dict(),
    "meta": {"vocab": itos, "pad_id": PAD_ID, "block": BLOCK}
}, ckpt)
print("Saved:", ckpt)
