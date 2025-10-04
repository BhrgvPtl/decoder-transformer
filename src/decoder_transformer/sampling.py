from typing import Optional
import torch
import torch.nn.functional as F

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    B, V = logits.shape
    if top_k is not None and 1 <= top_k < V:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
        logits = logits.masked_fill(logits < kth, filter_value)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        scatter = torch.zeros_like(mask).scatter(1, sorted_idx, mask)
        logits = logits.masked_fill(scatter, filter_value)
    return logits

def sampler(temperature=1.0, top_k=None, top_p=None):
    def step_fn(next_logits):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        next_logits = next_logits / temperature
        next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(next_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return step_fn
