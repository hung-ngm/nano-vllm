"""Record plain torch (no flash-attn, no Triton) for the layer math.

The original attention path calls flash-attn. This file is the CPU reference
the port is checked against: same formulas, real tensor numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent / "traces" / "cpu_model.jsonl"


def flat(t: torch.Tensor) -> list[str]:
    return [format(float(x), ".17g") for x in t.detach().float().reshape(-1).tolist()]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    y = x.float() * torch.rsqrt(var + eps)
    return y * weight.float()


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    a, b = x.float().chunk(2, dim=-1)
    return F.silu(a) * b


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    y1 = x1 * cos.float() - x2 * sin.float()
    y2 = x2 * cos.float() + x1 * sin.float()
    return torch.cat((y1, y2), dim=-1)


def rope_cache(head_dim: int, max_pos: int, base: float) -> torch.Tensor:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


def causal_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    length = q.shape[0]
    mask = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hts,shd->thd", probs, v.float())


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    return F.linear(x.float(), weight.float(), None if bias is None else bias.float())


def sample(logits: torch.Tensor, temperature: float, noise: torch.Tensor) -> torch.Tensor:
    scaled = logits.float() / temperature
    probs = torch.softmax(scaled, dim=-1)
    return (probs / noise.float().clamp_min(1e-10)).argmax(dim=-1)


def main() -> None:
    cases = []
    g = torch.Generator().manual_seed(0)
    for name, shape in [("rms-row", (1, 8)), ("rms-batch", (3, 4))]:
        x = torch.randn(*shape, generator=g)
        w = torch.randn(shape[-1], generator=g)
        cases.append({"op": "rms", "id": name, "x": flat(x), "shape": list(x.shape), "weight": flat(w), "eps": 1e-6, "out": flat(rms_norm(x, w))})
    for name, n in [("silu", 8), ("silu-2", 4)]:
        x = torch.randn(2, n, generator=g)
        cases.append({"op": "silu", "id": name, "x": flat(x), "shape": list(x.shape), "out": flat(silu_and_mul(x))})
    x = torch.randn(2, 4, 8, generator=g)
    cos = torch.randn(2, 4, 4, generator=g)
    sin = torch.randn(2, 4, 4, generator=g)
    cases.append({"op": "rope", "id": "rotary", "x": flat(x), "x_shape": list(x.shape), "cos": flat(cos), "cos_shape": list(cos.shape), "sin": flat(sin), "sin_shape": list(sin.shape), "out": flat(apply_rotary(x, cos, sin))})
    cache = rope_cache(8, 4, 10000.0)
    cases.append({"op": "rope_cache", "id": "cache", "head_dim": 8, "max_pos": 4, "base": 10000.0, "out": flat(cache), "shape": list(cache.shape)})
    q = torch.randn(3, 2, 4, generator=g)
    k = torch.randn(3, 2, 4, generator=g)
    v = torch.randn(3, 2, 4, generator=g)
    scale = 4 ** -0.5
    cases.append({"op": "attn", "id": "causal", "q": flat(q), "k": flat(k), "v": flat(v), "shape": [3, 2, 4], "scale": scale, "out": flat(causal_attn(q, k, v, scale))})
    x = torch.randn(2, 6, generator=g)
    weight = torch.randn(5, 6, generator=g)
    bias = torch.randn(5, generator=g)
    cases.append({"op": "linear", "id": "with-bias", "x": flat(x), "x_shape": [2, 6], "weight": flat(weight), "w_shape": [5, 6], "bias": flat(bias), "out": flat(linear(x, weight, bias))})
    logits = torch.randn(2, 6, generator=g)
    noise = torch.empty_like(logits).exponential_(1)
    tokens = sample(logits, 0.7, noise)
    cases.append({"op": "sample", "id": "gumbel", "logits": flat(logits), "shape": [2, 6], "temperature": 0.7, "noise": flat(noise), "out": [int(t) for t in tokens.tolist()]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, separators=(",", ":")) + "\n")
    print(f"cpu_model: {len(cases)} cases")


if __name__ == "__main__":
    main()
