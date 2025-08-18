import torch
import pytest

from lerobot.policies.pi0.flex_attention import flex_attention_forward


def python_attention(q, k, v, mask=None, scale=None):
    d = q.size(-1)
    scale = scale if scale is not None else d ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None and mask.numel() > 0:
        scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def test_flex_attention_matches_python():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for flex attention kernel")

    device = torch.device("cuda")
    B, Q, K, D = 2, 4, 4, 8
    q = torch.randn(B, Q, D, device=device)
    k = torch.randn(B, K, D, device=device)
    v = torch.randn(B, K, D, device=device)
    mask = torch.zeros(B, Q, K, device=device)

    ref = python_attention(q, k, v, mask)
    out = flex_attention_forward(mask, B, D, q, k, v)

    assert torch.allclose(out, ref, atol=1e-5)
