# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scaled dot-product attention with optional CUDA kernel."""

from __future__ import annotations

import math
import os
import pathlib
from typing import Optional

import torch

# Try to import the compiled extension.  If it is not yet built we lazily build
# it using ``torch.utils.cpp_extension``.
try:  # pragma: no cover - the extension may not be available during docs build
    from .cpp import flex_attention_cpp  # type: ignore
except Exception:  # pragma: no cover
    flex_attention_cpp = None
    _this_dir = pathlib.Path(__file__).parent
    _cpp_dir = _this_dir / "cpp" / "src"
    _cuda_dir = _cpp_dir / "cuda"
    _sources = [
        str(_cpp_dir / "flex_attention.cpp"),
        str(_cuda_dir / "flex_attention.cu"),
    ]
    if all(pathlib.Path(s).exists() for s in _sources):
        from torch.utils.cpp_extension import load  # type: ignore

        flex_attention_cpp = load(
            name="flex_attention_cpp", sources=_sources, verbose=False
        )


def _python_flex_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """Reference Python implementation."""
    scores = torch.matmul(query_states, key_states.transpose(-2, -1))
    scores = scores * scale
    if attention_mask is not None and attention_mask.numel() > 0:
        scores = scores + attention_mask
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, value_states)


def flex_attention_forward(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    head_dim: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: Optional[float] = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Parameters
    ----------
    attention_mask: tensor or ``None``
        Additive mask applied to attention scores.  It is expected to be broadcastable
        to ``[B, Q, K]``.
    batch_size: int
        Unused but kept for API compatibility with existing callers.
    head_dim: int
        Dimension of each attention head.  Used to compute the default scaling factor.
    query_states / key_states / value_states: tensor
        Input tensors of shape ``[B, Q, D]``, ``[B, K, D]`` and ``[B, K, D]`` respectively.
    scaling: float, optional
        Optional pre-computed scaling factor.  If ``None`` ``1/\sqrt(head_dim)`` is used.
    """

    scale = float(scaling) if scaling is not None else head_dim ** -0.5

    if query_states.is_cuda and flex_attention_cpp is not None:
        # The CUDA extension expects an explicit tensor for the mask.
        mask = attention_mask if attention_mask is not None else torch.empty(0, device=query_states.device)
        return flex_attention_cpp.FlexAttention.forward(
            query_states, key_states, value_states, mask, scale
        )

    # Fallback to pure Python implementation.
    return _python_flex_attention(query_states, key_states, value_states, attention_mask, scale)
