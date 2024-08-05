"""
layer norm modules
"""

import os
from typing import List, Union

import torch
from torch import nn

from internlm.model.ops.norm import RMSNorm

# from internlm.simulator.fake_ops import FakeLayerNorm
Shape = Union[int, List[int], torch.Size]


fake_mode = "fake_mode" in os.environ


def new_layer_norm(norm_type: str, normalized_shape: Shape, eps: float = 1e-5):
    # if fake_mode:
    #     return FakeLayerNorm(normalized_shape, eps)
    if norm_type == "rmsnorm":
        return RMSNorm(normalized_shape, eps)
    else:  # default: layernorm
        return nn.LayerNorm(normalized_shape, eps)
