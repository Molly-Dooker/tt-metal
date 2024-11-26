# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc, divup, is_grayskull, skip_for_blackhole


# @skip_for_blackhole("Mismatching on Blackhole, see #12349")
@pytest.mark.parametrize("W, Z, Y, X", [(1, 1, 32, 64)])
@pytest.mark.parametrize("cache_size", [2048])
@pytest.mark.parametrize("token_idx", [0])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("sincos_dtype", [ttnn.bfloat8_b])
def test_rotary_embedding_decode(W, Z, Y, X, cache_size, token_idx, input_dtype, sincos_dtype, device):
    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]
    x = torch.ones(input_shape).bfloat16().float()
    cos_cached = torch.ones(sin_cos_shape).bfloat16().float() * 9
    sin_cached = torch.ones(sin_cos_shape).bfloat16().float() * 99

    out_mem_config = ttnn.MemoryConfig()
    xt = ttnn.Tensor(x, input_dtype)

    # if branch taken for shape [1,1,32,64]
    if xt.shape.with_tile_padding()[-2] % 32 == 0 and xt.shape.with_tile_padding()[-1] % 32 == 0:
        xt = xt.to(ttnn.TILE_LAYOUT)
    elif input_dtype == ttnn.bfloat8_b:
        pytest.skip()

    xt = xt.to(device)
    cost = ttnn.Tensor(cos_cached, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    sint = ttnn.Tensor(sin_cached, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    xtt = ttnn.experimental.rotary_embedding(xt, cost, sint, token_idx, memory_config=out_mem_config)

    tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    pt_out = -x

    if not torch.equal(pt_out, tt_got_back):
        print("TENSOR MISMATCH")
        torch.set_printoptions(profile="full")
        print("torch tensor")
        print(pt_out)
        print("device tensor")
        print(tt_got_back)
        assert False
