# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc, divup, is_grayskull, skip_for_blackhole


def apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx):
    seq_len = x.shape[-2]
    sin = sin_cached[:, :, token_idx : token_idx + 1, ...]

    print("Sine")
    print(sin)

    x_embed = x * sin
    return x_embed


@pytest.mark.parametrize("W, Z, Y, X", [(1, 1, 32, 64)])
@pytest.mark.parametrize("cache_size", [32])
@pytest.mark.parametrize("token_idx", [0])
@pytest.mark.parametrize("in_sharded", [False])
@pytest.mark.parametrize("out_sharded", [False])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
# @pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
# @pytest.mark.parametrize("input_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("sincos_dtype", [ttnn.bfloat16])
def test_rotary_embedding_decode(
    W, Z, Y, X, cache_size, token_idx, in_sharded, out_sharded, input_dtype, sincos_dtype, device
):
    input_shape = [W, Z, Y, X]
    sin_cos_shape = [1, 1, cache_size, X]

    # Given the token of 0, x as 1s and sin_cache controlled
    # We expect the result to have rows which consist of
    # sequence of 0-1 in increments of 1/64. The cos values
    # are not used. If indexing is wrong, we will get 8.
    x = torch.ones(input_shape).bfloat16().float()
    sin_cached = torch.ones(sin_cos_shape).bfloat16().float() * 8
    sin_cached[0, 0, 0, 0:64] = torch.arange(0, 64) / 64

    cos_cached = torch.randn(sin_cos_shape).bfloat16().float()
    out_mem_config = ttnn.MemoryConfig()

    xt = ttnn.Tensor(x, input_dtype)
    if xt.shape.with_tile_padding()[-2] % 32 == 0 and xt.shape.with_tile_padding()[-1] % 32 == 0:
        xt = xt.to(ttnn.TILE_LAYOUT)
    elif input_dtype == ttnn.bfloat8_b:
        pytest.skip()
        print("Should have skipped")

    xt = xt.to(device)

    cost = ttnn.Tensor(cos_cached, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    sint = ttnn.Tensor(sin_cached, sincos_dtype).to(ttnn.TILE_LAYOUT).to(device)
    xtt = ttnn.experimental.rotary_embedding(xt, cost, sint, token_idx, memory_config=out_mem_config)

    tt_got_back = xtt.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    pt_out = apply_rotary_pos_emb(x, cos_cached, sin_cached, token_idx)

    torch.set_printoptions(profile="full")
    print("\nGolden tensor\n")
    print(pt_out)
    print("\nDevice tensor\n")
    print(tt_got_back)
    p, o = comp_pcc(pt_out, tt_got_back)
    logger.info(o)
    assert p
