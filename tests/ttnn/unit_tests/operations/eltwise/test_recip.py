# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_range_dtype,
)


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
    ],
)
def test_recip(device, shapes):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.ones(shapes, dtype=torch.bfloat16) * 1.3125
    torch_output_tensor = torch.reciprocal(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.reciprocal(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("tt", output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    print("torch", torch_output_tensor)

    assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.999


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_unary_composite_recip_ttnn(input_shapes, device):
    in_data1, input_tensor1 = data_gen_with_range_dtype(input_shapes, -3, 3, device, False, False, ttnn.bfloat8_b)

    output_tensor = ttnn.reciprocal(input_tensor1)
    output_tensor_rm = output_tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    golden_function = ttnn.get_golden_function(ttnn.reciprocal)
    golden_tensor = golden_function(in_data1)

    # for i in range(1):            # Batch size
    #     for j in range(1):        # Channels
    #         for k in range(32):   # Height
    #             for l in range(32):  # Width
    #                 print(f"input: {in_data1[i][j][k][l]} \t tt: {output_tensor_rm[i][j][k][l]} \t torch: {golden_tensor[i][j][k][l]} \n")

    comp_pass = compare_pcc([output_tensor], [golden_tensor], pcc=0.99)
    assert comp_pass
