# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_range_dtype,
)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_hypot(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -101, 100, device, True)
    other_data, other_tensor = data_gen_with_range(input_shapes, -52, 51, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -30, 30, device, True)

    tt_output_tensor_on_device = ttnn.hypot_bw(grad_tensor, input_tensor, other_tensor)

    golden_function = ttnn.get_golden_function(ttnn.hypot_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 1, 320, 384])),
        # (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_hypot_bf8b(input_shapes, device):
    in_data, input_tensor = data_gen_with_range_dtype(input_shapes, -3, 3, device, True, False, ttnn.bfloat8_b)
    other_data, other_tensor = data_gen_with_range_dtype(input_shapes, -3, 3, device, True, False, ttnn.bfloat8_b)
    grad_data, grad_tensor = data_gen_with_range_dtype(input_shapes, -3, 3, device, False, False, ttnn.bfloat8_b)

    tt_output_tensor_on_device = ttnn.hypot_bw(grad_tensor, input_tensor, other_tensor)
    output_tensor_rm1 = tt_output_tensor_on_device[0].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    output_tensor_rm2 = tt_output_tensor_on_device[1].cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
    golden_function = ttnn.get_golden_function(ttnn.hypot_bw)
    golden_tensor = golden_function(grad_data, in_data, other_data)
    print("tt1", output_tensor_rm1)
    print("tt2", output_tensor_rm2)
    print("torch", golden_tensor)
    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status
