import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from functools import partial
from models.utility_functions import torch_random
import argparse
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_test(input_tensor_a, input_tensor_b, out_mem_config):
    output = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)


@pytest.mark.parametrize(
    "dims",
    [
        (32, 5 * 32),  # 5
        (10 * 32, 2 * 32),  # 20
        (10 * 32, 20 * 32),  # 200
        (100 * 32, 5 * 32),  # 500
        (100 * 32, 30 * 32),  # 3k
        (100 * 32, 40 * 32),  # 4k
        (100 * 32, 70 * 32),  # 7k
        (100 * 32, 90 * 32),  # 9k
        (11 * 32, 1000 * 32),  # 11k
    ],
)
@pytest.mark.parametrize("input_a_mem_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("input_b_mem_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("out_mem_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16])
# @pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT])
def test_relu_interleaved(device, dims, input_a_mem_config, input_b_mem_config, out_mem_config, input_dtype):
    for i in range(10):
        h = dims[0]
        w = dims[1]
        torch_input_tensor_a = torch.rand((h, w))
        torch_input_tensor_b = torch.rand((h, w))

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            memory_config=input_a_mem_config,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=input_dtype,
        )

        input_tensor_b = ttnn.from_torch(
            torch_input_tensor_b,
            memory_config=input_b_mem_config,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=input_dtype,
        )

        run_test(input_tensor_a, input_tensor_b, out_mem_config)
