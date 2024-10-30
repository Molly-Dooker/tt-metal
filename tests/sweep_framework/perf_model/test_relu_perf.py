import pytest
import torch
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.generation_funcs import gen_func_with_cast_tt
from functools import partial
from models.utility_functions import torch_random
import argparse
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "dims",
    [
        (32, 32),  # 1
        (10 * 32, 32),  # 10
        (10 * 32, 10 * 32),  # 100
        (100 * 32, 10 * 32),  # 1k
        (100 * 32, 20 * 32),  # 2k
        (100 * 32, 50 * 32),  # 5k
        (100 * 32, 80 * 32),  # 8k
        (100 * 32, 100 * 32),  # 10k
        (32 * 500, 32 * 25),  # 12.5k
    ],
)
@pytest.mark.parametrize("input_mem_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("out_mem_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b])
# @pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT])
def test_relu_interleaved(device, dims, input_mem_config, out_mem_config, input_dtype):
    for i in range(10):
        h = dims[0]
        w = dims[1]
        torch_input_tensor_a = torch.rand((h, w))

        input_tensor_a = ttnn.from_torch(
            torch_input_tensor_a,
            memory_config=input_mem_config,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=input_dtype,
        )

        output = ttnn.relu(input_tensor_a, memory_config=out_mem_config)


# @pytest.mark.parametrize("dims", [(32, 32), (8 * 32, 32), (32 * 8, 32 * 8), (32 * 64, 32 * 8), (32 * 64, 32 * 64)])
# # @pytest.mark.parametrize("dims", [(32, 32)])
# @pytest.mark.parametrize("mem_config", ["height", "width", "block"])
# @pytest.mark.parametrize("mem_layout", [ttnn.TILE_LAYOUT])
# def test_relu_sharded(device, dims, mem_config, mem_layout):
#     iterations = 1
#     i = 0
#     for _ in range(iterations):  # Loop over the number of iterations
#         # print("Iteration: " + str(i))
#         # print(mem_layout == ttnn.TILE_LAYOUT)
#         i += 1
#         torch.manual_seed(0)
#         tensor_height = dims[0]
#         tensor_width = dims[1]
#         input_a_dtype = ttnn.bfloat16

#         torch_input_tensor_a = gen_func_with_cast_tt(
#             partial(torch_random, low=-100, high=100, dtype=torch.float32), input_a_dtype
#         )((tensor_height, tensor_width))
#         torch_output_tensor = torch.nn.functional.relu(torch_input_tensor_a)

#         shard_x = 1
#         shard_y = 1
#         if mem_config == "height":
#             if tensor_height >= 32 * 64:
#                 shard_x = 8
#                 shard_y = 8
#             if tensor_width >= 32 * 8:
#                 shard_x = 8
#             sharded_mem_config = ttnn.create_sharded_memory_config_(
#                 dims,
#                 (ttnn.CoreGrid(x=shard_x, y=shard_y)),
#                 ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
#                 ttnn.ShardOrientation.ROW_MAJOR,
#             )
#         if mem_config == "width":
#             if tensor_width >= 32 * 64:
#                 shard_y = 8
#                 shard_x = 8
#             elif tensor_width >= 32 * 8:
#                 shard_x = 8
#             sharded_mem_config = ttnn.create_sharded_memory_config_(
#                 dims,
#                 (ttnn.CoreGrid(x=shard_x, y=shard_y)),
#                 ttnn.TensorMemoryLayout.WIDTH_SHARDED,
#                 ttnn.ShardOrientation.ROW_MAJOR,
#             )
#         if mem_config == "block":
#             if tensor_height >= 32 * 8:
#                 shard_y = 8
#             if tensor_width >= 32 * 8:
#                 shard_x = 8
#             sharded_mem_config = ttnn.create_sharded_memory_config_(
#                 dims,
#                 (ttnn.CoreGrid(x=shard_x, y=shard_y)),
#                 ttnn.TensorMemoryLayout.BLOCK_SHARDED,
#                 ttnn.ShardOrientation.ROW_MAJOR,
#             )

#         # print(mem_layout)
#         input_tensor_a = ttnn.from_torch(
#             torch_input_tensor_a,
#             dtype=input_a_dtype,
#             layout=mem_layout,
#             device=device,
#             memory_config=sharded_mem_config,
#         )

#         result = ttnn.relu(input_tensor_a, memory_config=sharded_mem_config)
#         output_tensor = ttnn.to_torch(result)
