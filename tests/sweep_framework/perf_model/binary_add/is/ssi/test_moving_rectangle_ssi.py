import pytest
import torch
import ttnn
import csv
from tests.ttnn.utils_for_testing import assert_with_pcc


def write_to_csv(shard_x, shard_y, start_x, start_y):
    # Open the CSV file in append mode to add the data
    with open(
        "tests/sweep_framework/perf_model/binary_add/is/ssi/sharding_configurations.csv", mode="a", newline=""
    ) as file:
        writer = csv.writer(file)
        # Write a row with shard_x, shard_y, grid_x, grid_y
        writer.writerow([shard_x, shard_y, start_x, start_y])


def run_test(
    device,
    torch_input_tensor_a,
    torch_input_tensor_b,
    out_mem_config,
    shard_type,
    dims,
    shard_x,
    shard_y,
    input_dtype,
    start_x,
    start_y,
):
    if input_dtype == ttnn.bfloat16:
        size = 2
    if input_dtype == ttnn.bfloat4_b:
        size = 0.5
    if input_dtype == ttnn.bfloat8_b:
        size = 1
    if dims[0] * dims[1] / shard_x / shard_y * size >= 400000:
        return

    # Before the ttnn.add call, write the shard_x, shard_y, grid_x, grid_y to a CSV file
    print(str(start_x) + " " + str(start_y))
    write_to_csv(shard_x, shard_y, start_x, start_y)

    sharded_mem_config = ttnn.create_sharded_memory_config_(
        dims,
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(start_x, start_y),
                    ttnn.CoreCoord(start_x + shard_x - 1, start_y + shard_y - 1),
                )
            }
        ),
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=input_dtype,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=input_dtype,
    )

    output = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)


@pytest.mark.parametrize(
    "dims", [(64 * 32, 64 * 32), (32 * 32, 32 * 32), (16 * 32, 16 * 32), (8 * 32, 8 * 32), (4 * 32, 4 * 32)]
)
@pytest.mark.parametrize("out_mem_config", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("in_mem_config", ["block"])
def test_add_with_block_sharding(device, dims, out_mem_config, in_mem_config):
    torch.manual_seed(0)
    h = dims[0]
    w = dims[1]
    input_dtype = ttnn.bfloat16

    iterations = 1
    i = 0
    for _ in range(iterations):  # Loop over the number of iterations
        h = dims[0]
        w = dims[1]
        torch_input_tensor_a = torch.rand((h, w))
        torch_input_tensor_b = torch.rand((h, w))

        for i in range(7):
            if w >= 8 * 32:
                run_test(
                    device,
                    torch_input_tensor_a,
                    torch_input_tensor_b,
                    out_mem_config,
                    "block",
                    dims,
                    2,
                    8,
                    input_dtype,
                    i,
                    0,
                )

        for i in range(7):
            if h >= 8 * 32:
                run_test(
                    device,
                    torch_input_tensor_a,
                    torch_input_tensor_b,
                    out_mem_config,
                    "block",
                    dims,
                    8,
                    2,
                    input_dtype,
                    0,
                    i,
                )

        for i in range(5):
            for j in range(5):
                run_test(
                    device,
                    torch_input_tensor_a,
                    torch_input_tensor_b,
                    out_mem_config,
                    "block",
                    dims,
                    4,
                    4,
                    input_dtype,
                    i,
                    j,
                )
