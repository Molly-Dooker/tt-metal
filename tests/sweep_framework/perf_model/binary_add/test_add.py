import pytest
import torch
import ttnn


@pytest.mark.parametrize("dims", [(32, 32), (64, 64)])
def test_add_with_block_sharding(device, dims):
    torch.manual_seed(0)
    h = dims[0]
    w = dims[1]
    torch_input_tensor_a = torch.rand((h, w))
    torch_input_tensor_b = torch.rand((h, w))

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

    output = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
