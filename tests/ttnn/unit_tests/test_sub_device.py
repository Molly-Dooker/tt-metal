# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_sub_devices(device, enable_async_mode):
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    sub_device_1 = ttnn.SubDevice([tensix_cores0])
    sub_device_2 = ttnn.SubDevice([tensix_cores1])
    sub_device_manager1 = ttnn.CreateSubDeviceManager(device, [sub_device_1, sub_device_2], 3200)
    sub_device_manager2 = ttnn.CreateSubDeviceManager(device, [sub_device_2], 3200)
    ttnn.LoadSubDeviceManager(device, sub_device_manager1)
    ttnn.LoadSubDeviceManager(device, sub_device_manager2)
    ttnn.ClearLoadedSubDeviceManager(device)
    ttnn.RemoveSubDeviceManager(device, sub_device_manager1)
    ttnn.RemoveSubDeviceManager(device, sub_device_manager2)


@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_sub_device_program(device, enable_async_mode):
    tensix_cores0 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(3, 3),
            ),
        }
    )
    tensix_cores1 = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(4, 4),
                ttnn.CoreCoord(4, 4),
            ),
        }
    )
    sub_device_1 = ttnn.SubDevice([tensix_cores0])
    sub_device_2 = ttnn.SubDevice([tensix_cores1])
    sub_device_manager = ttnn.CreateSubDeviceManager(device, [sub_device_1, sub_device_2], 3200)
    ttnn.LoadSubDeviceManager(device, sub_device_manager)

    x = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    xt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    grid_size = device.compute_with_storage_grid_size()
    shard_size = [32, 64]
    shard_scheme = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    shard_orientation = ttnn.ShardOrientation.ROW_MAJOR
    yt = ttnn.interleaved_to_sharded(
        xt, grid_size, shard_size, shard_scheme, shard_orientation, output_dtype=ttnn.bfloat16
    )
    y = ttnn.to_torch(yt)

    eq = torch.equal(x, y)
    assert eq

    ttnn.ClearLoadedSubDeviceManager(device)
    ttnn.RemoveSubDeviceManager(device, sub_device_manager)
