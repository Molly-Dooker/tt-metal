# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def test_reshape_case_1(reset_seeds, mesh_device):
    input_tensor = torch.rand((128, 16, 5, 5), dtype=torch.bfloat16)  # NCHW
    x = torch.reshape(input_tensor, (input_tensor.shape[0], -1))

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    input_tensor = ttnn.from_torch(
        input_tensor,
        mesh_mapper=inputs_mesh_mapper,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y = ttnn.reshape(input_tensor, (input_tensor.shape[0], -1))

    y = ttnn.to_torch(y, mesh_composer=output_mesh_composer)

    assert_with_pcc(x, y, 1)  # pcc --0.0015663878122013654


def test_reshape_case_2(reset_seeds, mesh_device):
    input_tensor = torch.rand((128, 16, 5, 5), dtype=torch.bfloat16)  # NCHW
    x = torch.reshape(input_tensor, (input_tensor.shape[0], -1))

    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)

    input_tensor = ttnn.from_torch(
        input_tensor,
        mesh_mapper=inputs_mesh_mapper,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y = ttnn.reshape(input_tensor, (input_tensor.shape[0], -1))

    y = ttnn.from_device(y)
    y = ttnn.to_device(y, device=mesh_device)

    y = ttnn.to_layout(y, layout=ttnn.TILE_LAYOUT)

    y = ttnn.to_torch(y, mesh_composer=output_mesh_composer)

    assert_with_pcc(x, y, 1)  # pcc = 1git status
