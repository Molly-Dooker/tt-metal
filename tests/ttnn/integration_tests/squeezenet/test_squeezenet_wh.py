# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
import torch
import torch.nn as nn
from models.demos.wormhole.squeezenet.tt.tt_squeezenet import tt_squeezenet
from torchvision import models
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0, skip_for_grayskull
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.squeezenet.tt.tmp_sq import squeezenet


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        parameters["bias"] = ttnn.from_torch(
            model.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    return parameters


@pytest.mark.parametrize(
    "batch_size,input_height,input_width,conv_1_params, conv_2_params",
    [
        (2, 224, 224, [3, 96], [512, 1000]),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_model(batch_size, input_height, input_width, conv_1_params, conv_2_params, mesh_device):
    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    state_dict = torch_squeezenet.state_dict()
    torch_input = torch.randn(batch_size, input_height, input_width, conv_1_params[0])
    torch_input_for_premodel = torch.permute(torch_input, (0, 3, 1, 2))
    inputs_mesh_mapper = None
    weights_mesh_mapper = None
    output_mesh_composer = None
    if is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        with ttnn.distribute(weights_mesh_mapper):
            parameters = preprocess_model_parameters(
                initialize_model=lambda: torch_squeezenet,
                custom_preprocessor=custom_preprocessor,
                # device=mesh_device,
            )
    else:
        print("2 devices are not detected")
    ref_sq = squeezenet(state_dict=state_dict, input=torch_input_for_premodel)
    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper)
    tt_out = tt_squeezenet(
        mesh_device,
        parameters,
        tt_input,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )
    torch_out = torch_squeezenet(torch_input_for_premodel)
    # print("shapes are",torch_out.shape,tt_out.shape)
    tt_out_in_torch = ttnn.to_torch(tt_out, mesh_composer=output_mesh_composer)
    # print("final shapes,",tt_out_in_torch.shape,torch_out.shape)
    p1, p2 = assert_with_pcc(torch_out, tt_out_in_torch, pcc=0.99)
    print(p1, p2)
