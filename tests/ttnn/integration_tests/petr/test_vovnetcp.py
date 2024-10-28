# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP, Hsigmoid, eSEModule, _OSA_module
from models.experimental.functional_petr.tt.tt_vovnetcp import ttnn_hsigmoid, ttnn_esemodule
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnetcp(device, reset_seeds):
    input_tensor = torch.randn((6, 3, 320, 800))
    torch_model = VoVNetCP("V-99-eSE", out_features=["stage4", "stage5"])

    output = torch_model(input_tensor)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_vovnetcp_hsigmoid(device, reset_seeds):
    input_tensor = torch.randn((6, 256, 1, 1))
    torch_model = Hsigmoid()
    torch_output = torch_model(input_tensor)

    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_model = ttnn_hsigmoid(device)
    ttnn_output = ttnn_model(ttnn_input)

    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, eSEModule):
            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(torch.reshape(model.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
        # if isinstance(model, _OSA_module):

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_esemodule(device):
    torch_input_tensor = torch.randn(6, 256, 80, 200)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    torch_model = eSEModule(256)
    torch_model.eval()
    print(torch_model)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    print("parameters: ", parameters)

    torch_output = torch_model(torch_input_tensor)
    ttnn_model = ttnn_esemodule(parameters)

    ttnn_output = ttnn_model(device=device, x=ttnn_input_tensor)
    ttnn_output = ttnn.to_torch(ttnn_output)
    ttnn_output = ttnn_output.permute(0, 3, 1, 2)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vovnetcp_osa_module(device):
    torch_input_tensor = torch.randn(6, 128, 80, 200)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    torch_model = _OSA_module(128, 128, 256, 5, "OSA2_1", SE=True)
    torch_model.eval()
    print("Model: ", torch_model)

    torch_output = torch_model(torch_input_tensor)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    print("parameters: ", parameters)
