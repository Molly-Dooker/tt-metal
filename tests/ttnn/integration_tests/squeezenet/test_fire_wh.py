import ttnn
import pytest
import torch
from models.demos.wormhole.squeezenet.tt.tt_squeezenet import tt_Fire
from torchvision import models
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import is_wormhole_b0, skip_for_grayskull
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.squeezenet.tt.tmp_sq import Fire


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        parameters["bias"] = ttnn.from_torch(
            model.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
    return parameters


@pytest.mark.parametrize(
    "batch_size, input_height, input_width, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, features_block",
    [
        (2, 54, 54, 96, 16, 64, 64, 3),
        # (2, 54, 54, 128, 16, 64, 64, 4),
        # (2, 54, 54, 128, 32, 128, 128, 5),
        # (2, 27, 27, 256, 32, 128, 128, 7),
        # (2, 27, 27, 256, 48, 192, 192, 8),
        # (2, 27, 27, 384, 48, 192, 192, 9),
        # (2, 27, 27, 384, 64, 256, 256, 10),
        # (2, 13, 13, 512, 64, 256, 256, 12),
    ],
)
@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_fire(
    mesh_device,
    batch_size,
    input_height,
    input_width,
    inplanes,
    squeeze_planes,
    expand1x1_planes,
    expand3x3_planes,
    features_block,
):
    inputs_mesh_mapper = None
    weights_mesh_mapper = None
    output_mesh_composer = None
    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    state_dict = torch_squeezenet.state_dict()
    torch.manual_seed(42)
    torch_input = torch.randn([batch_size, input_height, input_width, inplanes])
    torch_input_for_premodel = torch.permute(torch_input, (0, 3, 1, 2))
    ref_sq = Fire(96, 16, 64, 64, state_dict=state_dict, input=torch_input_for_premodel, layer_idx=features_block)
    if is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        with ttnn.distribute(weights_mesh_mapper):
            parameters = preprocess_model_parameters(
                initialize_model=lambda: torch_squeezenet.features[features_block],
                custom_preprocessor=custom_preprocessor,
                # device=mesh_device,
            )
    else:
        print("2 devices are not detected")

    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=inputs_mesh_mapper
    )
    tt_out = tt_Fire(
        inplanes,
        squeeze_planes,
        expand1x1_planes,
        expand3x3_planes,
        input_tensor=tt_input,
        parameters=parameters,
        mesh_device=mesh_device,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        state_dict=state_dict,
        num=features_block,
    )

    # l1 = torch.load("/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/dumps/torch_out.pth")
    # l1 = l1.permute(0,2,3,1)
    tt_out_in_torch = ttnn.to_torch(tt_out, mesh_composer=output_mesh_composer).permute(0, 3, 1, 2)
    # tt_out_in_torch = torch.permute(tt_out_in_torch, (0, 3, 1, 2))
    torch_model = torch_squeezenet.features[features_block]
    torch_out = torch_model(torch_input_for_premodel)
    # torch_out = torch_out.permute(0,3,2,1)
    # l1 = torch.load("/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/dumps/torch_out_3.pth")
    # l2= torch.load("/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/dumps/tt_out_3.pth")
    # l1=l1.permute(0,3,2,1)
    # tt_out_in_torch = tt_out_in_torch.permute(0,2,3,1)
    # tt_out_in_torch = torch.reshape(tt_out_in_torch,(2,54,54,128))
    # l2 = torch.reshape(l2,((2,54,54,16)))
    # l2 = l2.permute(0,3,1,2)
    # print(l1.shape,l2.shape)
    # assert_with_pcc(l1, l2, pcc=0.99)
    # print(torch_out.shape,tt_out_in_torch.shape)
    assert_with_pcc(torch_out, tt_out_in_torch, pcc=0.99)
