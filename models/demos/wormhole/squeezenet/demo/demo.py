# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn
from torchvision import models
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    profiler,
)
from models.demos.wormhole.squeezenet.demo_utils import get_data_loader, get_batch, preprocess
from loguru import logger
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_model_parameters
from models.demos.wormhole.squeezenet.tt.tt_squeezenet import tt_squeezenet


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16)
        parameters["bias"] = ttnn.from_torch(model.bias.unsqueeze(0).unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16)

    return parameters


def run_squeezenet_imagenet_inference(
    batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    profiler.clear()

    torch_squeezenet = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
    torch_squeezenet.to(torch.bfloat16)
    torch_squeezenet.eval()
    inputs_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    weights_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    output_mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    with ttnn.distribute(ttnn.ReplicateTensorToMesh(mesh_device)):
        parameters = preprocess_model_parameters(
            initialize_model=lambda: torch_squeezenet, custom_preprocessor=custom_preprocessor, device=None
        )
    logger.info("ImageNet-1k validation Dataset")
    input_loc = str(model_location_generator("ImageNet_data"))
    data_loader = get_data_loader(input_loc, batch_size, iterations)
    # load ImageNet batch by batch
    # and run inference
    correct = 0
    torch_ttnn_correct = 0
    torch_correct = 0
    for iter in range(iterations):
        predictions = []
        torch_predictions = []
        inputs, labels = get_batch(data_loader)
        torch_outputs = torch_squeezenet(inputs)
        tt_batched_input_tensor = ttnn.from_torch(
            inputs.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            mesh_mapper=inputs_mesh_mapper,
        )
        tt_output = tt_squeezenet(
            mesh_device=mesh_device,
            parameters=parameters,
            tt_input=tt_batched_input_tensor,
            mesh_mapper=inputs_mesh_mapper,
            mesh_composer=output_mesh_composer,
        )
        tt_output = ttnn.to_torch(tt_output, mesh_composer=output_mesh_composer)
        prediction = tt_output.argmax(dim=-1)
        torch_prediction = torch_outputs[0].argmax(dim=-1)
        print("torch and tt predictions", torch_prediction, prediction)
        for i in range(batch_size):
            if prediction.dim() == 0:  # scalar (0-dimensional tensor)
                predictions.append(imagenet_label_dict[prediction.item()])
            else:  # batch size > 1, so it's a 1-dimensional tensor
                predictions.append(imagenet_label_dict[prediction[i].item()])

            if torch_prediction.dim() == 0:  # scalar (0-dimensional tensor)
                torch_predictions.append(imagenet_label_dict[torch_prediction.item()])
            else:  # batch size > 1, so it's a 1-dimensional tensor
                torch_predictions.append(imagenet_label_dict[torch_prediction[i].item()])

            logger.info(
                f"Iter: {iter} Sample: {i} - Expected Label: {imagenet_label_dict[labels[i]]} -- \n Torch Predicted label:{torch_predictions[-1]} \tPredicted Label: {predictions[-1]}"
            )
            if imagenet_label_dict[labels[i]] == predictions[-1]:
                correct += 1
            if imagenet_label_dict[labels[i]] == torch_predictions[-1]:
                torch_correct += 1
            if predictions[-1] == torch_predictions[-1]:
                torch_ttnn_correct += 1

        del tt_output, tt_batched_input_tensor, inputs, labels, predictions
    print("torch and ttnn correct, both correct", torch_correct, correct, torch_ttnn_correct)
    accuracy = correct / (batch_size * iterations)
    torch_accuracy = torch_correct / (batch_size * iterations)
    torch_ttnn_accuracy = torch_ttnn_correct / (batch_size * iterations)

    logger.info(f"Model SqueezeNet for Image Classification")
    logger.info(f"TTNN Accuracy for {batch_size}x{iterations} inputs: {accuracy}")
    logger.info(f"Torch Accuracy for {batch_size}x{iterations} inputs: {torch_accuracy}")
    logger.info(f"Torch vs TTNN Accuracy for {batch_size}x{iterations} inputs: {torch_ttnn_accuracy}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, iterations",
    ((2, 10),),
)
def test_demo_dataset(batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device):
    return run_squeezenet_imagenet_inference(
        batch_size, iterations, imagenet_label_dict, model_location_generator, mesh_device
    )
