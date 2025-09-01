# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import glob
import os
from typing import Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from torchvision import models
from tqdm import tqdm

from models.sample_data.huggingface_imagenet_classes import IMAGENET2012_CLASSES


@torch.no_grad()
def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    W = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=W.device, dtype=W.dtype)
    gamma = bn.weight if bn.affine else torch.ones_like(bn.running_var)
    beta = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    s = gamma / torch.sqrt(var + eps)
    W_fused = W * s.view(-1, 1, 1, 1)
    b_fused = (b - mu) * s + beta
    fused.weight.copy_(W_fused)
    fused.bias.copy_(b_fused)
    return fused


@torch.no_grad()
def _fuse_linear_bn(fc: nn.Linear, bn: nn.BatchNorm1d) -> nn.Linear:
    fused = nn.Linear(
        fc.in_features,
        fc.out_features,
        bias=True,
        device=fc.weight.device,
        dtype=fc.weight.dtype,
    )
    W = fc.weight
    b = fc.bias if fc.bias is not None else torch.zeros(fc.out_features, device=W.device, dtype=W.dtype)
    gamma = bn.weight if bn.affine else torch.ones_like(bn.running_var)
    beta = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    s = gamma / torch.sqrt(var + eps)  # (out_features,)
    W_fused = W * s.view(-1, 1)
    b_fused = (b - mu) * s + beta
    fused.weight.copy_(W_fused)
    fused.bias.copy_(b_fused)
    return fused


def _maybe_fuse_pair(parent: nn.Module, names: Tuple[str, str]):
    n1, n2 = names
    m1 = getattr(parent, n1)
    m2 = getattr(parent, n2)
    if isinstance(m1, nn.Conv2d) and isinstance(m2, nn.BatchNorm2d):
        fused = _fuse_conv_bn(m1, m2)
        setattr(parent, n1, fused)
        setattr(parent, n2, nn.Identity())
    elif isinstance(m1, nn.Linear) and isinstance(m2, nn.BatchNorm1d):
        fused = _fuse_linear_bn(m1, m2)
        setattr(parent, n1, fused)
        setattr(parent, n2, nn.Identity())


@torch.no_grad()
def _fold_batchnorm(module: nn.Module):
    children = list(module.named_children())
    for i in range(len(children) - 1):
        name1, m1 = children[i]
        name2, m2 = children[i + 1]
        _maybe_fuse_pair(module, (name1, name2))
    for _, child in module.named_children():
        _fold_batchnorm(child)


def _reshape_scale_for_broadcast(scale: torch.Tensor, x: torch.Tensor, channel_axis: int):
    if channel_axis < 0:
        channel_axis += x.ndim
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    return scale.view(*shape)


class InputExample(object):
    def __init__(self, image, label=None):
        self.image = image
        self.label = label


def get_input(image_path):
    img = Image.open(image_path)
    return img


def get_label(image_path):
    _, image_name = image_path.rsplit("/", 1)
    image_name_exact, _ = image_name.rsplit(".", 1)
    _, label_id = image_name_exact.rsplit("_", 1)
    label = list(IMAGENET2012_CLASSES).index(label_id)
    return label


def get_batch(data_loader, image_processor):
    loaded_images = next(data_loader)
    images = None
    labels = []
    for image in loaded_images:
        img = image.image
        labels.append(image.label)
        if img.mode == "L":
            img = img.convert(mode="RGB")
        img = image_processor(img, return_tensors="pt")
        img = img["pixel_values"]

        if images is None:
            images = img
        else:
            images = torch.cat((images, img), dim=0)
    return images, labels


def get_data_loader(input_loc, batch_size, iterations, download_entire_dataset=False):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = glob.glob(data_path)

    def loader():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=get_input(f1),
                    label=get_label(f1),
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    def loader_hf():
        examples = []
        for f1 in files:
            examples.append(
                InputExample(
                    image=f1["image"],
                    label=f1["label"],
                )
            )
            if len(examples) == batch_size:
                yield examples
                del examples
                examples = []

    if len(files) == 0:
        files_raw = iter(
            load_dataset("imagenet-1k", split="validation", use_auth_token=True, streaming=not download_entire_dataset)
        )
        files = []
        sample_count = batch_size * iterations
        for _ in tqdm(range(sample_count), desc="Loading samples"):
            files.append(next(files_raw))
        del files_raw
        return loader_hf()

    return loader()


def get_data(input_loc):
    img_dir = input_loc + "/"
    data_path = os.path.join(img_dir, "*G")
    files = sorted(glob.glob(data_path))
    examples = []
    for f1 in files:
        examples.append(
            InputExample(
                image=get_input(f1),
                label=get_label(f1),
            )
        )
    image_examples = examples

    return image_examples


def load_resnet50_model(model_location_generator):
    # TODO: Can generalize the version to an arg
    model_version = "IMAGENET1K_V1.pt"
    model_path = model_location_generator(model_version, model_subdir="ResNet50")
    if os.path.exists(model_path):
        torch_resnet50 = models.resnet50()
        torch_resnet50.load_state_dict(torch.load(model_path))
    else:
        torch_resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return torch_resnet50
