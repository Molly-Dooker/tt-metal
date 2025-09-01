import pytest
import torch
import ttnn
import ipdb
from tests.ttnn.utils_for_testing import assert_with_pcc
import torchvision
from typing import Tuple
import torch
import torch.nn as nn
import math
from pathlib import Path
import struct, zlib
from typing import Optional
from collections import OrderedDict
from tqdm import tqdm
import os

RESNET50_LAYERS = [
    "conv1",
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer1.0.conv3",
    "layer1.0.downsample.0",
    "layer1.1.conv1",
    "layer1.1.conv2",
    "layer1.1.conv3",
    "layer1.2.conv1",
    "layer1.2.conv2",
    "layer1.2.conv3",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.0.conv3",
    "layer2.0.downsample.0",
    "layer2.1.conv1",
    "layer2.1.conv2",
    "layer2.1.conv3",
    "layer2.2.conv1",
    "layer2.2.conv2",
    "layer2.2.conv3",
    "layer2.3.conv1",
    "layer2.3.conv2",
    "layer2.3.conv3",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.0.conv3",
    "layer3.0.downsample.0",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer3.1.conv3",
    "layer3.2.conv1",
    "layer3.2.conv2",
    "layer3.2.conv3",
    "layer3.3.conv1",
    "layer3.3.conv2",
    "layer3.3.conv3",
    "layer3.4.conv1",
    "layer3.4.conv2",
    "layer3.4.conv3",
    "layer3.5.conv1",
    "layer3.5.conv2",
    "layer3.5.conv3",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.0.conv3",
    "layer4.0.downsample.0",
    "layer4.1.conv1",
    "layer4.1.conv2",
    "layer4.1.conv3",
    "layer4.2.conv1",
    "layer4.2.conv2",
    "layer4.2.conv3",
    "fc",
]


@torch.no_grad()
def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    # 새 Conv(bias=True)로 생성 (원 Conv의 하이퍼파라미터 보존)
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

    # 파라미터 가져오기
    W = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=W.device, dtype=W.dtype)

    gamma = bn.weight if bn.affine else torch.ones_like(bn.running_var)
    beta = bn.bias if bn.affine else torch.zeros_like(bn.running_mean)
    mu = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    s = gamma / torch.sqrt(var + eps)  # (out_channels,)
    W_fused = W * s.view(-1, 1, 1, 1)  # broadcast
    b_fused = (b - mu) * s + beta  # (out_channels,)

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
    """parent 하위의 (name1, name2)가 (Conv2d, BN2d) 또는 (Linear, BN1d)이면 fuse"""
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
def fold_batchnorm_(module: nn.Module):
    """
    모듈 트리를 재귀적으로 순회하며 Conv2d+BN2d / Linear+BN1d를 접습니다.
    inplace로 변경됩니다. (BN은 Identity로 대체)
    """
    children = list(module.named_children())  # 유지되는 순서 중요
    for i in range(len(children) - 1):
        name1, m1 = children[i]
        name2, m2 = children[i + 1]
        _maybe_fuse_pair(module, (name1, name2))
    for _, child in module.named_children():
        fold_batchnorm_(child)


def _reshape_scale_for_broadcast(scale: torch.Tensor, x: torch.Tensor, channel_axis: int):
    # 음수 축도 허용
    if channel_axis < 0:
        channel_axis += x.ndim
    shape = [1] * x.ndim
    shape[channel_axis] = -1
    return scale.view(*shape)


@pytest.mark.parametrize(
    "layer",
    RESNET50_LAYERS,
)
def test_quantize(layer, device):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    fold_batchnorm_(model)
    m = model.get_submodule(layer)
    weight = m.weight.detach().clone()
    if isinstance(m, (nn.Conv2d)):
        weight = m.weight.permute(2, 3, 1, 0).reshape(1, 1, -1, m.weight.size(0))
    elif isinstance(m, (nn.Linear)):
        weight = m.weight.permute(1, 0)
    if isinstance(m, nn.Conv2d):
        per_channel_absmax = weight.abs().amax(dim=(0, 1, 2))
    else:  # Linear
        per_channel_absmax = weight.abs().amax(dim=0)
    scale = torch.clamp(per_channel_absmax, 1e-12) / 127
    scale = _reshape_scale_for_broadcast(scale, weight, -1)
    q_torch = torch.round(weight / scale).clamp(-127, 127).to(torch.bfloat16)
    # qdq = q_torch * scale
    q_ttnn = ttnn.from_torch(
        q_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q_ttnn_back = ttnn.to_torch(q_ttnn)
    assert torch.equal(q_ttnn_back, q_torch)


@pytest.mark.parametrize(
    "layer",
    RESNET50_LAYERS,
)
def test_dequantize(layer, device):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    fold_batchnorm_(model)
    m = model.get_submodule(layer)
    weight = m.weight.detach().clone()
    if isinstance(m, (nn.Conv2d)):
        weight = m.weight.permute(2, 3, 1, 0).reshape(1, 1, -1, m.weight.size(0))
    elif isinstance(m, (nn.Linear)):
        weight = m.weight.permute(1, 0)
    if isinstance(m, nn.Conv2d):
        per_channel_absmax = weight.abs().amax(dim=(0, 1, 2))
    else:  # Linear
        per_channel_absmax = weight.abs().amax(dim=0)
    scale = torch.clamp(per_channel_absmax, 1e-12) / 127
    scale = _reshape_scale_for_broadcast(scale, weight, -1)
    q_torch = torch.round(weight / scale).clamp(-127, 127).to(torch.bfloat16)
    qdq_torch = q_torch * scale
    q_ttnn = ttnn.from_torch(
        q_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale_ = scale.squeeze()
    zp_ = (scale_ * 0).to(torch.int32)
    scale_tt = ttnn.from_torch(scale_, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    zp_tt = ttnn.from_torch(zp_, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    q_ttnn_int32 = ttnn.typecast(q_ttnn, ttnn.int32)
    qdq_ttnn = ttnn.dequantize(q_ttnn_int32, scale_tt, zp_tt, axis=-1, dtype=ttnn.bfloat16)
    qdq_torch_recon = ttnn.to_torch(qdq_ttnn)
    assert_with_pcc(qdq_torch, qdq_torch_recon)


def test_check_size(device):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    fold_batchnorm_(model)
    fp32 = 0
    bfp8b = 0
    for name, m in model.named_modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        if isinstance(m, (nn.Conv2d)):
            weight = m.weight.permute(2, 3, 1, 0).reshape(1, 1, -1, m.weight.size(0))
        elif isinstance(m, (nn.Linear)):
            weight = m.weight.permute(1, 0)
        h, w = weight.shape[-2], weight.shape[-1]
        h_pad = math.ceil(h / 32) * 32
        w_pad = math.ceil(w / 32) * 32
        pad_h = h_pad - h
        pad_w = w_pad - w
        padded = torch.nn.functional.pad(weight, (0, pad_w, 0, pad_h))
        print(
            f"{name:24} | weight:{weight.numel()*4/1024/1024:.2f}MB->{padded.numel()*8.5/8/1024/1024:.2f}MB | shape:{weight.shape}->{padded.shape}"
        )
        fp32 += weight.numel() * 4 / 1024 / 1024
        bfp8b += padded.numel() * 8.5 / 8 / 1024 / 1024
    print(f"weight size: {fp32:.4f}MB -> {bfp8b:.4f}MB  ratio : {bfp8b/fp32:.4f}")


@pytest.mark.parametrize(
    "dtype",
    [ttnn.float32, ttnn.bfloat8_b],
)
def test_flatbuffer(device, dtype):
    savepath = "dump/"
    os.makedirs(savepath, exist_ok=True)
    savepath = savepath + f"{dtype.name}.bin"
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    fold_batchnorm_(model)
    statedict_torch = OrderedDict()
    statedict = OrderedDict()
    for name, value in model.state_dict().items():
        scale = None
        if "weight" in name:
            value = (
                value.permute(2, 3, 1, 0).reshape(1, 1, -1, value.size(0)) if "fc" not in name else value.permute(1, 0)
            )
            # calculate scale
            per_channel_absmax = value.abs().amax(dim=tuple(range(value.ndim - 1)))
            scale = torch.clamp(per_channel_absmax, 1e-12) / 127
            if dtype == ttnn.float32:
                value_ttnn = ttnn.from_torch(
                    value,
                    dtype=dtype,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            elif dtype == ttnn.bfloat8_b:
                scale_ = _reshape_scale_for_broadcast(scale, value, -1)
                value = torch.round(value / scale_).clamp(-127, 127).to(torch.bfloat16)
                value_ttnn = ttnn.from_torch(
                    value,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat8_b,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
        else:  # bias
            value_ttnn = ttnn.from_torch(
                value,
                dtype=ttnn.float32,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        statedict[name] = value_ttnn
        statedict_torch[name] = value
        if scale is not None:
            statedict[name[: -len(".weight")] + ".scale"] = ttnn.from_torch(
                scale, dtype=ttnn.float32, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            statedict_torch[name[: -len(".weight")] + ".scale"] = scale
    ttnn.save_ttnn_state(savepath, statedict)
    loaded_statedict = ttnn.load_ttnn_state(savepath, device=device)
    for key, value in tqdm(loaded_statedict.items()):
        value_recon = ttnn.to_torch(value) if isinstance(value, ttnn.Tensor) else value
        value_torch = statedict_torch[key]
        assert_with_pcc(value_torch, value_recon)
