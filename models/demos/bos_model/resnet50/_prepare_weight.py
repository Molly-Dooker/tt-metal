from collections import OrderedDict
from pathlib import Path

import torch
import torchvision

from models.demos.ttnn_resnet.tests.demo_utils import _fold_batchnorm, _reshape_scale_for_broadcast

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).eval()
_fold_batchnorm(model)

state_dict = model.state_dict()
torch.save(state_dict, Path(__file__).parent / "fp32.bin")

state_dict_q = OrderedDict()
for key, value in state_dict.items():
    if "weight" in key:
        module_name = key[:-7]
        scale_key = module_name + ".scale"
        if "fc" in key:  # linear
            per_channel_absmax = value.abs().amax(dim=1)
        else:  # conv
            per_channel_absmax = value.abs().amax(dim=(1, 2, 3))
        scale = torch.clamp(per_channel_absmax, 1e-12) / 127
        scale = _reshape_scale_for_broadcast(scale, value, 0)
        w_q = torch.round(value / scale).clamp(-127, 127).to(torch.int8)
        state_dict_q[key] = w_q
        state_dict_q[scale_key] = scale
    elif "bias" in key:
        state_dict_q[key] = value
torch.save(state_dict_q, Path(__file__).parent / "int8.bin")
