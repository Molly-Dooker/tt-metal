# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_petr.tt.common import Conv
import torch


class ttnn_hsigmoid:
    def __init__(self, device, inplace=True):
        self.inplace = inplace

    def __call__(self, x):
        x = x + 3.0
        x = ttnn.relu6(x)
        x = ttnn.div(x, 6.0)
        return x


class ttnn_esemodule:
    def __init__(self, parameters):
        self.avg_pool = ttnn.global_avg_pool2d
        self.fc = Conv([1, 1, 0, 0], parameters["fc"])
        # self.hsigmoid = Hsigmoid()

    def __call__(self, device, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(device, x)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.div(ttnn.relu6(x + 3.0), 6.0)  # Hsigmoid()
        return input * x


class ttnn_osa_module:
    def __init__(
        self,
        parameters,
        in_ch,
        stage_ch,
        concat_ch,
        layer_per_block,
        module_name,
        SE=False,
        identity=False,
        depthwise=False,
        with_cp=True,
    ):
        self.identity = identity
        self.depthwise = depthwise
        self.isReduced = False
        self.parameters = parameters

        self.layers = []
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
            self.conv_reduction = Conv(
                [1, 1, 0, 0], parameters["{}_reduction_0".format(module_name)], activation="relu"
            )
        for i in range(layer_per_block):
            if self.depthwise:
                self.layers.append(Conv([1, 1, 1, 1], parameters["{}_{}".format(module_name, i)], activation="relu"))
            else:
                self.layers.append(
                    Conv([1, 1, 1, 1], parameters["{}_{}".format(module_name, i)], activation="relu", act_block_h=32)
                )

        self.concat = Conv([1, 1, 0, 0], parameters["{}_{}".format(module_name, "concat")], activation="relu")
        print("concat parameters: ", parameters["{}_{}".format(module_name, "concat")])
        self.ese = ttnn_esemodule(parameters)

    def __call__(self, device, x):
        identity_feat = x

        output = []
        output.append(x)
        if self.depthwise and self.isReduced:
            x = self.conv_reduction(x)
        for layer in self.layers:
            print("layer:", layer)
            x = layer(device, x)
            output.append(x)
        x = ttnn.concat(output, dim=3)  # PCC = 0.99
        for y in output:
            ttnn.deallocate(y)
        # x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        xt = self.concat(device, x)  # PCC = 0.062789802576302
        # print(x.shape)
        # x = ttnn.to_torch(x)
        # x = torch.reshape(x, (1, 80, 200, 768))
        # x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        # torch.save(x, "concat_input.pt")
        # xt = ttnn.to_layout(xt, ttnn.TILE_LAYOUT)
        # xt = self.ese(device, xt)

        # if self.identity:
        #     xt = xt + identity_feat

        return xt  # PCC = 0.0626915480243774
