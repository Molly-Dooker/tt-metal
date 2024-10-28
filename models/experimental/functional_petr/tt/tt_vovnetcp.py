# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_petr.tt.common import Conv


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

        self.layers = []
        in_channel = in_ch
        if self.depthwise and in_channel != stage_ch:
            self.isReduced = True
