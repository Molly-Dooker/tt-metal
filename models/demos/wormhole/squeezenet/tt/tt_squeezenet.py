# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn
import numpy as np
import torch


def conv(
    activation,
    input_tensor,
    weight,
    bias,
    in_channels,
    out_channels,
    mesh_device,
    kernel_size,
    padding,
    stride,
    in_h,
    in_w,
    shard_layout,
    h=1,
):
    reader_patterns_cache = {}
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        activation=activation,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        input_channels_alignment=(16 if False or (in_channels == 16 and input_tensor.shape[-2] == 115) else 32),
        shard_layout=shard_layout,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.TILE_LAYOUT,
    )

    [output_tt_tensor, out_height, out_width, weights, bias] = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=in_channels,
        out_channels=out_channels,
        device=mesh_device,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        batch_size=input_tensor.shape[0],
        input_height=in_h,
        input_width=in_w,
        conv_config=conv_config,
        groups=1,
        debug=False,
        conv_op_cache=reader_patterns_cache,
        memory_config=None,
    )
    reader_patterns_cache.clear()
    return output_tt_tensor, out_height, out_width


def tt_Fire(
    inplanes: int,
    squeeze_planes: int,
    expand1x1_planes: int,
    expand3x3_planes: int,
    input_tensor: ttnn.Tensor,
    parameters,
    mesh_device,
    mesh_mapper,
    mesh_composer,
    state_dict=None,
    dtype=ttnn.bfloat16,
    activation="relu",
    num=1,
):
    sweight = parameters.squeeze.weight
    sbias = parameters.squeeze.bias
    expand1x1_weights = parameters.expand1x1.weight
    expand1x1_bias = parameters.expand1x1.bias
    expand3x3_weights = parameters.expand3x3.weight
    expand3x3_bias = parameters.expand3x3.bias

    # conv1
    # print("input tensor info",input_tensor.shape,input_tensor.layout,input_tensor.memory_config())
    # ttnn.Shape([1, 54, 54, 96]) Layout.ROW_MAJOR MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)
    output_tt_tensor_1, out_height_1, out_width_1 = conv(
        activation="relu",
        input_tensor=input_tensor,
        weight=sweight,
        bias=sbias,
        in_channels=inplanes,
        out_channels=squeeze_planes,
        mesh_device=mesh_device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        in_h=input_tensor.shape[1],
        in_w=input_tensor.shape[2],
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )
    print(
        "shape COVN1", output_tt_tensor_1.shape, out_height_1, out_width_1
    )  # ttnn.Shape([1, 1, 2916[2944], 16[32]]) 54 54
    output_tt_tensor_1 = ttnn.sharded_to_interleaved(output_tt_tensor_1, ttnn.DRAM_MEMORY_CONFIG)
    output_tt_tensor_1 = ttnn.to_layout(output_tt_tensor_1, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tt_tensor_1 = ttnn.reshape(
        output_tt_tensor_1, (output_tt_tensor_1.shape[0], out_height_1, out_width_1, output_tt_tensor_1.shape[-1])
    )
    # ttnn.Shape([1, 54, 54, 16]) 54 54
    # torch.save(ttnn.to_torch(output_tt_tensor_1,mesh_composer=mesh_composer).permute(0,3,1,2),f"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/dumps/fg.pth")
    # conv2
    output_tt_tensor_2, out_height_2, out_width_2 = conv(
        activation="relu",
        input_tensor=output_tt_tensor_1,
        weight=expand1x1_weights,
        bias=expand1x1_bias,
        in_channels=squeeze_planes,
        out_channels=expand1x1_planes,
        mesh_device=mesh_device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        in_h=output_tt_tensor_1.shape[1],
        in_w=output_tt_tensor_1.shape[2],
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )
    print("shape conv2", output_tt_tensor_2.shape, out_height_2, out_width_2)
    output_tt_tensor_2 = ttnn.sharded_to_interleaved(output_tt_tensor_2, ttnn.DRAM_MEMORY_CONFIG)
    output_tt_tensor_2 = ttnn.to_layout(output_tt_tensor_2, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tt_tensor_2 = ttnn.reshape(
        output_tt_tensor_2, (output_tt_tensor_2.shape[0], out_height_2, out_width_2, output_tt_tensor_2.shape[-1])
    )
    # if num==8:
    #     torch.save(ttnn.to_torch(output_tt_tensor_2,mesh_composer=mesh_composer).permute(0,3,1,2),f"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/new_dumps/{num}/tt__before__output.pth")
    # conv3

    # torch covn3
    # if (num==8 or  num==9):

    # expand3x3_weights = ttnn.to_torch(expand3x3_weights, mesh_composer=mesh_composer)
    # expand3x3_bias = ttnn.to_torch(expand3x3_bias, mesh_composer=mesh_composer)
    # expand3x3_weights = expand3x3_weights[: (expand3x3_weights.shape[0] // 2), :, :, :]
    # expand3x3_bias = expand3x3_bias.squeeze().view(-1)
    # expand3x3_bias = expand3x3_bias[: (expand3x3_bias.shape[0] // 2),]
    #     torch.save(expand3x3_weights,f"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/new_dumps/{num}/weight.pth")
    #     torch.save(expand3x3_bias,f"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/new_dumps/{num}/bias.pth")
    #     torch.save(ttnn.to_torch(output_tt_tensor_1,mesh_composer=mesh_composer).permute(0,3,1,2),f"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/new_dumps/{num}/input.pth")
    # torch_conv_3 = torch.nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
    # torch_conv_3.weight = torch.nn.Parameter(expand3x3_weights)
    # torch_conv_3.weight.data = torch_conv_3.weight.data.to(torch.bfloat16)
    # torch_conv_3.bias = torch.nn.Parameter(expand3x3_bias)
    # torch_conv_3.bias.data = torch_conv_3.bias.data.to(torch.bfloat16)
    # output_tensor_1 = ttnn.to_torch(output_tt_tensor_1, mesh_composer=mesh_composer)
    # output_tensor_1 = output_tensor_1.permute(0, 3, 1, 2)
    # output_tensor_3 = torch_conv_3(output_tensor_1)
    # relu = torch.nn.ReLU(inplace=True)
    # output_tensor_3 = relu(output_tensor_3)
    # output_tt_tensor_3 = output_tensor_3.permute(0, 2, 3, 1)
    # output_tt_tensor_3 = ttnn.from_torch(
    #     output_tt_tensor_3, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=mesh_mapper
    # )
    # else:
    # output_tt_tensor_2 = ttnn.to_memory_config(output_tt_tensor_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print(
        "input,weights,bias",
        output_tt_tensor_2.memory_config(),
        expand3x3_weights.memory_config(),
        expand3x3_bias.memory_config(),
    )
    output_tt_tensor_3, out_height_3, out_width_3 = conv(
        activation="relu",
        input_tensor=output_tt_tensor_1,
        weight=expand3x3_weights,
        bias=expand3x3_bias,
        in_channels=squeeze_planes,
        out_channels=expand3x3_planes,
        mesh_device=mesh_device,
        kernel_size=(3, 3),
        padding=(1, 1),
        stride=(1, 1),
        in_h=output_tt_tensor_1.shape[1],
        in_w=output_tt_tensor_1.shape[2],
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    # output_tt_tensor_3 = ttnn.to_torch(output_tt_tensor_3,mesh_composer=mesh_composer).permute(0,3,1,2)
    # relu = torch.nn.ReLU(inplace=True)
    # output_tt_tensor_3 = relu(output_tt_tensor_3).permute(0,2,3,1)
    # output_tt_tensor_3 = ttnn.from_torch(output_tt_tensor_3,mesh_mapper=mesh_mapper,device=mesh_device)
    # if num==8 or num==9 :
    #     torch.save(ttnn.to_torch(output_tt_tensor_3,mesh_composer=mesh_composer).reshape(2,out_height_3,out_width_3,output_tt_tensor_3.shape[-1]).permute(0,3,1,2),f"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/new_dumps/{num}/tt_output.pth")
    # print("out conv3",output_tt_tensor_3.shape, out_height_3, out_width_3,output_tt_tensor_3.memory_config())
    output_tt_tensor_3 = ttnn.sharded_to_interleaved(output_tt_tensor_3, ttnn.DRAM_MEMORY_CONFIG)
    output_tt_tensor_3 = ttnn.to_layout(output_tt_tensor_3, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tt_tensor_3 = ttnn.reshape(
        output_tt_tensor_3, (output_tt_tensor_3.shape[0], out_height_3, out_width_3, output_tt_tensor_3.shape[-1])
    )

    # incase relu of ttnn conv fails
    # output_tt_tensor_3 = ttnn.to_torch(output_tt_tensor_3,mesh_composer=mesh_composer).permute(0,3,1,2)
    # relu = torch.nn.ReLU(inplace=True)
    # output_tt_tensor_3 = relu(output_tt_tensor_3).permute(0,2,3,1)

    # output_tt_tensor_3 = ttnn.from_torch(output_tt_tensor_3,mesh_mapper=mesh_mapper,device=mesh_device)
    # torch.save(ttnn.to_torch(output_tt_tensor_3,mesh_composer=mesh_composer).permute(0,3,1,2),"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/dumps/fg.pth")

    # concat

    # output_tt_tensor_2_c = ttnn.to_memory_config(output_tt_tensor_2, memory_config=ttnn.L1_MEMORY_CONFIG)
    # output_tt_tensor_3_c = ttnn.to_memory_config(output_tt_tensor_3, memory_config=ttnn.L1_MEMORY_CONFIG)
    # output_tt_tensor_2_c = ttnn.to_layout(output_tt_tensor_2, layout=ttnn.ROW_MAJOR_LAYOUT)
    # output_tt_tensor_3_c = ttnn.to_layout(output_tt_tensor_3_c, layout=ttnn.ROW_MAJOR_LAYOUT)
    print("before", output_tt_tensor_2.shape, output_tt_tensor_3.shape)
    final_output = ttnn.concat([output_tt_tensor_2, output_tt_tensor_3], dim=3)

    return final_output


def tt_squeezenet(
    mesh_device,
    parameters,
    tt_input,
    mesh_mapper,
    mesh_composer,
    dtype=ttnn.bfloat16,
    activation="relu",
    num_classes=1000,
):
    max_pool_in_tt = False  # ceilmode issue
    conv_1_weights = parameters.features[0].weight
    conv_1_bias = parameters.features[0].bias
    batch_size_tt = tt_input.shape[0]
    total_batch_size = 2 * batch_size_tt
    print("info of input", tt_input.shape, tt_input.layout, tt_input.memory_config())
    output_tt, out_height_1, out_width_1 = conv(
        activation="relu",
        input_tensor=tt_input,
        weight=conv_1_weights,
        bias=conv_1_bias,
        in_channels=conv_1_weights.shape[1],
        out_channels=conv_1_weights.shape[0],
        mesh_device=mesh_device,
        kernel_size=(conv_1_weights.shape[2], conv_1_weights.shape[3]),
        padding=(0, 0),
        stride=(2, 2),
        in_h=tt_input.shape[1],
        in_w=tt_input.shape[2],
        shard_layout=None,
    )

    output_tt = ttnn.to_layout(output_tt, layout=ttnn.ROW_MAJOR_LAYOUT)  # 1,1,h*w*n,c

    output_tt_reshaped = ttnn.reshape(
        output_tt, (1, 1, (output_tt.shape[0] * output_tt.shape[1] * output_tt.shape[2]), output_tt.shape[3])
    )
    print("output shape of conv1", output_tt.shape)

    max_pool_in_tt_1 = False  # PCC Drops once tt_maxpool is enabled and this drop persists after it is disabled
    if max_pool_in_tt_1:
        out_pool = ttnn.max_pool2d(
            input_tensor=output_tt,
            batch_size=batch_size_tt,
            input_h=out_height_1,
            input_w=out_width_1,
            channels=output_tt.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=None,
            applied_shard_scheme=None,
        )
        out_pool_reshaped = ttnn.reshape(out_pool, (batch_size_tt, 54, 54, out_pool.shape[3]))
        tt_input_2 = ttnn.from_device(out_pool_reshaped)
    else:
        torch_tensor_1 = ttnn.to_torch(output_tt, mesh_composer=mesh_composer)
        torch_tensor_1 = torch.reshape(
            torch_tensor_1, (total_batch_size, out_height_1, out_width_1, output_tt.shape[3])
        )
        torch_tensor_1 = torch.permute(torch_tensor_1, (0, 3, 1, 2))
        torch_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch op
        torch_out = torch_pool(torch_tensor_1).permute(0, 2, 3, 1)
        tt_input_2 = ttnn.from_torch(torch_out, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

    tt_int_tensor_3 = tt_Fire(
        96,
        16,
        64,
        64,
        input_tensor=tt_input_2,
        mesh_device=mesh_device,
        parameters=parameters["features"][3],
        mesh_mapper=mesh_mapper,
        mesh_composer=mesh_composer,
    )
    tt_int_tensor_4 = tt_Fire(
        128,
        16,
        64,
        64,
        input_tensor=tt_int_tensor_3,
        mesh_device=mesh_device,
        parameters=parameters["features"][4],
        mesh_mapper=mesh_mapper,
        mesh_composer=mesh_composer,
    )
    tt_int_tensor_5 = tt_Fire(
        128,
        32,
        128,
        128,
        input_tensor=tt_int_tensor_4,
        mesh_device=mesh_device,
        parameters=parameters["features"][5],
        mesh_mapper=mesh_mapper,
        mesh_composer=mesh_composer,
    )

    print("after 3f(3,4,5)", tt_int_tensor_5.shape)
    if max_pool_in_tt:  # enable when ceil_mode issue is fixed(#15039)
        tt_int_tensor_5_reshaped = ttnn.reshape(
            tt_int_tensor_5,
            (
                1,
                1,
                (tt_int_tensor_5.shape[0] * tt_int_tensor_5.shape[1] * tt_int_tensor_5.shape[2]),
                tt_int_tensor_5.shape[3],
            ),
        )
        out_pool_2 = ttnn.max_pool2d(
            input_tensor=tt_int_tensor_5_reshaped,
            batch_size=batch_size_tt,
            input_h=tt_int_tensor_5.shape[1],
            input_w=tt_int_tensor_5.shape[2],
            channels=tt_int_tensor_5_reshaped.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=None,
            applied_shard_scheme=None,
        )
        out_pool_2_reshaped = ttnn.reshape(out_pool_2, (batch_size_tt, 27, 27, out_pool_2.shape[3]))
        tt_input_3 = ttnn.from_device(out_pool_2_reshaped)
    else:
        torch_in = ttnn.to_torch(tt_int_tensor_5, mesh_composer=mesh_composer).permute(0, 3, 1, 2)
        torch_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch op
        torch_out_2 = torch_pool(torch_in).permute(0, 2, 3, 1)
        tt_input_3 = ttnn.from_torch(torch_out_2, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

    tt_int_tensor_6 = tt_Fire(
        256,
        32,
        128,
        128,
        input_tensor=tt_input_3,
        mesh_device=mesh_device,
        parameters=parameters["features"][7],
        mesh_mapper=mesh_mapper,
        mesh_composer=mesh_composer,
    )
    print("after f7", tt_int_tensor_6.shape)
    tt_int_tensor_7 = tt_Fire(
        256,
        48,
        192,
        192,
        input_tensor=tt_int_tensor_6,
        mesh_device=mesh_device,
        parameters=parameters["features"][8],
        mesh_mapper=mesh_mapper,
        mesh_composer=mesh_composer,
        num=8,
    )
    print("after f8", tt_int_tensor_7.shape)
    tt_int_tensor_8 = tt_Fire(
        384,
        48,
        192,
        192,
        input_tensor=tt_int_tensor_7,
        mesh_device=mesh_device,
        parameters=parameters["features"][9],
        mesh_mapper=mesh_mapper,
        mesh_composer=mesh_composer,
        num=9,
    )
    print("after f9", tt_int_tensor_8.shape)
    tt_int_tensor_9 = tt_Fire(
        384,
        64,
        256,
        256,
        input_tensor=tt_int_tensor_8,
        mesh_device=mesh_device,
        parameters=parameters["features"][10],
        mesh_mapper=mesh_mapper,
        mesh_composer=mesh_composer,
    )
    print("after f10", tt_int_tensor_9.shape)
    tt_int_tensor_9 = ttnn.to_layout(tt_int_tensor_9, layout=ttnn.ROW_MAJOR_LAYOUT)
    # torch.save(ttnn.to_torch(tt_int_tensor_9,mesh_composer=mesh_composer).permute(0,3,1,2),f"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/dumps/fg.pth")
    # print("before pool, after 10 f before reshape",tt_int_tensor_9.shape)
    tt_int_tensor_9_reshaped = ttnn.reshape(
        tt_int_tensor_9,
        (
            1,
            1,
            (tt_int_tensor_9.shape[0] * tt_int_tensor_9.shape[1] * tt_int_tensor_9.shape[2]),
            tt_int_tensor_9.shape[3],
        ),
    )

    # print("before pool, after 10 f",tt_int_tensor_9_reshaped.shape)
    out_pool_3 = ttnn.max_pool2d(
        input_tensor=tt_int_tensor_9_reshaped,
        batch_size=batch_size_tt,
        input_h=tt_int_tensor_9.shape[1],
        input_w=tt_int_tensor_9.shape[2],
        channels=tt_int_tensor_9.shape[3],
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
        memory_config=None,
        applied_shard_scheme=None,
    )

    out_pool_reshaped = ttnn.reshape(out_pool_3, (batch_size_tt, 13, 13, out_pool_3.shape[-1]))
    out_pool_reshaped = ttnn.from_device(out_pool_reshaped)
    tt_int_tensor_10 = tt_Fire(
        512,
        64,
        256,
        256,
        input_tensor=out_pool_reshaped,
        mesh_device=mesh_device,
        parameters=parameters["features"][12],
        mesh_mapper=mesh_mapper,
        mesh_composer=mesh_composer,
        num=12,
    )
    classifier_w = parameters.classifier[1].weight
    classifier_b = parameters.classifier[1].bias
    output_tt_tensor_11, out_height_11, out_width_11 = conv(
        activation="relu",
        input_tensor=tt_int_tensor_10,
        weight=classifier_w,
        bias=classifier_b,
        in_channels=classifier_w.shape[1],
        out_channels=classifier_w.shape[0],
        mesh_device=mesh_device,
        kernel_size=(classifier_w.shape[2], classifier_w.shape[3]),
        padding=(0, 0),
        stride=(1, 1),
        in_h=tt_int_tensor_10.shape[1],
        in_w=tt_int_tensor_10.shape[2],
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )

    output_tt_tensor_11 = ttnn.sharded_to_interleaved(output_tt_tensor_11, ttnn.DRAM_MEMORY_CONFIG)

    print("beofore layout", output_tt_tensor_11.shape)
    output_tt_tensor_11 = ttnn.to_layout(
        output_tt_tensor_11, layout=ttnn.ROW_MAJOR_LAYOUT
    )  # for reshape, pcc drops in tile
    output_tt_tensor_11 = ttnn.reshape(output_tt_tensor_11, (batch_size_tt, out_height_11, out_width_11, num_classes))
    # torch.save(ttnn.to_torch(output_tt_tensor_11,mesh_composer=mesh_composer).permute(0,3,1,2),"/home/ubuntu/venkatesh/tt-metal/models/demos/wormhole/squeezenet/tt/dumps/fg.pth")
    print("after layout", output_tt_tensor_11.shape, output_tt_tensor_11.memory_config())

    output_tt_tensor_11 = ttnn.to_layout(output_tt_tensor_11, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.global_avg_pool2d(output_tt_tensor_11)
    output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    print("after global avg", output_tensor.shape)
    output_tensor = ttnn.squeeze(output_tensor, dim=1)
    output_tensor = ttnn.squeeze(output_tensor, dim=1)
    return output_tensor
