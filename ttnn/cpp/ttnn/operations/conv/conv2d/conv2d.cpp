// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv2d.hpp"
#include <sys/types.h>
#include <cstdint>
#include <optional>
#include <utility>

#include "common/constants.hpp"
#include "common/math.hpp"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_op.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/pool/downsample/device/downsample_op.hpp"
#include "tt_metal/detail/reports/memory_reporter.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

using namespace tt;
namespace ttnn {
namespace operations::conv {
using sliding_window::SlidingWindowConfig;
using sliding_window::ParallelConfig;

namespace conv2d {

static uint32_t find_closest_largest_divisor(uint32_t num, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    while (num % divisor != 0) divisor = divisor - 1;
    return divisor;
}

static uint32_t find_closest_largest_divisor(uint32_t num1, uint32_t num2, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    while (num1 % divisor != 0 or num2 % divisor != 0) divisor = divisor - 1;
    return divisor;
}

static uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    uint32_t padded_num = round_up(num, divisor);
    while ((padded_num - num) >= padded_num / divisor) {
        divisor = divisor - 1;
        padded_num = round_up(num, divisor);
    }
    return divisor;
}

static uint32_t find_closest_largest_divisor_with_num_padding(uint32_t num1, uint32_t num2, uint32_t start_divisor) {
    uint32_t divisor = start_divisor;
    uint32_t padded_num1 = round_up(num1, divisor);
    uint32_t padded_num2 = round_up(num2, divisor);
    while ((padded_num1 - num1) >= (padded_num1 / divisor) || (padded_num2 - num2) >= (padded_num2 / divisor)) {
        divisor = divisor - 1;
        padded_num1 = round_up(num1, divisor);
        padded_num2 = round_up(num2, divisor);
    }
    return divisor;
}

// Converts convolution weights to tilized 2d matrix layout.
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_tiled_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype){
        return tt::tt_metal::convert_conv_weight_tensor_to_tiled_layout(std::move(conv_weight_tensor), in1_block_h, in1_block_w, output_dtype);
    }

// Converts convolution weights to tilized 2d matrix layout with special block height padding
// Returns a new tensor with layout=Tile
Tensor convert_conv_weight_tensor_to_special_padding_tiled_layout(
    const Tensor& conv_weight_tensor,
    uint32_t in1_block_h,
    uint32_t in1_block_w,
    std::optional<DataType> output_dtype){
        return tt::tt_metal::convert_conv_weight_tensor_to_special_padding_tiled_layout(std::move(conv_weight_tensor), in1_block_h, in1_block_w, output_dtype);
    }

// Converts convolution weights to grouped layout with padded zeros
Tensor convert_conv_weight_tensor_to_grouped_layout(const Tensor& conv_weight_tensor, uint32_t num_groups, DataType output_dtype){
       return tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(std::move(conv_weight_tensor), num_groups, output_dtype);
}

ParallelConfig determine_parallel_config(
    const TensorMemoryLayout shard_layout,
    uint32_t batch_size,
    uint32_t input_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t output_channels,
    const CoreCoord& compute_grid_size,
    ShardOrientation block_shard_orientation,
    bool enable_channels_padding,
    bool is_out_tiled,
    bool is_non_tile_mul_width) {

    uint32_t effective_tile_height = is_out_tiled ? tt::constants::TILE_HEIGHT : 1;
    uint32_t effective_tile_width = is_out_tiled ? tt::constants::TILE_WIDTH : 1;
    uint32_t out_nhw_ntiles = tt::round_up(batch_size * output_height * output_width, tt::constants::TILE_HEIGHT) / effective_tile_height;
    uint32_t input_channles_ntiles = tt::div_up(input_channels, effective_tile_width);
    uint32_t out_channels_ntiles = tt::div_up(output_channels, effective_tile_width);

    // calculate num_core_nhw and the grid
    uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
    CoreRangeSet grid;
    if (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding(out_nhw_ntiles, max_num_cores);
        grid = num_cores_to_corerangeset(num_cores_nhw, compute_grid_size, true);
    } else if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        uint32_t start_divisor =
                block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.x : compute_grid_size.y;
        uint32_t num_cores_nhw = find_closest_largest_divisor_with_num_padding(out_nhw_ntiles, start_divisor);
        auto channels_per_core = std::ceil((float)input_channels / effective_tile_width);
        if(is_non_tile_mul_width) {
            channels_per_core = input_channels;
        }

        uint32_t start_divisor_c =
            block_shard_orientation == ShardOrientation::COL_MAJOR ? compute_grid_size.y : compute_grid_size.x;
        uint32_t num_cores_c =
            enable_channels_padding
                ? find_closest_largest_divisor_with_num_padding(
                      out_channels_ntiles, input_channles_ntiles, start_divisor_c)
                : find_closest_largest_divisor(out_channels_ntiles, channels_per_core, start_divisor_c);
        uint32_t cores_x = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_nhw : num_cores_c;
        uint32_t cores_y = block_shard_orientation == ShardOrientation::COL_MAJOR ? num_cores_c : num_cores_nhw;
        CoreRange core_range = CoreRange(CoreCoord({0, 0}), CoreCoord({cores_x - 1, cores_y - 1}));
        grid = CoreRangeSet({core_range});
    } else if (shard_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        uint32_t num_cores_c = enable_channels_padding
                                   ? find_closest_largest_divisor_with_num_padding(input_channles_ntiles, max_num_cores)
                                   : find_closest_largest_divisor(input_channles_ntiles, max_num_cores);
        grid = num_cores_to_corerangeset(num_cores_c, compute_grid_size, true);
    } else {
        TT_THROW("Conv2d supports Height, Block or Width Sharded Layouts but got {}", shard_layout);
    }

    auto shard_orientation = shard_layout == TensorMemoryLayout::BLOCK_SHARDED ? block_shard_orientation : ShardOrientation::ROW_MAJOR; // NOTE: taking ROW_MAJOR as default orientation for HEIGHT_SHARDED and WIDTH_SHARDED
    ParallelConfig pconfig = {
        .grid = grid,
        .shard_scheme = shard_layout,
        .shard_orientation = shard_orientation };

    return pconfig;
}

static ParallelConfig determine_output_parallel_config(
    const ParallelConfig& input_parallel_config,
    const CoreCoord& compute_grid_size,
    uint32_t out_channels,
    bool is_mm_conv) {
    ParallelConfig output_parallel_config = input_parallel_config;
    if (input_parallel_config.shard_scheme == ttnn::TensorMemoryLayout::WIDTH_SHARDED && !is_mm_conv) {
        uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
        output_parallel_config = {
            .grid = num_cores_to_corerangeset(
                find_closest_largest_divisor_with_num_padding(
                    tt::div_up(out_channels, tt::constants::TILE_WIDTH), max_num_cores),
                compute_grid_size,
                true),
            .shard_scheme = ttnn::TensorMemoryLayout::WIDTH_SHARDED,
            .shard_orientation = input_parallel_config.shard_orientation};
    }
    return output_parallel_config;
}

uint32_t get_num_cores_nhw_from_parallel_config(const ParallelConfig& pconfig) {
    TT_ASSERT(!pconfig.grid.ranges().empty());
    TT_ASSERT(
        pconfig.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED ||
        pconfig.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED ||
        pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED);
    auto grid_size = pconfig.grid.bounding_box().grid_size();
    uint32_t num_cores = pconfig.grid.num_cores();
    uint32_t num_cores_nhw = 0;
    if(pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        return 1;
    }

    if (pconfig.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_nhw = num_cores;
    } else if (pconfig.shard_orientation == ShardOrientation::COL_MAJOR) {
        num_cores_nhw = grid_size.x;
    } else {
        TT_ASSERT(pconfig.shard_orientation == ShardOrientation::ROW_MAJOR);
        num_cores_nhw = grid_size.y;
    }

    TT_ASSERT(num_cores_nhw > 0);
    return num_cores_nhw;
}

uint32_t get_num_cores_channels_from_parallel_config(const ParallelConfig& pconfig) {
    TT_ASSERT(!pconfig.grid.ranges().empty());
    TT_ASSERT(
        pconfig.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED ||
        pconfig.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED ||
        pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED);
    auto grid_size = pconfig.grid.bounding_box().grid_size();
    uint32_t num_cores_channels = 0;
    if (pconfig.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        num_cores_channels = 1;
    } else if(pconfig.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        num_cores_channels = pconfig.grid.num_cores();
    } else if (pconfig.shard_orientation == ShardOrientation::COL_MAJOR) {
        num_cores_channels = grid_size.y;
    } else {
        TT_ASSERT(pconfig.shard_orientation == ShardOrientation::ROW_MAJOR);
        num_cores_channels = grid_size.x;
    }
    TT_ASSERT(num_cores_channels > 0);
    return num_cores_channels;
}

MemoryConfig create_sharded_memory_config_from_parallel_config(
    const ttnn::Shape& tensor_shape, ParallelConfig& parallel_config, uint32_t tile_size) {

    log_debug(tt::LogOp, "create_sharded_memory_config_from_parallel_config: tensor_shape: {}, parallel_config: {}, tile_size: {}", tensor_shape, parallel_config, tile_size);
    // tensor_shape is [N, H, W, C]
    TT_ASSERT(tensor_shape[0] == 1 && tensor_shape[1] == 1);  // todo: add support for generic non-2d shapes
    // uint32_t channels = tensor_shape[3];
    uint32_t channels = tensor_shape.with_tile_padding()[3];
    uint32_t num_cores_nhw = get_num_cores_nhw_from_parallel_config(parallel_config);
    uint32_t num_cores_channels = get_num_cores_channels_from_parallel_config(parallel_config);
    auto shard_scheme = parallel_config.shard_scheme;
    auto shard_orientation = parallel_config.shard_orientation;

    uint32_t nhw_shape = tensor_shape[0] * tensor_shape[1] * tensor_shape[2];
    uint32_t nhw_padded = nhw_shape;
    if(shard_scheme != TensorMemoryLayout::WIDTH_SHARDED) {
        nhw_padded = round_up(nhw_shape, num_cores_nhw * tile_size);
    }
    uint32_t nhw_shard = nhw_padded / num_cores_nhw;
    TT_ASSERT(channels % num_cores_channels == 0, "Channels: {}, num core channels: {}", channels, num_cores_channels);
    uint32_t channel_shard = channels / num_cores_channels;
    auto shard_spec = ShardSpec{parallel_config.grid, {nhw_shard, channel_shard}, shard_orientation};
    log_debug("Calculated Shard Spec = {}", shard_spec);
    return MemoryConfig{shard_scheme, BufferType::L1, shard_spec};
}


OptimizedConvParallelizationConfig determine_conv_op_parallel_config_from_conv_output_mem_config(
    const MemoryConfig& conv_output_mem_config, uint32_t num_cores_nhw, uint32_t num_cores_c) {
    TT_ASSERT(conv_output_mem_config.shard_spec.has_value());
    const auto& shard_spec = conv_output_mem_config.shard_spec.value();
    const auto& shard_shape = shard_spec.shape;
    uint32_t per_core_out_matrix_height_ntiles = div_up(shard_shape[0], 32);
    return {
        .grid_size = shard_spec.grid.bounding_box().grid_size(),
        .num_cores_nhw = num_cores_nhw,
        .num_cores_c = num_cores_c,
        .per_core_out_matrix_height = shard_shape[0],
        .per_core_out_matrix_width = shard_shape[1],
    };
}

std::pair<uint32_t, uint32_t> determine_largest_subblock_size(
    uint32_t block_height, uint32_t block_width, bool fp32_accum, bool split_reader_enabled) {
    constexpr std::array<std::pair<uint32_t, uint32_t>, 20> subblocks = {{
        {2, 4}, {4, 2}, {1, 8}, {8, 1}, {1, 7}, {7, 1}, {2, 3}, {3, 2}, {1, 6}, {6, 1},
        {1, 5}, {5, 1}, {2, 2}, {1, 4}, {4, 1}, {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 1},
    }};

    uint32_t subblock_h = 0;
    uint32_t subblock_w = 0;
    for (auto [subblock_height, subblock_width] : subblocks) {
        if (fp32_accum && (subblock_height * subblock_width > 4)) {
            continue;
        }

        if (split_reader_enabled && (block_height / subblock_height) < 2) {
            continue;
        }

        if ((block_height % subblock_height == 0) && (block_width % subblock_width == 0)) {
            if (subblock_width != block_width && subblock_height != 1) {
                continue;
            }
            subblock_h = subblock_height;
            subblock_w = subblock_width;
            break;
        }
    }
    TT_ASSERT(subblock_h > 0 && subblock_w > 0);
    return {subblock_h, subblock_w};
}

OptimizedConvBlockConfig determine_per_core_conv_block_config(
    const ParallelConfig& parallel_config,
    const OptimizedConvParallelizationConfig& conv_op_parallel_config,
    uint32_t padded_in_channels,
    uint32_t padded_output_height_ntiles,
    uint32_t act_block_h_override,
    uint32_t act_block_w_div,
    uint32_t window_h,
    uint32_t window_w,
    bool fp32_accum,
    bool split_reader_enabled) {

    uint32_t act_block_h_ntiles = div_up(conv_op_parallel_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT);
    if (act_block_h_override > 0) {
        if (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
            log_info(LogOp, "act_block_h_override is set, but ignored when Width Sharding is used");
        } else {
            uint32_t act_block_h_override_ntiles = act_block_h_override / constants::TILE_HEIGHT;
            if (padded_output_height_ntiles % act_block_h_override_ntiles == 0) {
                act_block_h_ntiles = act_block_h_override_ntiles;
            } else {
                log_info(
                    LogOp,
                    "act_block_h_override {} is not a valid override for padded_output_height_ntiles {}, override will "
                    "be ignored",
                    act_block_h_override_ntiles,
                    padded_output_height_ntiles);
            }
        }
    }
    auto grid_size = parallel_config.grid.bounding_box().grid_size();
    uint32_t act_c_num_blocks = parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED ? 1
                                : parallel_config.shard_orientation == ShardOrientation::COL_MAJOR ? grid_size.y
                                                                                                   : grid_size.x;
    TT_ASSERT(padded_in_channels % act_c_num_blocks == 0);
    uint32_t act_block_w = parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED
                            ? round_up(padded_in_channels * window_w, 32)
                            : round_up((padded_in_channels / act_c_num_blocks) * window_h * window_w, tt::constants::TILE_WIDTH);
    if(parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED) {
        act_block_w = (padded_in_channels * window_h * window_w)/(parallel_config.grid.num_cores() * act_block_w_div);
    }
    TT_ASSERT(act_block_w % 32 == 0);
    uint32_t act_block_w_ntiles = act_block_w / 32;
    uint32_t out_block_h_ntiles = div_up(conv_op_parallel_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT);
    uint32_t weight_block_w_ntiles = div_up(conv_op_parallel_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH);
    auto [out_subblock_h_ntiles, out_subblock_w_ntiles] =
        determine_largest_subblock_size(act_block_h_ntiles, weight_block_w_ntiles, fp32_accum, split_reader_enabled);
    return {
        .act_block_h_ntiles = act_block_h_ntiles,
        .act_block_w_ntiles = act_block_w_ntiles,
        .out_subblock_h_ntiles = out_subblock_h_ntiles,
        .out_subblock_w_ntiles = out_subblock_w_ntiles};
}

bool use_matmul_for_1x1_conv(
    const std::array<uint32_t, 2>& kernel_size,
    const std::array<uint32_t, 2>& stride,
    const std::array<uint32_t, 2>& padding,
    const std::array<uint32_t, 2>& dilation,
    uint32_t groups) {
    return kernel_size[0] == 1 && kernel_size[1] == 1 && stride[0] == stride[1] && stride[0] == 1 && padding[0] == 0 &&
           padding[1] == 0 && dilation[0] == 1 && dilation[1] == 1 && groups == 1;
}

// Implements a heuristic for selecting shard layout based on how many tenix cores are available
// for each shard.
static TensorMemoryLayout select_shard_spec(
    bool is_mm_conv,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t weights_width,
    uint32_t input_width,
    ShardOrientation shard_orientation,
    const CoreCoord& compute_grid_size) {
    auto get_core_count_for_sharding = [&](TensorMemoryLayout shard_layout) {
        return determine_parallel_config(
            shard_layout,
            batch_size,
            in_channels,
            output_height,
            output_width,
            out_channels,
            compute_grid_size,
            shard_orientation,
            !is_mm_conv).grid.num_cores();
    };

    // 1d convs support only height sharding
    const bool is_conv1d = weights_width == 1 && input_width == 1;

    const uint32_t cc_height = get_core_count_for_sharding(TensorMemoryLayout::HEIGHT_SHARDED);
    // matmul doesn't support width sharding
    const uint32_t cc_width =
        !is_mm_conv && !is_conv1d ? get_core_count_for_sharding(TensorMemoryLayout::WIDTH_SHARDED) : 0;
    const uint32_t cc_block = !is_conv1d ? get_core_count_for_sharding(TensorMemoryLayout::BLOCK_SHARDED) : 0;

    uint32_t max_cc = cc_block;
    TensorMemoryLayout shard_layout = TensorMemoryLayout::BLOCK_SHARDED;

    // Prefer block sharding over height sharding but make sure that we got at least
    // some blocking on width dimension as well.
    // Also for larger number of cores pefer block sharding, as it will divide weights along
    // the cores.
    const uint32_t max_num_cores = compute_grid_size.x * compute_grid_size.y;
    const uint32_t tree_quarter_cores = static_cast<uint32_t>(0.75f * max_num_cores);
    if ((cc_height > max_cc && max_cc < tree_quarter_cores) || (cc_height == max_cc && cc_height <= compute_grid_size.x)) {
        shard_layout = TensorMemoryLayout::HEIGHT_SHARDED;
        max_cc = cc_height;
    }

    if (cc_width >= max_cc) {
        shard_layout = TensorMemoryLayout::WIDTH_SHARDED;
        max_cc = cc_width;
    }

    if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        // For large number of input channels prefer width sharding
        // even if it has less cores.
        // For BH we probably need to adjust this, or even better we make block sharding
        // more configurable regarding L1 memory usage.
        if (cc_width >= 40 && in_channels > 1280) {
            shard_layout = TensorMemoryLayout::WIDTH_SHARDED;
            log_debug(LogOp, "Switching to WIDTH_SHARDED layout due to large in_channels");
            max_cc = cc_width;
        }
    }
    log_debug(LogOp, "Selected shard layout: {}, num cores: {}", shard_layout, max_cc);

    return shard_layout;
}

template <typename T>
std::tuple<ttnn::Shape, ttnn::MemoryConfig, bool, bool> get_conv_padded_input_shape_and_mem_config(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_);
    bool needs_shard_or_reshard = false;
    if (conv_config.override_sharding_config && conv_config.reshard_if_not_optimal) {
        TT_ASSERT(
            false,
            "Incorrect config provided: reshard_if_not_optimal and override_sharding_config cannot both be set.");
    }

    TT_FATAL(
        (!input_tensor_on_device || input_tensor_.is_sharded()) || conv_config.shard_layout.has_value(),
        "Tesor must be sharded or shard_layout must be set.");

    TensorMemoryLayout shard_layout;
    if (conv_config.shard_layout.has_value()) {
        shard_layout = conv_config.shard_layout.value();
    }

    ParallelConfig input_tensor_parallel_config;
    if (!input_tensor_on_device) {
        needs_shard_or_reshard = true;
    } else {
        const auto& input_memory_config = input_tensor_.memory_config();
        if (!input_memory_config.is_sharded()) {
            needs_shard_or_reshard = true;
        } else {
            const auto input_shard_scheme = input_memory_config.memory_layout;
            const auto input_shard_orientation = input_memory_config.shard_spec.value().orientation;
            const auto input_shard_grid = input_memory_config.shard_spec.value().grid;
            ParallelConfig pconfig = {
                .grid = input_shard_grid,
                .shard_scheme = input_shard_scheme,
                .shard_orientation = input_shard_orientation};
            input_tensor_parallel_config = pconfig;
            if (input_shard_scheme != TensorMemoryLayout::BLOCK_SHARDED &&
                input_shard_orientation != ShardOrientation::ROW_MAJOR) {
                needs_shard_or_reshard = true;
            }
            if (input_shard_scheme != TensorMemoryLayout::HEIGHT_SHARDED &&
                input_shard_scheme != TensorMemoryLayout::BLOCK_SHARDED &&
                input_shard_scheme != TensorMemoryLayout::WIDTH_SHARDED) {
                needs_shard_or_reshard = true;
            }
            if (conv_config.override_sharding_config) {
                TT_FATAL(conv_config.core_grid.has_value(), "If override_sharding_config is set, core_grid must be set as well.");
                TT_FATAL(conv_config.shard_layout.has_value(), "If override_sharding_config is set, shard_layout must be set as well.");
                if (conv_config.core_grid.value() != input_shard_grid) {
                    needs_shard_or_reshard = true;
                }
                if(shard_layout!=input_shard_scheme) {
                    needs_shard_or_reshard = true;
                }
                bool input_transpose_shards = input_shard_orientation == ShardOrientation::COL_MAJOR;
                if (shard_layout == TensorMemoryLayout::BLOCK_SHARDED && conv_config.transpose_shards != input_transpose_shards) {
                    needs_shard_or_reshard = true;
                }
            }
        }
    }

    // shallow conv variriant not supported
    // out_channels <= 256 incorrect output from pack_untilize_dst if output > 256 Tracking --> #14236
    // bf8 not supported due to limation of sharding dim multipl of 32
    bool use_non_tile_height = (shard_layout == TensorMemoryLayout::HEIGHT_SHARDED) && out_channels <= 256 && conv_config.act_block_h_override == 0 &&
        (conv_config.dtype == DataType::BFLOAT16 || conv_config.dtype == DataType::FLOAT32) && conv_config.output_layout == Layout::ROW_MAJOR && conv_config.input_channels_alignment != 16; //shalow conv varient

    ParallelConfig parallel_config = input_tensor_parallel_config;
    if (conv_config.reshard_if_not_optimal || needs_shard_or_reshard) {
        auto block_shard_orientation =
            conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
        auto num_cores_c = block_shard_orientation == ShardOrientation::COL_MAJOR ? device->compute_with_storage_grid_size().y : device->compute_with_storage_grid_size().x;
        auto elem_size = conv_config.weights_dtype == DataType::BFLOAT8_B ? 1 : 2;
        bool is_non_tile_mul_width = (shard_layout == TensorMemoryLayout::BLOCK_SHARDED) && conv_config.act_block_h_override == 0 &&
        conv_config.output_layout == Layout::ROW_MAJOR && ((elem_size*in_channels) % (16 * num_cores_c)) == 0 && (elem_size*in_channels);
        ParallelConfig optimal_parallel_config = determine_parallel_config(
            shard_layout,
            batch_size,
            in_channels,
            height,
            width,
            out_channels,
            device->compute_with_storage_grid_size(),
            block_shard_orientation,
            !is_mm_conv && !is_non_tile_mul_width,
            !use_non_tile_height,
            is_non_tile_mul_width);

        if (conv_config.override_sharding_config) {
            TT_FATAL(conv_config.core_grid.has_value(), "Error");
            // override parallel config
            auto shard_orientation = shard_layout == TensorMemoryLayout::BLOCK_SHARDED
                                         ? block_shard_orientation
                                         : ShardOrientation::ROW_MAJOR;
            parallel_config = {
                .grid = conv_config.core_grid.value(),
                .shard_scheme = shard_layout,
                .shard_orientation = shard_orientation};
        } else {
            parallel_config = optimal_parallel_config;
        }
        if (input_tensor_parallel_config != parallel_config) {
            needs_shard_or_reshard = true;
        }
    }
    if (needs_shard_or_reshard) {
        uint32_t input_num_cores_nhw = get_num_cores_nhw_from_parallel_config(parallel_config);
        uint32_t input_num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);

        // TT_ASSERT(input_tensor.get_legacy_shape() == input_tensor.get_shape());
        uint32_t tensor_height =
            input_tensor.get_shape()[0] * input_tensor.get_shape()[1] * input_tensor.get_shape()[2];
        uint32_t round_up_size = tt::constants::TILE_HEIGHT;
        if ((use_non_tile_height || shard_layout == TensorMemoryLayout::WIDTH_SHARDED) &&
            input_tensor_.layout() == Layout::ROW_MAJOR) {
            round_up_size = 1;
        }
        uint32_t input_tensor_height_snapped_to_tile =  tt::round_up(tensor_height, input_num_cores_nhw * round_up_size);
        TT_ASSERT(input_tensor_height_snapped_to_tile >= tensor_height);
        uint32_t input_tensor_width_snapped_to_channels_alignment =
            tt::round_up(input_tensor.get_shape()[3], input_num_cores_c * conv_config.input_channels_alignment);

        auto input_padded_shape = ttnn::Shape(std::array<uint32_t, 4>{
            1,
            1,
            input_tensor_height_snapped_to_tile,
            input_tensor_width_snapped_to_channels_alignment});  // TODO: resolve ttnn::types::Shape and
                                                                // tt::tt_metal::LegacyShape issue to clean up next line
        MemoryConfig input_tensor_sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            ttnn::Shape(std::array<uint32_t, 4>{
                input_padded_shape[0], input_padded_shape[1], input_padded_shape[2], input_padded_shape[3]}),
            parallel_config,
            round_up_size);

        return {input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard, use_non_tile_height};
    } else {
        return {input_tensor.shape(), input_tensor.memory_config(), needs_shard_or_reshard, use_non_tile_height};
    }
}

template <typename T>
std::tuple<ttnn::Tensor, ParallelConfig, ParallelConfig, bool, bool> shard_or_reshard_tensor_if_required(
    T* device,
    const ttnn::Tensor& input_tensor_,
    const Conv2dConfig& conv_config,
    uint32_t batch_size,
    uint32_t height,
    uint32_t width,
    uint32_t in_channels,
    uint32_t out_channels,
    bool is_mm_conv) {
    ttnn::Tensor input_tensor = input_tensor_;  // tensor to return
    bool input_tensor_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_);
    auto compute_grid_size = device->compute_with_storage_grid_size();

    auto [input_padded_shape, input_tensor_sharded_memory_config, needs_shard_or_reshard, use_non_tile_height] =
        get_conv_padded_input_shape_and_mem_config(
            device,
            input_tensor_,
            conv_config,
            batch_size,
            height,
            width,
            in_channels,
            out_channels,
            is_mm_conv);
    ParallelConfig parallel_config = {
        .grid = input_tensor_sharded_memory_config.shard_spec.value().grid,
        .shard_scheme = input_tensor_sharded_memory_config.memory_layout,
        .shard_orientation = input_tensor_sharded_memory_config.shard_spec.value().orientation
    };

    ParallelConfig output_parallel_config =
        determine_output_parallel_config(parallel_config, compute_grid_size, out_channels, is_mm_conv);

    if (needs_shard_or_reshard) {
        if (input_tensor.get_shape()[0] != 1 or input_tensor.get_shape()[1] != 1) {
            // reshape to [1, 1, N*H*W, C]
            input_tensor = ttnn::reshape(
                input_tensor,
                ttnn::SimpleShape(std::array<uint32_t, 4>{
                    1,
                    1,
                    input_tensor.get_shape()[0] * input_tensor.get_shape()[1] * input_tensor.get_shape()[2],
                    input_tensor.get_shape()[3]}));
        }

        uint32_t tensor_height = input_tensor.get_shape()[2];
        uint32_t tensor_width = input_tensor.get_shape()[3];

        if (!input_tensor_on_device) {
            if (input_padded_shape[-2] != tensor_height || input_padded_shape[-1] != tensor_width) {
                input_tensor = ttnn::pad(
                    input_tensor,
                    tt::tt_metal::Array4D({input_tensor.get_shape()[0],
                     input_tensor.get_shape()[1],
                     input_padded_shape[-2],
                     input_padded_shape[-1]}),
                    tt::tt_metal::Array4D({0, 0, 0, 0}),
                    0);
            }
        }

        if (input_tensor_on_device) {
            if (is_mm_conv && input_tensor.layout() == Layout::ROW_MAJOR &&
                parallel_config.shard_scheme != TensorMemoryLayout::HEIGHT_SHARDED) {
                // Workaround #13979 ttnn::tilize doesn't support BLOCK_SHARDED layout
                input_tensor =
                    ttnn::to_layout(input_tensor, Layout::TILE, std::nullopt, std::nullopt, input_tensor.device());
            }
            auto resharded_input_tensor = ttnn::to_memory_config(
                input_tensor, input_tensor_sharded_memory_config, std::nullopt);
            if (conv_config.deallocate_activation) {
                input_tensor.deallocate();
                resharded_input_tensor = ttnn::operations::core::reallocate(resharded_input_tensor, resharded_input_tensor.memory_config());
            }
            input_tensor = resharded_input_tensor;
        } else {
            if (is_mm_conv && input_tensor.layout() == Layout::ROW_MAJOR &&
                parallel_config.shard_scheme != TensorMemoryLayout::HEIGHT_SHARDED) {
                // Workaround #13979 ttnn::tilize doesn't support BLOCK_SHARDED layout
                input_tensor = ttnn::to_device(input_tensor, device, std::nullopt);
                input_tensor =
                    ttnn::to_layout(input_tensor, Layout::TILE, std::nullopt, std::nullopt, input_tensor.device());
                input_tensor = ttnn::to_memory_config(input_tensor, input_tensor_sharded_memory_config, std::nullopt);
            } else {
                input_tensor = ttnn::to_device(input_tensor, device, input_tensor_sharded_memory_config);
            }
        }
    }
    return {input_tensor, parallel_config, output_parallel_config, needs_shard_or_reshard, use_non_tile_height};
}

void validate_weight_and_bias_tensors(
    const ttnn::Tensor& weight_tensor, std::optional<const ttnn::Tensor>& bias_tensor) {
    TT_ASSERT(!ttnn::has_storage_type_of(weight_tensor, ttnn::DEVICE_STORAGE_TYPE));
    TT_ASSERT(weight_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_ASSERT(weight_tensor.get_shape().rank() == 4);
    // TODO: enable this assert
    // TT_ASSERT(weight_tensor.get_shape() == weight_tensor.get_legacy_shape());
    if (bias_tensor.has_value()) {
        TT_ASSERT(!ttnn::has_storage_type_of(bias_tensor.value(), ttnn::DEVICE_STORAGE_TYPE));
        TT_ASSERT(bias_tensor.value().get_shape().rank() == 4);
        TT_ASSERT(bias_tensor.value().get_layout() == Layout::ROW_MAJOR);
        // TODO: enable this assert
        // TT_ASSERT(bias_tensor.value().get_shape() == bias_tensor.value().get_legacy_shape());
    }
}

template <typename T>
std::pair<ttnn::Tensor, std::optional<ttnn::Tensor>> prepare_conv_weights_biases_and_move_to_device(
    const ttnn::Tensor& weight_tensor,
    std::optional<const ttnn::Tensor>& bias_tensor,
    uint32_t input_channels_alignment,
    DataType weights_bias_dtype,
    uint32_t weight_block_h_ntiles,
    uint32_t weight_block_w_ntiles,
    const ParallelConfig& parallel_config,
    T * device,
    uint32_t groups,
    uint32_t act_block_h_ntiles,
    uint32_t input_width) {

    validate_weight_and_bias_tensors(weight_tensor, bias_tensor);
    ttnn::Tensor weight_tensor_;  // tensor to return
    ttnn::Tensor bias_tensor_;

    auto original_weights_shape = weight_tensor.get_shape();
    uint32_t original_weights_out_channels = original_weights_shape[0];
    uint32_t original_weights_in_channels = original_weights_shape[1];
    uint32_t original_weights_window_h = original_weights_shape[2];
    uint32_t original_weights_window_w = original_weights_shape[3];

    bool is_conv1d = original_weights_window_w == 1 && input_width == 1;
    bool is_depthwise_conv = groups == original_weights_out_channels && original_weights_in_channels == 1;

    weight_tensor_ = weight_tensor;

    // Convert weight tensor to 0 padded shape if groups > 1
    if (!is_conv1d and groups > 1) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, weights_bias_dtype);
    }
    else if (is_conv1d and groups > 1) {
        if (is_depthwise_conv) {
            weight_tensor_ = convert_conv_weight_tensor_to_depthwise_layout(weight_tensor_, act_block_h_ntiles, weights_bias_dtype);
            weight_block_h_ntiles = act_block_h_ntiles;
        }
        else{
           weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_grouped_layout(weight_tensor_, groups, weights_bias_dtype);
        }
    }

    auto weights_shape = weight_tensor_.get_shape();
    uint32_t out_channels = weights_shape[0];
    uint32_t in_channels = weights_shape[1];
    uint32_t window_h = weights_shape[2];
    uint32_t window_w = weights_shape[3];

    uint32_t num_cores_channels = get_num_cores_channels_from_parallel_config(parallel_config);
    uint32_t out_channels_padded = tt::round_up(out_channels, num_cores_channels * tt::constants::TILE_WIDTH);
    uint32_t in_channels_padded = tt::round_up(in_channels, num_cores_channels * input_channels_alignment);
    uint32_t out_channel_padding = out_channels_padded - out_channels;

    tt::tt_metal::LegacyShape weights_channels_padded_shape = tt::tt_metal::LegacyShape(std::array<uint32_t, 4>(
        {out_channels_padded, in_channels_padded, window_h, window_w}));
    if (weights_bias_dtype == DataType::BFLOAT8_B) {
        TT_ASSERT(weight_tensor_.get_dtype() == DataType::FLOAT32);
        if (bias_tensor.has_value()) {
            TT_ASSERT(bias_tensor.value().get_dtype() == DataType::FLOAT32);
        }
    } else {
        // TODO: fix the need to check this. We should be able to accept any datatype and convert
        TT_ASSERT(weight_tensor_.get_dtype() == weights_bias_dtype);
        if (bias_tensor.has_value()) {
            TT_ASSERT(bias_tensor.value().get_dtype() == weights_bias_dtype);
        }
    }
    weight_tensor_ = ttnn::pad(weight_tensor_, weights_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D({0, 0, 0, 0}), 0);

    uint32_t num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);
    // for conv op, pad the weights to block shape
    if (parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_special_padding_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, weights_bias_dtype);
    } else if(parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_tiled_layout_block_sharded(
            weight_tensor_, num_cores_c, weights_bias_dtype);
    } else {
        weight_tensor_ = tt::tt_metal::convert_conv_weight_tensor_to_tiled_layout(
            weight_tensor_, weight_block_h_ntiles, weight_block_w_ntiles, weights_bias_dtype);
    }

    uint32_t weight_matrix_height = in_channels * window_h * window_w;
    int32_t weight_matrix_height_padding = weight_tensor_.shape()[2] - weight_matrix_height;
    TT_FATAL(weight_matrix_height_padding >= 0," Matrix Height Padding can't be negative");
    if(parallel_config.shard_scheme == TensorMemoryLayout::BLOCK_SHARDED) {
        /*weight_matrix_height = weight_tensor_.shape()[2];*/
        /*weight_matrix_height_padding = 0;*/
        /* For non-tile multiple of input channel case, Padding for each weight matrix shard is needed to accomodate
         * right amount of channels in each shard. For example, If we have 320 input and output channels, each
         * activation matrix shard would have 40 channels for 8 cores. Similarly, each weight matrix shard also needs to
         * have 40 output channels each to match number of cores for each shard. Current implementation of
         * convert_conv_weight_tensor_to_tiled_layout_block_sharded takes care of that and increases number of columns
         * in weight matrix. so for such cases, out_channels need to be
         * initialized with weight_tensor_.shape()[3].
         * However, there are cases like output channels are 255 with input as tile multiple, For such cases, padding is
         * added at last and output_channels for each shard would be of multiple of 32.
         * */
        /*out_channels = ((out_channels % 16 != 0) && (in_channels / num_cores_c) % 32 == 0) ? out_channels : weight_tensor_.shape()[3];*/
    }

    auto target_shape = ttnn::Shape(std::array<uint32_t,4>{1, 1, weight_matrix_height, out_channels},
            std::array<std::array<uint32_t, 2>, 4>{
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, 0},
            std::array<uint32_t, 2>{0, weight_matrix_height_padding},
            std::array<uint32_t, 2>{0, out_channel_padding}
            });
    weight_tensor_ = ttnn::reshape(weight_tensor_, target_shape);

    weight_tensor_ = ttnn::operations::core::to_device(weight_tensor_, device, std::nullopt);
    if (bias_tensor.has_value()) {
        if (parallel_config.shard_scheme != TensorMemoryLayout::BLOCK_SHARDED) {
            bias_tensor_ = bias_tensor.value();
            auto bias_shape = bias_tensor_.get_shape();
            TT_ASSERT(bias_shape[3] == out_channels && bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1);
            tt::tt_metal::LegacyShape bias_channels_padded_shape = tt::tt_metal::LegacyShape(
                std::array<uint32_t, 4>({1, 1, 32, round_up(out_channels, weight_block_w_ntiles * 32)}));
            bias_tensor_ = ttnn::pad(bias_tensor_, bias_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D{0, 0, 0, 0}, 0);
            bias_tensor_ = ttnn::to_layout(
                bias_tensor_, Layout::TILE, std::nullopt, std::nullopt, (T*)nullptr);
            if (bias_tensor_.get_dtype() != weights_bias_dtype) {
                bias_tensor_ = ttnn::to_dtype(bias_tensor_, weights_bias_dtype);
            }
        } else {
            bias_tensor_ = bias_tensor.value();
            auto bias_shape = bias_tensor_.get_shape();
            TT_ASSERT(bias_shape[3] == out_channels && bias_shape[0] == 1 && bias_shape[1] == 1 && bias_shape[2] == 1);
            tt::tt_metal::LegacyShape bias_channels_padded_shape = tt::tt_metal::LegacyShape(
                std::array<uint32_t, 4>({1, 1, 32, round_up(out_channels, weight_block_w_ntiles * 32)}));
            bias_tensor_ = ttnn::pad(bias_tensor_, bias_channels_padded_shape.to_array_4D(), tt::tt_metal::Array4D{0, 0, 0, 0}, 0);
            bias_tensor_ = convert_conv_bias_tensor_to_tiled_layout_block_sharded(
                bias_tensor_, num_cores_c, weights_bias_dtype);
        }
        bias_tensor_ = ttnn::operations::core::to_device(bias_tensor_, device, std::nullopt);
    }

    return {weight_tensor_, bias_tensor.has_value() ? bias_tensor_ : std::optional<ttnn::Tensor>()};
}

ttnn::operations::matmul::MatmulProgramConfig determine_matmul_op_config_from_conv_op_config(
    OptimizedConvParallelizationConfig conv_parallelization_config,
    OptimizedConvBlockConfig conv_blocking_config,
    bool height_sharded,
    const string& activation,
    bool transpose_mcast,
    uint32_t grid_size_along_c) {
    if (height_sharded) {
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig matmul_config = {
            .compute_with_storage_grid_size = conv_parallelization_config.grid_size,
            .in0_block_w = conv_blocking_config.act_block_w_ntiles,
            .out_subblock_h = conv_blocking_config.out_subblock_h_ntiles,
            .out_subblock_w = conv_blocking_config.out_subblock_w_ntiles,
            .per_core_M = div_up(conv_parallelization_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT),
            .per_core_N = div_up(conv_parallelization_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH),
            .fuse_batch = true,
            .mcast_in0 = false};
        if (activation != "") {
            matmul_config.fused_activation = ttnn::operations::unary::utils::string_to_unary_with_param(activation);
        }
        return matmul_config;
    } else {
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig matmul_config = {
            .compute_with_storage_grid_size = conv_parallelization_config.grid_size,
            .in0_block_w = conv_blocking_config.act_block_w_ntiles,
            .out_subblock_h = conv_blocking_config.out_subblock_h_ntiles,
            .out_subblock_w = conv_blocking_config.out_subblock_w_ntiles,
            .out_block_h = div_up(conv_parallelization_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT),
            .out_block_w = div_up(conv_parallelization_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH),
            .per_core_M = div_up(conv_parallelization_config.per_core_out_matrix_height, tt::constants::TILE_HEIGHT),
            .per_core_N = div_up(conv_parallelization_config.per_core_out_matrix_width, tt::constants::TILE_WIDTH),
            .transpose_mcast = transpose_mcast};
        if (activation != "") {
            matmul_config.fused_activation = ttnn::operations::unary::utils::string_to_unary_with_param(activation);
        }
        return matmul_config;
    }
}

static void adjust_conv_op_config_for_auto_shard(
    bool is_mm_conv,
    uint32_t batch_size,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t output_height,
    uint32_t output_width,
    uint32_t weights_width,
    uint32_t input_width,
    const CoreCoord& compute_grid_size,
    Conv2dConfig& conv_config,
    Layout input_tensor_layout) {
    ShardOrientation shard_orientation =
        conv_config.transpose_shards ? ShardOrientation::COL_MAJOR : ShardOrientation::ROW_MAJOR;
    conv_config.shard_layout = select_shard_spec(
        is_mm_conv,
        batch_size,
        in_channels,
        out_channels,
        output_height,
        output_width,
        weights_width,
        input_width,
        shard_orientation,
        compute_grid_size);

    if (conv_config.act_block_h_override == 0 && conv_config.shard_layout != TensorMemoryLayout::WIDTH_SHARDED) {
        if (in_channels <= constants::TILE_WIDTH / 2 && conv_config.input_channels_alignment == constants::TILE_WIDTH &&
            !is_mm_conv && conv_config.shard_layout == TensorMemoryLayout::HEIGHT_SHARDED && input_tensor_layout == Layout::ROW_MAJOR) {
            log_debug(LogOp, "Auto shard, enable shallow conv");
            // height sharded, non matmul conv, with input channels <= 16, and default setting for
            // input_channels_alignment
            // Currently data-movement ops have too many restrictions to support shallow convs with tiled input.
            conv_config.input_channels_alignment = constants::TILE_WIDTH / 2;
        }

        // Set act_block_h_override to min value to
        // be conservative with L1 memory usage.
        conv_config.act_block_h_override = constants::TILE_HEIGHT;
    }
}

template <typename T>
Result conv2d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    T* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const MemoryConfig>& memory_config) {
    const bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding, dilation, groups);
    const uint32_t output_height = ((input_height - kernel_size[0] - ((kernel_size[0] - 1 ) * (dilation[0] - 1)) + 2 * padding[0]) / stride[0]) + 1;
    const uint32_t output_width =
        ((input_width - kernel_size[1] - ((kernel_size[0] - 1) * (dilation[0] - 1)) + 2 * padding[1]) / stride[1]) + 1;

    Conv2dConfig conv_config = conv_config_.value_or(Conv2dConfig());
    const auto compute_grid_size = device->compute_with_storage_grid_size();

    if (!input_tensor.is_sharded() && !conv_config.shard_layout.has_value()) {
        // In this case we deduce the shard layout.
        adjust_conv_op_config_for_auto_shard(
            mm_conv,
            batch_size,
            in_channels,
            out_channels,
            output_height,
            output_width,
            weight_tensor.get_shape()[3],
            input_width,
            compute_grid_size,
            conv_config,
            input_tensor.layout());
    }

    auto [input_tensor_post_tm, parallel_config, output_parallel_config, tensor_manipulated, use_non_tile_height] = shard_or_reshard_tensor_if_required(
        device, input_tensor, conv_config, batch_size, output_height, output_width, in_channels, out_channels, mm_conv);
    if (tensor_manipulated) {
        if (conv_config.deallocate_activation) {
            ttnn::Tensor input_tensor_ = input_tensor;  // TODO: allow in place modification of inputs to the op
            input_tensor_.deallocate();
            // ttnn::operations::core::deallocate(input_tensor_);
        }
        conv_config.deallocate_activation = true;
    }

    uint32_t round_up_size = !use_non_tile_height ? tt::constants::TILE_HEIGHT : 1;
    uint32_t nhw_out = batch_size * output_height * output_width;
    uint32_t out_channels_padded = tt::round_up(
        out_channels,
        get_num_cores_channels_from_parallel_config(output_parallel_config) * tt::constants::TILE_WIDTH);
    MemoryConfig conv_out_memory_config = create_sharded_memory_config_from_parallel_config(
        ttnn::Shape(std::array<uint32_t, 4>{1, 1, nhw_out, out_channels_padded}),
        output_parallel_config,
        round_up_size);
    ParallelConfig largest_parallel_config = output_parallel_config.grid.num_cores() > parallel_config.grid.num_cores() ? output_parallel_config : parallel_config;

    OptimizedConvParallelizationConfig opt_conv_op_parallel_config = determine_conv_op_parallel_config_from_conv_output_mem_config(
        conv_out_memory_config,
        get_num_cores_nhw_from_parallel_config(largest_parallel_config),
        get_num_cores_channels_from_parallel_config(largest_parallel_config));

    uint32_t in_channels_padded = tt::round_up(
        in_channels,
        get_num_cores_channels_from_parallel_config(parallel_config) * conv_config.input_channels_alignment);

    uint32_t nhw_out_padded_ntile = get_num_cores_nhw_from_parallel_config(output_parallel_config) *
                                    conv_out_memory_config.shard_spec.value().shape[0] / tt::constants::TILE_HEIGHT;

    OptimizedConvBlockConfig opt_conv_op_block_config = determine_per_core_conv_block_config(
        parallel_config,
        opt_conv_op_parallel_config,
        in_channels_padded,
        nhw_out_padded_ntile,
        conv_config.act_block_h_override,
        conv_config.act_block_w_div,
        kernel_size[0],
        kernel_size[1],
        conv_config.fp32_dest_acc_enabled,
        conv_config.enable_split_reader);
    bool weight_is_on_device = ttnn::is_tensor_on_device_or_multidevice(weight_tensor);
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    std::optional<ttnn::Tensor> bias_tensor_on_device = bias_tensor;
    if (!weight_is_on_device) {
        // prepare weights in desired layout and move to device
        tie(weight_tensor_on_device, bias_tensor_on_device) = prepare_conv_weights_biases_and_move_to_device(
            weight_tensor,
            bias_tensor,
            conv_config.input_channels_alignment,
            conv_config.weights_dtype,
            opt_conv_op_block_config.act_block_w_ntiles,
            opt_conv_op_block_config.out_subblock_w_ntiles,
            parallel_config,
            device,
            groups,
            opt_conv_op_block_config.act_block_h_ntiles,
            input_width);
    }
    // if 1x1 conv w/ stride 1, convert input tensor to tile layout if required
    if (mm_conv) {
        Tensor input_tensor_post_tm_out = ttnn::to_layout(
            input_tensor_post_tm, Layout::TILE, conv_config.dtype, input_tensor_post_tm.memory_config(), device);
        if (conv_config.deallocate_activation) {
            input_tensor_post_tm.deallocate();
            // ttnn::operations::core::deallocate(input_tensor_post_tm);
        }
        input_tensor_post_tm = input_tensor_post_tm_out;
    }
    // call optimized conv op or matmul micro op
    bool input_is_on_device = ttnn::is_tensor_on_device_or_multidevice(input_tensor_post_tm);
    TT_ASSERT(input_is_on_device);
    DeviceComputeKernelConfig compute_kernel_config = ttnn::init_device_compute_kernel_config(
        device->arch(),
        std::nullopt,
        conv_config.math_fidelity,
        conv_config.math_approx_mode_enabled,
        conv_config.fp32_dest_acc_enabled,
        conv_config.packer_l1_accum_enabled);

    if (!mm_conv) {
        // call halo op
        SlidingWindowConfig sliding_window_config = SlidingWindowConfig{
            .batch_size = batch_size,
            .input_hw = {input_height, input_width},
            .window_hw = {kernel_size[0], kernel_size[1]},
            .stride_hw = {stride[0], stride[1]},
            .pad_hw = {padding[0], padding[1]},
            .dilation_hw = {dilation[0], dilation[1]},
            .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
            .core_range_set = input_tensor_post_tm.memory_config().shard_spec.value().grid,
            .snap_to_tile = !use_non_tile_height,
        };

        bool bypass_halo =
            (parallel_config.shard_scheme == TensorMemoryLayout::WIDTH_SHARDED &&
             sliding_window_config.pad_hw.first == 0 && sliding_window_config.pad_hw.second == 0);

        if (bypass_halo) {
            if (input_tensor_post_tm.layout() == Layout::TILE) {
                input_tensor_post_tm = ttnn::to_layout(
                    input_tensor_post_tm, Layout::ROW_MAJOR, std::nullopt, std::nullopt, device);
            }
        } else {
            Tensor halo_output = ttnn::halo(
                DefaultQueueId,
                input_tensor_post_tm,
                sliding_window_config,
                0,
                false,
                parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
                0,
                input_tensor_post_tm.memory_config(),
                !use_non_tile_height);

            if (conv_config.deallocate_activation) {
                ttnn::operations::core::deallocate(input_tensor_post_tm);
            }

            if (conv_config.reallocate_halo_output) {
                auto move_output = ttnn::operations::core::reallocate(halo_output, input_tensor_post_tm.memory_config());
                ttnn::operations::core::deallocate(halo_output);
                halo_output = move_output;
            }
            input_tensor_post_tm = halo_output;
        }

        // call conv micro op
        auto conv_output = optimized_conv_new(
            input_tensor_post_tm,
            weight_tensor_on_device,
            bias_tensor_on_device,
            sliding_window_config,
            out_channels,
            groups,
            conv_config.output_layout == Layout::ROW_MAJOR,
            conv_config.activation == "relu",
            conv_config.math_fidelity,
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            conv_out_memory_config,
            conv_config.dtype,
            {batch_size, input_height, input_width, in_channels},
            conv_config.input_channels_alignment == 16,
            compute_kernel_config,
            conv_config.enable_act_double_buffer,
            conv_config.enable_weights_double_buffer,
            conv_config.enable_split_reader,
            conv_config.enable_subblock_padding,
            use_non_tile_height);

        if (memory_config.has_value() && memory_config.value() != conv_output.memory_config()) {
            conv_output = ttnn::to_memory_config(conv_output, memory_config.value(), std::nullopt);
        }
        return {conv_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    } else {
        // run conv as matmul
        uint32_t num_cores_c = get_num_cores_channels_from_parallel_config(parallel_config);
        auto matmul_program_config = determine_matmul_op_config_from_conv_op_config(
            opt_conv_op_parallel_config,
            opt_conv_op_block_config,
            parallel_config.shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED,
            conv_config.activation,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            num_cores_c);
        Tensor matmul_input = input_tensor_post_tm;
        if (stride[0] > 1) {
            // run downsample
            matmul_input = ttnn::operations::downsample::downsample(
                input_tensor_post_tm, {batch_size, input_height, input_width, stride[0], stride[1]});
            if (conv_config.deallocate_activation) {
                ttnn::operations::core::deallocate(input_tensor_post_tm);
            }
        }
        auto matmul_output = ttnn::operations::matmul::matmul(
            matmul_input,
            weight_tensor_on_device,
            bias_tensor_on_device,
            ttnn::operations::matmul::Matmul{
            matmul_program_config,
            /*bcast_batch=*/std::nullopt,
            conv_out_memory_config,
            conv_config.dtype,
            compute_kernel_config});
        if (conv_config.deallocate_activation) {
            ttnn::operations::core::deallocate(matmul_input);
        }

        if (memory_config.has_value() && memory_config.value() != matmul_output.memory_config()) {
            matmul_output = ttnn::to_memory_config(matmul_output, memory_config.value(), std::nullopt);
        }

        return {matmul_output, output_height, output_width, weight_tensor_on_device, bias_tensor_on_device};
    }
}

Result Conv2dOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    Device * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const MemoryConfig>& memory_config){
    return conv2d(input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation, groups, std::move(bias_tensor), std::move(conv_config_), memory_config);
}

Result Conv2dOperation::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    MeshDevice * device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups,
    std::optional<const ttnn::Tensor> bias_tensor,
    const std::optional<const Conv2dConfig>& conv_config_,
    const std::optional<const MemoryConfig>& memory_config){
    return conv2d(input_tensor, weight_tensor, device, in_channels, out_channels, batch_size, input_height, input_width, kernel_size, stride, padding, dilation, groups, std::move(bias_tensor), std::move(conv_config_), memory_config);
}

}  // namespace conv2d
}  // namespace operations
}  // namespace ttnn
