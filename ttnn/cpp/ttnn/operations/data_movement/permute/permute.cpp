// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "permute.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"

#include "tt_metal/common/constants.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/cpp/ttnn/operations/copy.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

inline bool is_on_device(const Tensor& t) {
    return ttnn::has_storage_type_of(t, ttnn::StorageType::DEVICE) or
    ttnn::has_storage_type_of(t, ttnn::StorageType::MULTI_DEVICE);
}

inline bool has_tile_padding(const Tensor& t) {
    if (t.get_logical_shape().rank() > 1) {
        auto the_shape = t.get_logical_shape();
        auto the_shape_with_padding = t.get_padded_shape();
        return the_shape[-1] != the_shape_with_padding[-1] or the_shape[-2] != the_shape_with_padding[-2];
    }
    return false;
}

ttnn::Tensor permute_impl(const ttnn::Tensor &a, const SmallVector<uint32_t>& dims, const MemoryConfig& output_mem_config, const std::optional<float>& pad_value) {
    using ttnn::operations::experimental::auto_format::AutoFormat;
    Device * device;

    // Get the device
    if (a.storage_type() != StorageType::DEVICE) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = a.device();
    }

    if (a.get_shape().rank() > 4) {
        auto input = a.get_layout() == Layout::TILE ? ttnn::to_layout(a, Layout::ROW_MAJOR, std::nullopt, std::nullopt, (Device*)nullptr) : a;
        TT_FATAL(!(pad_value.has_value() && pad_value.value() != 0.0f), "Non-zero padding is not supported for permute on tensors with rank > 4.");
        input = ttnn::prim::permute(input, dims, output_mem_config, std::nullopt);
        return ttnn::to_layout(input, a.get_layout(), std::nullopt, std::nullopt, (Device*)nullptr);
    }

    TT_FATAL(dims.size() == 4, "Only 4D tensor are supported for permute.");
    uint32_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];

    // Convert tensor back to original
    auto input_shape = a.get_logical_shape();

    auto formatted_input_tensor = a;
    // WH and CN should be supported without typecast
    bool wh = N == 0 && C == 1 && H == 3 && W == 2;
    bool cn = N == 1 && C == 0 && H == 2 && W == 3;
    bool cnwh = N == 1 && C == 0 && H == 3 && W == 2;
    bool bfloat8_supported = wh || cn || cnwh;
    bool typecast = formatted_input_tensor.get_dtype() == DataType::BFLOAT8_B and !bfloat8_supported && !a.is_sharded();
    formatted_input_tensor = typecast ? ttnn::typecast(formatted_input_tensor, DataType::BFLOAT16) : formatted_input_tensor;

    auto output = formatted_input_tensor;
    auto transpose_wh = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::transpose(input, -2, -1, output_mem_config, std::nullopt);
    };

    auto transpose_hc = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::transpose(input, 1, -2, output_mem_config, pad_value);
    };

    auto transpose_cn = [&](const ttnn::Tensor& input) -> ttnn::Tensor {
        return ttnn::transpose(input, 0, 1, output_mem_config, std::nullopt);
    };

    if (N == 0 && C == 1 && H == 2 && W == 3) {
        output = formatted_input_tensor;
    } else if (N == 0 && C == 1 && H == 3 && W == 2) {
        output = transpose_wh(formatted_input_tensor);
    } else if (N == 0 && C == 2 && H == 1 && W == 3) {
        output = transpose_hc(formatted_input_tensor);
    } else if (N == 0 && C == 2 && H == 3 && W == 1) {
        output = transpose_wh(transpose_hc(formatted_input_tensor));
    } else if (N == 0 && C == 3 && H == 1 && W == 2) {
        output = transpose_hc(transpose_wh(formatted_input_tensor));
    } else if (N == 0 && C == 3 && H == 2 && W == 1) {
        output = transpose_wh(transpose_hc(transpose_wh(formatted_input_tensor)));
    } else if (N == 1 && C == 0 && H == 2 && W == 3) {
        output = transpose_cn(formatted_input_tensor);
    } else if (N == 1 && C == 0 && H == 3 && W == 2) {
        output = transpose_wh(transpose_cn(formatted_input_tensor));
    } else if (N == 1 && C == 2 && H == 0 && W == 3) {
        output = transpose_hc(transpose_cn(formatted_input_tensor));
    } else if (N == 1 && C == 2 && H == 3 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_cn(formatted_input_tensor)));
    } else if (N == 1 && C == 3 && H == 0 && W == 2) {
        output = transpose_hc(transpose_wh(transpose_cn(formatted_input_tensor)));
    } else if (N == 1 && C == 3 && H == 2 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(formatted_input_tensor))));
    } else if (N == 2 && C == 0 && H == 1 && W == 3) {
        output = transpose_cn(transpose_hc(formatted_input_tensor));
    } else if (N == 2 && C == 0 && H == 3 && W == 1) {
        output = transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor)));
    } else if (N == 2 && C == 1 && H == 0 && W == 3) {
        output = transpose_cn(transpose_hc(transpose_cn(formatted_input_tensor)));
    } else if (N == 2 && C == 1 && H == 3 && W == 0) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_cn(formatted_input_tensor))));
    } else if (N == 2 && C == 3 && H == 0 && W == 1) {
        output = transpose_hc(transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor))));
    } else if (N == 2 && C == 3 && H == 1 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(formatted_input_tensor)))));
    } else if (N == 3 && C == 0 && H == 1 && W == 2) {
        output = transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor)));
    } else if (N == 3 && C == 0 && H == 2 && W == 1) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor))));
    } else if (N == 3 && C == 1 && H == 0 && W == 2) {
        output = transpose_cn(transpose_hc(transpose_cn(transpose_wh(formatted_input_tensor))));
    } else if (N == 3 && C == 1 && H == 2 && W == 0) {
        output = transpose_wh(transpose_cn(transpose_hc(transpose_cn(transpose_wh(formatted_input_tensor)))));
    } else if (N == 3 && C == 2 && H == 0 && W == 1) {
        output = transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor)))));
    } else if (N == 3 && C == 2 && H == 1 && W == 0) {
        output = transpose_wh(transpose_hc(transpose_wh(transpose_cn(transpose_hc(transpose_wh(formatted_input_tensor))))));
    } else {
        TT_ASSERT(false, "Illegal permute args");
    }
    output = typecast ? ttnn::typecast(output, DataType::BFLOAT8_B) : output;
    return output;
}

ttnn::Tensor permute_launch(const ttnn::Tensor &a, tt::stl::Span<const int64_t> dims, const MemoryConfig& output_mem_config, const std::optional<float>& pad_value) {
    std::vector<ttnn::Tensor> output_tensors = {ttnn::Tensor(operation::get_workers_for_op_output({a}))};
    operation::launch_with_autoformat(
        [dims, output_mem_config, pad_value]  (const std::vector<ttnn::Tensor>& input_tensors, const std::vector<std::optional<const ttnn::Tensor>>& optional_input_tensors, const std::vector<std::optional<ttnn::Tensor>>& optional_output_tensors) mutable -> std::vector<ttnn::Tensor> {
            auto& a = input_tensors.at(0);
            SmallVector<uint32_t> normalized_dims(dims.size());
            std::transform(dims.begin(), dims.end(), normalized_dims.begin(), [a](std::int64_t idx) {return a.get_legacy_shape().get_normalized_index(idx);});
            SmallVector<uint32_t> seq_dims(dims.size());
            std::iota(seq_dims.begin(), seq_dims.end(), 0);
            if (normalized_dims == seq_dims) {
                return {ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_mem_config(a, output_mem_config)};
            }
            return {permute_impl(a, normalized_dims, output_mem_config, pad_value)};
        }, {a}, output_tensors);
    return output_tensors.at(0);
}

Tensor composite_invoke(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int64_t> dims,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<float>& pad_value) {

    auto output_tensor = permute_launch(input_tensor, dims, memory_config.value_or(input_tensor.memory_config()), pad_value);
    return output_tensor;
}

} // detail namespace

ttnn::Tensor ExecutePermute::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int64_t> dims,
    const std::optional<MemoryConfig>& memory_config,
    bool composite,
    const std::optional<float>& pad_value) {

    if (composite)
        return detail::composite_invoke(input_tensor, dims, memory_config, pad_value);

    const bool initial_input_tensor_on_device = detail::is_on_device(input_tensor);
    const auto input_layout = input_tensor.get_layout();
    const auto input_rank = input_tensor.get_logical_shape().rank();

    TT_FATAL(
        input_rank == dims.size(),
        "The number of dimensions in the tensor input does not match the length of the desired ordering");

    auto adjust_order = [](tt::stl::Span<const int64_t> dims) {
        ttnn::SmallVector<int64_t> new_order;
        TT_FATAL(dims.size() <= 4, "Error");
        int additional_ranks = 4 - dims.size();
        for (int i = 0; i < additional_ranks; i++) {
            new_order.push_back(i);
        }
        for (int i = 0; i < dims.size(); i++) {
            new_order.push_back(dims[i] + additional_ranks);
        }
        return new_order;
    };
    auto itensor = (input_tensor.get_logical_shape().rank() < 4) ? ttnn::unsqueeze_to_4D(input_tensor) : input_tensor;
    auto iorder = dims.size() < 4 ? adjust_order(dims) : dims;  // internals of permute_impl already adjust negative indices

    TT_FATAL(detail::is_on_device(itensor), "Error");
    auto output_tensor = detail::permute_launch(itensor, iorder, memory_config.value_or(input_tensor.memory_config()), pad_value);
    output_tensor = ttnn::to_layout(output_tensor, input_layout, std::nullopt, std::nullopt, (Device*)nullptr);

    if (input_rank < 4) {
        const auto shape = output_tensor.get_shape();
        const auto full_shape = output_tensor.get_shape().with_tile_padding();
        SmallVector<uint32_t> shape_vec{};
        SmallVector<uint32_t> full_shape_vec{};
        int i = 0;
        while (i < 3 and shape[i] == 1) i++;
        for (; i < shape.rank(); i++) {
            shape_vec.push_back(shape[i]);
            full_shape_vec.push_back(full_shape[i]);
        }
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(shape_vec, full_shape_vec));
    }

    if (initial_input_tensor_on_device and not detail::is_on_device(output_tensor)) {
        output_tensor = ttnn::to_device(
            output_tensor, input_tensor.device(), memory_config.value_or(input_tensor.memory_config()));
    }

    return output_tensor;
}

ttnn::Tensor ExecutePermute::invoke(
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const int64_t> dims,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<float>& pad_value) {
    return invoke(DefaultQueueId, input_tensor, dims, memory_config, true, pad_value);
}

ttnn::Tensor ExecutePermute::invoke(const ttnn::Tensor& input_tensor, tt::stl::Span<const int64_t> dims, const std::optional<float>& pad_value) {
    return invoke(input_tensor, dims, std::nullopt, pad_value);
}

} // ttnn::operations::data_movement namespace
