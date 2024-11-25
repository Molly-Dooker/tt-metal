// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <optional>

#include "common/bfloat4.hpp"
#include "common/bfloat8.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_stl/concepts.hpp"

namespace tt {

namespace tt_metal {

namespace tensor_impl {

// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              Low Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                        Data type converters, packers, and unpackers
// ======================================================================================
// TODO(arakhmati): Should cast_vec be a generator?

template <typename OutputDataType, template <typename> typename BufferType, typename InputDataType>
std::vector<OutputDataType> cast_vec(const BufferType<InputDataType>& data_to_convert) {
    std::vector<OutputDataType> converted_data;
    for (auto datum : data_to_convert) {
        if constexpr (std::is_same_v<OutputDataType, float> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back(datum.to_float());
        } else if constexpr (std::is_same_v<OutputDataType, uint32_t> and std::is_same_v<InputDataType, bfloat16>) {
            converted_data.push_back((uint32_t)datum.to_uint16());
        } else {
            converted_data.push_back(static_cast<OutputDataType>(datum));
        }
    }
    return converted_data;
}

uint32_t element_size_bytes(DataType dtype);

template <typename T>
constexpr inline size_t packed_buffer_size_bytes(size_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(T);
    return (volume_unpacked_data / num_type_in_u32) * sizeof(uint32_t);
}

// Specialization for float because it gets converted to bfloat16 before being packed
template <>
constexpr inline size_t packed_buffer_size_bytes<float>(size_t volume_unpacked_data) {
    auto num_type_in_u32 = sizeof(uint32_t) / sizeof(float);
    return (volume_unpacked_data / num_type_in_u32) * sizeof(uint32_t);
}

template <>
constexpr inline size_t packed_buffer_size_bytes<bfloat8_b>(size_t volume_unpacked_data) {
    return packed_buffer_size_bytes<uint32_t>(volume_unpacked_data);
}

template <>
constexpr inline size_t packed_buffer_size_bytes<bfloat4_b>(size_t volume_unpacked_data) {
    return packed_buffer_size_bytes<uint32_t>(volume_unpacked_data);
}

// ======================================================================================
//                                  Layout converters
// ======================================================================================
namespace detail {
static ttnn::SmallVector<uint32_t> to_4D_shape(const tt::tt_metal::LegacyShape& shape) {
    if (shape.rank() == 1) {
        return {1, 1, 1, shape[-1]};
    } else if (shape.rank() == 2) {
        return {1, 1, shape[-2], shape[-1]};
    } else if (shape.rank() == 3) {
        return {1, shape[-3], shape[-2], shape[-1]};
    } else if (shape.rank() == 4) {
        return {shape[-4], shape[-3], shape[-2], shape[-1]};
    } else {
        TT_THROW("Rank {} is not supported!", shape.rank());
    }
}

}  // namespace detail

template <typename T, template <typename> typename BufferType>
inline std::vector<T> convert_layout_row_major_to_tile(const tt::tt_metal::LegacyShape& shape, const Tile& tile, const BufferType<T>& data_to_convert) {
    TT_FATAL(
        (shape[-2] % tile.get_tile_shape()[0] == 0 && shape[-1] % tile.get_tile_shape()[1] == 0),
        "Unsupported shape for tensor conversion from row-major to tile layout. The tensor shape height and width must be a multiple of tile height ({}) and width ({}), but the provided shape is {}", tile.get_tile_shape()[0], tile.get_tile_shape()[1], shape);

    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    // TODO: Push this logic higher and use physical_size instead of shape
    int H = shape[shape.size() - 2], W = shape[shape.size() - 1];
    auto batch_size = 1;
    for (int i = 0; i < shape.size() - 2; i++) {
        batch_size *= shape[i];
    }
    H *= batch_size;
    return convert_layout(
        data_to_convert, std::array<uint32_t, 2>{H, W}, tests::utils::TensorLayoutType::LIN_ROW_MAJOR, tests::utils::TensorLayoutType::TILED_NFACES, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
}

template <typename T, template <typename> typename BufferType>
inline std::vector<T> convert_layout_tile_to_row_major(const tt::tt_metal::LegacyShape& shape, const Tile& tile, const BufferType<T>& data_to_convert) {
    auto tile_shape = tile.get_tile_shape();
    auto face_shape = tile.get_face_shape();
    auto transpose_within_face = tile.get_transpose_within_face();
    auto transpose_of_faces = tile.get_transpose_of_faces();

    // TODO: Push this logic higher and use physical_size instead of shape
    int H = shape[shape.size() - 2], W = shape[shape.size() - 1];
    auto batch_size = 1;
    for (int i = 0; i < shape.size() - 2; i++) {
        batch_size *= shape[i];
    }
    H *= batch_size;
    return convert_layout(
        data_to_convert, std::array<uint32_t, 2>{H, W}, tests::utils::TensorLayoutType::TILED_NFACES, tests::utils::TensorLayoutType::LIN_ROW_MAJOR, tile_shape, face_shape, transpose_within_face, transpose_of_faces);
}

// ======================================================================================
//                                      Validators
// ======================================================================================
void validate_on_device_dtype_and_layout(Device* device, const ttnn::SimpleShape& shape, DataType dtype, Layout layout);
void validate_sharded_buffer_allocation(
    const ttnn::SimpleShape& shape,
    Layout layout,
    DataType data_type,
    const ShardSpecBuffer& shard_params,
    const MemoryConfig& memory_config,
    const Tile& tile);
// -----------------------------------------------------------------------------------------------------------------------------------------------
// ===============================================================================================================================================
//                                                              High Level APIs
// ===============================================================================================================================================
// -----------------------------------------------------------------------------------------------------------------------------------------------

// ======================================================================================
//                           Data reader, writer, and initializers
// ======================================================================================

DeviceBuffer allocate_buffer_on_device(Device* device, const TensorSpec& tensor_spec);

template <typename T>
inline void read_data_from_device_buffer(
    CommandQueue& cq, DeviceBuffer device_buffer, void* host_buffer_data, bool blocking) {
    EnqueueReadBuffer(cq, device_buffer, host_buffer_data, blocking);
}

template <typename T>
inline void read_data_from_device_buffer(DeviceBuffer device_buffer, std::vector<T>& host_buffer) {
    ::detail::ReadFromBuffer(device_buffer, host_buffer);
}

// ======================================================================================
//                                         .to()
// ======================================================================================

template <typename T>
Tensor to_host(const Tensor& tensor, bool blocking = true, uint8_t cq_id = ttnn::DefaultQueueId);

template <typename T>
Tensor to_host_sharded(const Tensor& tensor);

template <typename T>
Tensor to_device(
    const Tensor& tensor,
    Device* target_device,
    const MemoryConfig& memory_config,
    std::optional<std::reference_wrapper<CommandQueue>> queue);

template <typename T>
Tensor to_layout(const Tensor& tensor, Layout target_layout);

template <typename T>
Tensor to_layout_bfloat(const Tensor& tensor, Layout target_layout);

// ======================================================================================
//                                  .pad() and .unpad()
// ======================================================================================
template <typename T>
Tensor pad(const Tensor& tensor, const tt::tt_metal::LegacyShape& output_shape, const ttnn::SimpleShape& input_tensor_start, float pad_value);

template <typename T>
Tensor unpad(const Tensor& tensor, const ttnn::SimpleShape& output_tensor_start, const ttnn::SimpleShape& output_tensor_end);

// ======================================================================================
//                                         Print
// ======================================================================================

std::ostream& operator<<(std::ostream& os, const DataType& dtype);

enum class TensorPrintProfile {
    Empty,
    Short,
    Full,
};

extern TensorPrintProfile TTNN_TENSOR_PRINT_PROFILE;

template <typename T>
std::string to_string(const Tensor& tensor, std::optional<DataType> original_dtype = std::nullopt);

template <typename T>
Tensor extract_shard(const Tensor& tensor, const uint32_t& core_id);

}  // namespace tensor_impl

}  // namespace tt_metal

}  // namespace tt
