// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>

// Forward decl for binary_generated.h
namespace tt::target::lightmetal {
    struct TraceDescriptor;
    struct TraceDescriptorByTraceId;
    struct LightMetalBinary;
}

// Forward Declaration to avoid trace_buffer.hpp
namespace tt::tt_metal::detail {
    class TraceDescriptor;
}

namespace tt::tt_metal {
inline namespace v0 {

// Convert from Flatbuffer TraceDescriptor to C++ struct
detail::TraceDescriptor fromFlatBuffer(const tt::target::lightmetal::TraceDescriptor* fb_desc);

// Open a binary file and return contents as a binary blob via vector.
void readBinaryBlobFromFile(const std::string& filename, std::vector<uint8_t>& blob);

const target::lightmetal::LightMetalBinary* openFlatBufferBinary(std::vector<uint8_t>& buffer);

// Deserialize a specific trace by trace_id.
std::optional<detail::TraceDescriptor> getTraceByTraceId(const std::string& filename, uint32_t target_trace_id);

bool executeLightMetalBinary(const std::vector<uint8_t>& blob);

}  // namespace v0
}  // namespace tt::tt_metal
