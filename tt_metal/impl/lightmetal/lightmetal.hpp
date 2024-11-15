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

// LightMetalTraceConfig struct
struct LightMetalTraceConfig {
    bool capture_enabled = false;
    bool auto_serialize_metal_trace = true;
    std::string filename = "/tmp/light_metal_trace.bin";
};

// LightMetalTrace struct
struct LightMetalTrace {
    std::vector<std::pair<uint32_t, detail::TraceDescriptor>> traces;
    LightMetalTraceConfig config;
};

using TraceDescriptorByTraceIdOffset = flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>;

// Convert TraceDescriptor + trace_id to flatbuffer
TraceDescriptorByTraceIdOffset toFlatBuffer(flatbuffers::FlatBufferBuilder& builder, const detail::TraceDescriptor& traceDesc, const uint32_t trace_id);

// Take built-up light metal trace data and serialize it to a flatbuffer binary.
std::vector<std::uint8_t> createLightMetalBinary(LightMetalTrace& light_metal_trace);

// Write an arbitrary binary blob to file.
bool writeBinaryBlobToFile(const std::string& filename, const std::vector<uint8_t>& blob);

// Convert from Flatbuffer TraceDescriptor to C++ struct
detail::TraceDescriptor fromFlatBuffer(const tt::target::lightmetal::TraceDescriptor* fb_desc);

// Read an arbitrary binary blob from file.
std::vector<uint8_t> readBinaryBlobFromFile(const std::string& filename);

// Deserialize a specific trace by trace_id.
std::optional<detail::TraceDescriptor> getTraceByTraceId(const std::string& filename, uint32_t target_trace_id);

}  // namespace v0
}  // namespace tt::tt_metal
