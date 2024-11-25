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

}  // namespace v0
}  // namespace tt::tt_metal
