// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal.hpp"
#include "binary_generated.h"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/common/logger.hpp"

namespace tt::tt_metal {
inline namespace v0 {

//////////////////////////////////////
// Serialization Functions          //
//////////////////////////////////////

TraceDescriptorByTraceIdOffset toFlatBuffer(flatbuffers::FlatBufferBuilder& builder, const detail::TraceDescriptor& traceDesc, const uint32_t trace_id) {
    auto trace_data_offset = builder.CreateVector(traceDesc.data);

    return tt::target::lightmetal::CreateTraceDescriptorByTraceId(
        builder,
        trace_id,
        tt::target::lightmetal::CreateTraceDescriptor(
            builder,
            trace_data_offset,
            traceDesc.num_completion_worker_cores,
            traceDesc.num_traced_programs_needing_go_signal_multicast,
            traceDesc.num_traced_programs_needing_go_signal_unicast
        )
    );
}

std::vector<uint8_t> createLightMetalBinary(LightMetalTrace& light_metal_trace) {
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<tt::target::lightmetal::TraceDescriptorByTraceId>> trace_desc_vec;
    trace_desc_vec.reserve(light_metal_trace.traces.size()); // Reserve space for performance

    for (const auto& [trace_id, trace_desc] : light_metal_trace.traces) {
        trace_desc_vec.emplace_back(toFlatBuffer(builder, trace_desc, trace_id));
        log_info(tt::LogMetal, "KCM {} for trace_id: {}", __FUNCTION__, trace_id);
    }

    auto sorted_trace_descriptors = builder.CreateVectorOfSortedTables(&trace_desc_vec);
    auto light_metal_binary = CreateLightMetalBinary(builder, sorted_trace_descriptors);
    builder.Finish(light_metal_binary);

    const uint8_t* buffer_ptr = builder.GetBufferPointer();
    size_t buffer_size = builder.GetSize();
    return {buffer_ptr, buffer_ptr + buffer_size};
}

bool writeBinaryBlobToFile(const std::string& filename, const std::vector<uint8_t>& blob) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Unable to open file: " << filename << " for writing." << std::endl;
        return false;
    }

    if (!outFile.write(reinterpret_cast<const char*>(blob.data()), blob.size())) {
        std::cerr << "Failed to write binary data to file: " << filename << std::endl;
        return false;
    }

    return true;
}

}  // namespace v0
}  // namespace tt::tt_metal
