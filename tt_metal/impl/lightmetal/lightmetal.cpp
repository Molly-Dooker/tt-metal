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

//////////////////////////////////////
// Deserialization Functions        //
//////////////////////////////////////

// FIXME - Move to library only used by LightMetalBinary Executor. Here for testing purposes.
detail::TraceDescriptor fromFlatBuffer(const tt::target::lightmetal::TraceDescriptor* fb_desc) {
    if (!fb_desc) {
        std::cerr << "TraceDescriptor is null." << std::endl;
        return {};
    }

    detail::TraceDescriptor traceDesc;
    if (auto trace_data_fb = fb_desc->trace_data()) {
        traceDesc.data.assign(trace_data_fb->begin(), trace_data_fb->end());
    }
    traceDesc.num_completion_worker_cores = fb_desc->num_completion_worker_cores();
    traceDesc.num_traced_programs_needing_go_signal_multicast = fb_desc->num_traced_programs_needing_go_signal_multicast();
    traceDesc.num_traced_programs_needing_go_signal_unicast = fb_desc->num_traced_programs_needing_go_signal_unicast();

    return traceDesc;
}

// FIXME - Move to library only used by LightMetalBinary Executor. Here for testing purposes.
std::vector<uint8_t> readBinaryBlobFromFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary | std::ios::ate);
    if (!inFile) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    auto size = inFile.tellg();
    if (size <= 0) {
        throw std::runtime_error("Invalid file size: " + filename);
    }

    std::vector<uint8_t> blob(static_cast<size_t>(size));
    inFile.seekg(0);
    if (!inFile.read(reinterpret_cast<char*>(blob.data()), size)) {
        throw std::runtime_error("Failed to read data from file: " + filename);
    }

    return blob;
}

// FIXME - Move to library only used by LightMetalBinary Executor. Here for testing purposes.
std::optional<detail::TraceDescriptor> getTraceByTraceId(const std::string& filename, uint32_t target_trace_id) {
    try {
        std::vector<uint8_t> blob = readBinaryBlobFromFile(filename);
        const uint8_t* data = blob.data();
        size_t size = blob.size();

        flatbuffers::Verifier verifier(data, size);
        if (!target::lightmetal::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify Flatbuffer data." << std::endl;
            return std::nullopt;
        }

        const auto* lm_binary = target::lightmetal::GetLightMetalBinary(data);
        if (!lm_binary || !lm_binary->trace_descriptors()) {
            std::cerr << "No trace_descriptors found in the Flatbuffer file." << std::endl;
            return std::nullopt;
        }

        const auto* fb_trace_desc_by_id = lm_binary->trace_descriptors()->LookupByKey(target_trace_id);
        if (!fb_trace_desc_by_id) {
            std::cout << "Trace ID " << target_trace_id << " not found." << std::endl;
            return std::nullopt;
        }

        const auto* fb_desc = fb_trace_desc_by_id->desc();
        if (!fb_desc) {
            std::cerr << "Descriptor is null for trace_id: " << target_trace_id << std::endl;
            return std::nullopt;
        }

        return fromFlatBuffer(fb_desc);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return std::nullopt;
    }
}

}  // namespace v0
}  // namespace tt::tt_metal
