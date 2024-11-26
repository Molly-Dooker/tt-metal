// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal_replay.hpp"
#include "binary_generated.h"
#include "command_generated.h"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/common/logger.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace tt::tt_metal {
inline namespace v0 {

// There are a bunch of things to do in this file and figure out
// 1. Executor: Open Flatbuffer binary, loop over contents, execute contents.
// 2. In order to do that, need deserialize/convert from flatbuffer representation
// 3. And have handlers to call Host API functions.

//////////////////////////////////////
// File/Blob IO                     //
//////////////////////////////////////

// A convenience function - Read arbitrary binary blob from file.
void readBinaryBlobFromFile(const std::string& filename, std::vector<uint8_t>& blob) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("File is empty or invalid: " + filename);
    }

    blob.resize(static_cast<size_t>(size));

    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(blob.data()), size)) {
        throw std::runtime_error("Failed to read file: " + filename);
    }
}


// Verify and return LightMetalBinary flatbuffers object given a matching binary blob
const target::lightmetal::LightMetalBinary* getLightMetalBinary(const std::vector<uint8_t>& blob) {
    try {
        // Read the binary blob from the file.
        const uint8_t* data = blob.data();
        size_t size = blob.size();

        // Verify the FlatBuffer data.
        flatbuffers::Verifier verifier(data, size);
        if (!target::lightmetal::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify Flatbuffer data." << std::endl;
            return nullptr;
        }
        log_info(tt::LogMetalTrace, "Finished VerifyLightMetalBinaryBuffer");

        // Parse and return the FlatBuffer object.
        const auto* lm_binary = target::lightmetal::GetLightMetalBinary(data);
        if (!lm_binary) {
            std::cerr << "Failed to get LightMetalBinary object from Flatbuffer data." << std::endl;
            return nullptr;
        }

        return lm_binary;
    } catch (const std::exception& e) {
        std::cerr << "Exception while opening FlatBuffer binary: " << e.what() << std::endl;
        return nullptr;
    }
}

//////////////////////////////////////
// Deserialization Functions        //
//////////////////////////////////////

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

// FIXME - Could simplify this, don't open from file again, instead store flatbuffer binary in state.
std::optional<detail::TraceDescriptor> getTraceByTraceId(const std::string& filename, uint32_t target_trace_id) {
    try {

        std::vector<uint8_t> binary_blob;
        readBinaryBlobFromFile(filename, binary_blob);
        const auto* lm_binary = getLightMetalBinary(binary_blob);

        if (!lm_binary) {
            std::cerr << "Failed to open FlatBuffer binary." << std::endl;
            return std::nullopt;
        }

        // Ensure the trace_descriptors field exists.
        if (!lm_binary->trace_descriptors()) {
            std::cerr << "No trace_descriptors found in the FlatBuffer file." << std::endl;
            return std::nullopt;
        }

        // Lookup the trace descriptor by key.
        const auto* fb_trace_desc_by_id = lm_binary->trace_descriptors()->LookupByKey(target_trace_id);
        if (!fb_trace_desc_by_id) {
            std::cout << "Trace ID " << target_trace_id << " not found." << std::endl;
            return std::nullopt;
        }

        // Retrieve and validate the descriptor.
        const auto* fb_desc = fb_trace_desc_by_id->desc();
        if (!fb_desc) {
            std::cerr << "Descriptor is null for trace_id: " << target_trace_id << std::endl;
            return std::nullopt;
        }

        // Convert and return the descriptor as a detail::TraceDescriptor.
        return fromFlatBuffer(fb_desc);
    } catch (const std::exception& e) {
        std::cerr << "Exception in getTraceByTraceId: " << e.what() << std::endl;
        return std::nullopt;
    }
}


//////////////////////////////////////
// Debug Code                       //
//////////////////////////////////////


void printLightMetalBinaryContents(const tt::target::lightmetal::LightMetalBinary* lm_binary) {
    if (!lm_binary) {
        std::cerr << "Invalid LightMetalBinary object." << std::endl;
        return;
    }

    const auto* trace_descriptors = lm_binary->trace_descriptors();
    if (!trace_descriptors) {
        std::cout << "No trace descriptors found in the binary." << std::endl;
    } else {
        // Print all trace descriptors.
        std::cout << "Number of trace descriptors: " << trace_descriptors->size() << std::endl;
        for (const auto* descriptor_by_id : *trace_descriptors) {
            if (!descriptor_by_id) continue;

            uint32_t trace_id = descriptor_by_id->trace_id();
            const auto* trace_desc = descriptor_by_id->desc();

            if (!trace_desc) {
                std::cerr << "Descriptor is null for trace_id: " << trace_id << std::endl;
                continue;
            }

            // Print trace descriptor details.
            std::cout << "Trace ID: " << trace_id << std::endl;
            std::cout << "  Number of completion worker cores: "
                      << trace_desc->num_completion_worker_cores() << std::endl;
            std::cout << "  Number of programs needing multicast: "
                      << trace_desc->num_traced_programs_needing_go_signal_multicast() << std::endl;
            std::cout << "  Number of programs needing unicast: "
                      << trace_desc->num_traced_programs_needing_go_signal_unicast() << std::endl;

            // Print trace data.
            const auto* trace_data = trace_desc->trace_data();
            if (trace_data && trace_data->size() > 0) {
                std::cout << "  Trace Data (size: " << trace_data->size() << "): ";
                for (uint32_t value : *trace_data) {
                    std::cout << value << " ";
                }
                std::cout << std::endl;
            } else {
                std::cout << "  Trace Data: None" << std::endl;
            }
        }
    }

    // Print all commands.
    const auto* commands = lm_binary->commands();
    if (!commands || commands->size() == 0) {
        std::cout << "No commands found in the binary." << std::endl;
    } else {
        std::cout << "Number of commands: " << commands->size() << std::endl;
        for (const auto* command : *commands) {
            if (!command) continue;

            auto cmd_type = command->cmd_type();
            switch (cmd_type) {
                case tt::target::CommandUnion_ReplayTrace: {
                    const auto* replay_trace = command->cmd_as_ReplayTrace();
                    if (replay_trace) {
                        std::cout << "ReplayTrace Command:" << std::endl;
                        std::cout << "  cq_id: " << replay_trace->cq_id() << std::endl;
                        std::cout << "  tid: " << replay_trace->tid() << std::endl;
                        std::cout << "  blocking: " << (replay_trace->blocking() ? "true" : "false") << std::endl;
                    }
                    break;
                }
                case tt::target::CommandUnion_EnqueueTrace: {
                    const auto* enqueue_trace = command->cmd_as_EnqueueTrace();
                    if (enqueue_trace) {
                        std::cout << "EnqueueTrace Command:" << std::endl;
                        std::cout << "  cq_id: " << enqueue_trace->cq_id() << std::endl;
                        std::cout << "  tid: " << enqueue_trace->tid() << std::endl;
                        std::cout << "  blocking: " << (enqueue_trace->blocking() ? "true" : "false") << std::endl;
                    }
                    break;
                }
                default:
                    std::cout << "Unknown Command type: " << cmd_type << std::endl;
                    break;
            }
        }
    }
}


//////////////////////////////////////
// Executor                         //
//////////////////////////////////////


bool example_code() {
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, 900000000);
    tt_metal::CommandQueue& cq = device->command_queue();
    tt_metal::Program program = tt_metal::CreateProgram();
    bool pass = tt_metal::CloseDevice(device);
    return pass;
}

// Main entry point to execute a light metal binary blob and return success (0) /failure (1)
bool executeLightMetalBinary(const std::vector<uint8_t>& blob) {

    log_info(tt::LogMetalTrace, "Starting {}", __FUNCTION__);
    bool pass = true;

    try {
        pass = example_code();
        const auto* lm_binary = getLightMetalBinary(blob);

        if (lm_binary) {
            printLightMetalBinaryContents(lm_binary);
            // FIXME - Replace with executor handler here.
        } else {
            std::cerr << "Failed to load FlatBuffer binary." << std::endl;
        }

    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    return !pass;
}

}  // namespace v0
}  // namespace tt::tt_metal
