// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal_replay.hpp"
#include <iostream>
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
// Helper Functions                 //
//////////////////////////////////////

// Alternative to this is a switch statement that calls handlers for each command type (like tt-mlir)
// Keep registry of function replay handlers to seperate from execution loop.
using CommandReplayHandler = std::function<void(const tt::target::Command*)>;
std::map<::tt::target::CommandType, CommandReplayHandler> init_function_replay_registry() {

    std::map<::tt::target::CommandType, CommandReplayHandler> registry;

    registry[tt::target::CommandType::ReplayTraceCommand] = [](const ::tt::target::Command* cmd_union) {
        auto cmd = cmd_union->cmd_as_ReplayTraceCommand();
        log_info(tt::LogMetalTrace, "Calling ReplayTrace(). cq_id: {} tid: {} blocking: {}", cmd->cq_id(), cmd->tid(), cmd->blocking());
        // ReplayTrace(this->device_, cmd->cq_id(), cmd->tid(), cmd->blocking());
    };

    registry[tt::target::CommandType::EnqueueTraceCommand] = [](const ::tt::target::Command* cmd_union) {
        auto cmd = cmd_union->cmd_as_EnqueueTraceCommand();
        log_info(tt::LogMetalTrace, "Calling EnqueueTrace(). cq_id: {} tid: {} blocking: {}", cmd->cq_id(), cmd->tid(), cmd->blocking());
        // CommandQueue& command_queue = this->device_->command_queue();
        // EnqueueTrace(command_queue, cmd->tid(), cmd->blocking());
    };

    // LoadTraceId API will be called, but it doesn't have acess to flatbuffer binary. Need to have it take blob I think.
    registry[tt::target::CommandType::LightMetalLoadTraceIdCommand] = [](const ::tt::target::Command* cmd_union) {
        auto cmd = cmd_union->cmd_as_LightMetalLoadTraceIdCommand();
        log_info(tt::LogMetalTrace, "Calling LightMetalLoadTraceId(). cq_id: {} tid: {}", cmd->cq_id(), cmd->tid());
        // Idea: Replay handler can fetch data based on identifier and pass to function being called.
    };

    return registry;
}

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

//////////////////////////////////////
// LightMetalReplay Class           //
//////////////////////////////////////

LightMetalReplay::LightMetalReplay(std::vector<uint8_t> blob)
    : blob_(std::move(blob)), lm_binary_(nullptr) {
    lm_binary_ = parseFlatBufferBinary();  // Parse and store the FlatBuffer binary
    if (!lm_binary_) {
        throw std::runtime_error("Failed to parse FlatBuffer binary during initialization.");
    }
}

const target::lightmetal::LightMetalBinary* LightMetalReplay::parseFlatBufferBinary() {
    try {
        const uint8_t* data = blob_.data();
        size_t size = blob_.size();

        // Verify the FlatBuffer data.
        flatbuffers::Verifier verifier(data, size);
        if (!target::lightmetal::VerifyLightMetalBinaryBuffer(verifier)) {
            std::cerr << "Failed to verify FlatBuffer data." << std::endl;
            return nullptr;
        }

        // Parse and return the FlatBuffer object.
        return target::lightmetal::GetLightMetalBinary(data);
    } catch (const std::exception& e) {
        std::cerr << "Exception while parsing FlatBuffer binary: " << e.what() << std::endl;
        return nullptr;
    }
}

// FIXME - Could probably simplify this further.
std::optional<detail::TraceDescriptor> LightMetalReplay::getTraceByTraceId(uint32_t target_trace_id) {

    try {

        if (!lm_binary_) {
            std::cerr << "FlatBuffer binary not initialized." << std::endl;
            return std::nullopt;
        }

        // Ensure the trace_descriptors field exists.
        if (!lm_binary_->trace_descriptors()) {
            std::cerr << "No trace_descriptors found in the FlatBuffer file." << std::endl;
            return std::nullopt;
        }

        // Lookup the trace descriptor by key.
        const auto* fb_trace_desc_by_id = lm_binary_->trace_descriptors()->LookupByKey(target_trace_id);
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

bool example_code() {
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, 900000000);
    tt_metal::CommandQueue& cq = device->command_queue();
    tt_metal::Program program = tt_metal::CreateProgram();
    bool pass = tt_metal::CloseDevice(device);
    return pass;
}

// Temporary debug function to print the contents of the FlatBuffer binary.
void LightMetalReplay::printLightMetalBinaryContents() {

    if (!lm_binary_) {
        std::cerr << "FlatBuffer binary not initialized." << std::endl;
        return;
    }

    const auto* trace_descriptors = lm_binary_->trace_descriptors();
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
    const auto* commands = lm_binary_->commands();
    if (!commands || commands->size() == 0) {
        std::cout << "No commands found in the binary." << std::endl;
    } else {
        std::cout << "Number of commands: " << commands->size() << std::endl;
        for (const auto* command : *commands) {
            if (!command) continue;

            auto cmd_type = command->cmd_type();
            switch (cmd_type) {
                case tt::target::CommandType::ReplayTraceCommand: {
                    const auto* cmd_variant = command->cmd_as_ReplayTraceCommand();
                    if (cmd_variant) {
                        std::cout << "ReplayTrace Command:" << std::endl;
                        std::cout << "  cq_id: " << cmd_variant->cq_id() << std::endl;
                        std::cout << "  tid: " << cmd_variant->tid() << std::endl;
                        std::cout << "  blocking: " << (cmd_variant->blocking() ? "true" : "false") << std::endl;
                    }
                    break;
                }
                case tt::target::CommandType::EnqueueTraceCommand: {
                    const auto* cmd_variant = command->cmd_as_EnqueueTraceCommand();
                    if (cmd_variant) {
                        std::cout << "EnqueueTrace Command:" << std::endl;
                        std::cout << "  cq_id: " << cmd_variant->cq_id() << std::endl;
                        std::cout << "  tid: " << cmd_variant->tid() << std::endl;
                        std::cout << "  blocking: " << (cmd_variant->blocking() ? "true" : "false") << std::endl;
                    }
                    break;
                }
                case tt::target::CommandType::LightMetalLoadTraceIdCommand: {
                    const auto* cmd_variant = command->cmd_as_LightMetalLoadTraceIdCommand();
                    if (cmd_variant) {
                        std::cout << "LightMetalLoadTraceId Command:" << std::endl;
                        std::cout << "  tid: " << cmd_variant->tid() << std::endl;
                        std::cout << "  cq_id: " << cmd_variant->cq_id() << std::endl;
                    }
                    break;
                }
                default:
                    std::cout << "Unsupported Command type: " << EnumNameCommandType(cmd_type) << std::endl;
                    break;
            }
        }
    }
}


void LightMetalReplay::setupDevices() {
    log_info(tt::LogMetalTrace, "Setting up system now...");

    // FIXME - Get these from lm_binary_ systemdesc once available. For now hardcode.
    const size_t buffer_size = 2048;
    this->arch_ = tt::ARCH::WORMHOLE_B0;
    const int device_id = 0;
    const auto dispatch_core_type = tt_metal::DispatchCoreType::WORKER;
    const chip_id_t mmio_device_id = 0;
    auto devices_map = tt::tt_metal::detail::CreateDevices({mmio_device_id}, 1, DEFAULT_L1_SMALL_SIZE, buffer_size, dispatch_core_type);
    this->device_ = devices_map.at(mmio_device_id);
}

//////////////////////////////////////
// Executor                         //
//////////////////////////////////////

// Some open questions...
// 1. How to pass Device* to replay functions? Can use a global variable for now.
// 2. How to pass other things like input tensors?
// 3. Can we fully encapsulate each host API command here.


// Get blob fropm filename
// Create Replay(blob)
// Replay.executeLightMetalBinary()
// Main entry point to execute a light metal binary blob and return success (0) /failure (1)
bool LightMetalReplay::executeLightMetalBinary() {

    if (!lm_binary_) {
        std::cerr << "FlatBuffer binary not initialized." << std::endl;
        return false;
    }

    try {
        example_code();

        const auto* trace_descriptors = lm_binary_->trace_descriptors();
        const auto* commands = lm_binary_->commands();
        if (!commands) {
            std::cerr << "No commands in binary." << std::endl;
            return false;
        }

        auto replay_registry = init_function_replay_registry();
        setupDevices();
        log_info(tt::LogMetalTrace, "{} - cmds: {} funcs: {} traces: {}", __FUNCTION__, commands->size(), replay_registry.size(), trace_descriptors->size());

        // Just loop over all commands, and execute. This is purposely kept simple for prototyping v0,
        // should expand to cover multiple program, devices, cqs, etc. FIXME
        for (const auto* cmd : *commands) {
            if (!cmd) continue;
            auto handlerIt = replay_registry.find(cmd->cmd_type());
            log_info(tt::LogMetalTrace, "Found command type: {}", cmd->cmd_type());
            if (handlerIt != replay_registry.end()) {
                handlerIt->second(cmd);
            } else {
                std::cerr << "Unknown Host API cmd, add to registry." << std::endl;
            }
        }

        return true;
    } catch (const std::exception& e) {
        log_fatal(e.what());
        return false;
    }
}


}  // namespace v0
}  // namespace tt::tt_metal
