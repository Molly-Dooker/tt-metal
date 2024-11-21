// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal_replay.hpp"
#include "binary_generated.h"
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

bool example_code() {
    int device_id = 0;
    tt_metal::Device* device = tt_metal::CreateDevice(device_id, 1, DEFAULT_L1_SMALL_SIZE, 900000000);
    tt_metal::CommandQueue& cq = device->command_queue();
    tt_metal::Program program = tt_metal::CreateProgram();
    bool pass = tt_metal::CloseDevice(device);
    return pass;
}

// Main entry point to execute a light metal binary file and return success (0) /failure (1)
bool executeLightMetalBinary(const std::string& filename) {

    log_info(tt::LogMetalTrace, "Starting {} filename: {}", __FUNCTION__, filename);
    bool pass = true;

    try {
        pass = example_code();
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    return !pass;
}

}  // namespace v0
}  // namespace tt::tt_metal
