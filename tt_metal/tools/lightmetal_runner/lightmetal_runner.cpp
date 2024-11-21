// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "tt_metal/impl/lightmetal/lightmetal_replay.hpp"
#include "tt_metal/common/logger.hpp"
#include "tt_metal/common/assert.hpp"

using namespace tt;

int main(int argc, char* argv[]) {

    // Process cmdline arguments
    TT_FATAL(argc == 2, "Invalid number of supplied arguments. Usage: ./lightmetal_runner <binary_file>");
    std::string filename = argv[1];

    // Execute the contents of the light metal binary.
    bool failed = tt::tt_metal::executeLightMetalBinary(filename);

    if (failed) {
        log_fatal("Binary {} failed to execute or encountered errors.", filename);
    } else {
        log_info(tt::LogMetalTrace, "Binary {} executed successfully", filename);
    }

    return failed;
}
