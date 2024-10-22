// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>

#include "tt_metal/impl/buffers/buffer.hpp"

namespace tt::tt_metal {

namespace detail {
struct TraceDescriptor {
    struct Descriptor {
        uint32_t num_completion_worker_cores = 0;
        uint32_t num_traced_programs_needing_go_signal_multicast = 0;
        uint32_t num_traced_programs_needing_go_signal_unicast = 0;
    };
    std::unordered_map<uint32_t, Descriptor> descriptors;
    std::vector<uint32_t> data;
};
}  // namespace detail

struct TraceBuffer {
    std::shared_ptr<detail::TraceDescriptor> desc;
    std::shared_ptr<Buffer> buffer;

    TraceBuffer(std::shared_ptr<detail::TraceDescriptor> desc, std::shared_ptr<Buffer> buffer);
    ~TraceBuffer();
};

}  // namespace tt::tt_metal
