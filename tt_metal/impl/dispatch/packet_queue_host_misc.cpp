// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tuple>
#include "buffers/buffer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"

namespace packet_queue_host
{

constexpr uint32_t PACKET_QUEUE_SCRATCH_BUFFER_SLOT_BYTES = 64; // sizeof(uint32_t); Size is extended to accomodate for remote shadow values
// static_assert(PACKET_QUEUE_SCRATCH_BUFFER_SLOT_BYTES == 4, "Size of uint32_t expected to be 4B for packet queue");

constexpr size_t packet_queue_buffer_set_wptr = 0;
constexpr size_t packet_queue_buffer_set_rptr_sent = 1;
constexpr size_t packet_queue_buffer_set_rptr_cleared = 2;

using packet_queue_buffer_set =
    std::tuple<std::vector<std::shared_ptr<Buffer>>, std::vector<std::shared_ptr<Buffer>>, std::vector<std::shared_ptr<Buffer>>>;

packet_queue_buffer_set make_buffer_set(Device* device, int n) {
    tt::tt_metal::InterleavedBufferConfig config{
        .device = device,
        .size = PACKET_QUEUE_SCRATCH_BUFFER_SLOT_BYTES,
        .page_size = PACKET_QUEUE_SCRATCH_BUFFER_SLOT_BYTES,
        .buffer_type = tt::tt_metal::BufferType::L1
    };

    std::vector<std::shared_ptr<Buffer>> wptr_buffers;
    std::vector<std::shared_ptr<Buffer>> rptr_sent_buffers;
    std::vector<std::shared_ptr<Buffer>> rptr_cleared_buffers;

    for (uint32_t i = 0; i < n; i++) {
        wptr_buffers.push_back(CreateBuffer(config));
        rptr_sent_buffers.push_back(CreateBuffer(config));
        rptr_cleared_buffers.push_back(CreateBuffer(config));
    }
    return {wptr_buffers, rptr_sent_buffers, rptr_cleared_buffers};
}

}
