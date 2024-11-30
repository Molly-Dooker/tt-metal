// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "tt_metal/impl/buffers/global_semaphore.hpp"
#include "tt_metal/host_api.hpp"

namespace ttnn::global_semaphore {

std::shared_ptr<GlobalSemaphore> create_global_semaphore(
    Device* device, const CoreRangeSet& cores, uint32_t initial_value, BufferType buffer_type) {
    std::shared_ptr<GlobalSemaphore> global_semaphore = nullptr;
    device->push_work(
        [device, cores, initial_value, buffer_type, &global_semaphore] {
            global_semaphore = GlobalSemaphore::create(device, cores, initial_value, buffer_type);
        },
        true);
    return global_semaphore;
}

DeviceAddr get_global_semaphore_address(const std::shared_ptr<GlobalSemaphore>& global_semaphore) {
    auto device = global_semaphore->device();
    DeviceAddr address = 0;
    device->push_work([global_semaphore, &address] { address = global_semaphore->address(); }, true);
    return address;
}

void reset_global_semaphore_value(const std::shared_ptr<GlobalSemaphore>& global_semaphore) {
    auto device = global_semaphore->device();
    device->push_work([global_semaphore] { global_semaphore->reset_semaphore_value(); });
}

}  // namespace ttnn::global_semaphore
