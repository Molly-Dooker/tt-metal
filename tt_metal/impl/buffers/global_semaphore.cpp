// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/buffers/global_semaphore.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "tt_metal/common/assert.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/command_queue.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal {

GlobalSemaphore::GlobalSemaphore(
    v1::DeviceHandle device, const CoreRangeSet &cores, uint32_t initial_value, BufferType buffer_type) :
    device_(device), cores_(cores), initial_value_(initial_value), buffer_(this->setup_buffer(buffer_type)), host_buffer_(this->cores_.num_cores(), this->initial_value_) {
    this->reset_semaphore_value();
}

GlobalSemaphore::GlobalSemaphore(v1::DeviceHandle device, CoreRangeSet &&cores, uint32_t initial_value, BufferType buffer_type) :
    device_(device), cores_(cores), initial_value_(initial_value), buffer_(this->setup_buffer(buffer_type)), host_buffer_(this->cores_.num_cores(), this->initial_value_) {
    this->reset_semaphore_value();
}

v1::BufferHandle GlobalSemaphore::setup_buffer(BufferType buffer_type) {
    TT_FATAL(
        buffer_type == BufferType::L1 or buffer_type == BufferType::L1_SMALL,
        "Global semaphore can only be created for L1 buffer types");
    TT_FATAL(this->device_ != nullptr, "Device cannot be null");
    TT_FATAL(this->cores_.num_cores() > 0, "CoreRangeSet must have at least one core");
    uint32_t num_cores = this->cores_.num_cores();
    auto shard_parameters =
        ShardSpecBuffer(this->cores_, {1, 1}, ShardOrientation::ROW_MAJOR, false, {1, 1}, {num_cores, 1});

    return v1::CreateBuffer({
        this->device_,
        num_cores * sizeof(uint32_t),
        sizeof(uint32_t),
        buffer_type,
        TensorMemoryLayout::HEIGHT_SHARDED,
        shard_parameters});
}

std::unique_ptr<GlobalSemaphore> GlobalSemaphore::create(
    v1::DeviceHandle device, const CoreRangeSet &cores, uint32_t initial_value, BufferType buffer_type) {
    return std::make_unique<GlobalSemaphore>(device, cores, initial_value, buffer_type);
}
std::unique_ptr<GlobalSemaphore> GlobalSemaphore::create(
    v1::DeviceHandle device, CoreRangeSet &&cores, uint32_t initial_value, BufferType buffer_type) {
    return std::make_unique<GlobalSemaphore>(device, std::move(cores), initial_value, buffer_type);
}

DeviceAddr GlobalSemaphore::address() const { return buffer_->address(); }

void GlobalSemaphore::reset_semaphore_value() {
    // Blocking write of semaphore value to buffer
    if (this->device_->using_slow_dispatch()) {
        detail::WriteToBuffer(*this->buffer_, this->host_buffer_);
        tt::Cluster::instance().l1_barrier(this->device_->id());
    } else {
        EnqueueWriteBuffer(GetDefaultCommandQueue(this->device_), this->buffer_, this->host_buffer_, true);
    }
}

}  // namespace tt::tt_metal
