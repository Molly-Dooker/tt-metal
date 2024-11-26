// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "tt_metal/buffer.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/device.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "tt_metal/llrt/hal.hpp"

namespace tt::tt_metal {

inline namespace v0 {

class GlobalSemaphore {
   public:
    GlobalSemaphore(
        v1::DeviceHandle device, const CoreRangeSet &cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    GlobalSemaphore(
        v1::DeviceHandle device, CoreRangeSet &&cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    GlobalSemaphore(const GlobalSemaphore &) = default;
    GlobalSemaphore &operator=(const GlobalSemaphore &) = default;

    GlobalSemaphore(GlobalSemaphore &&) noexcept = default;
    GlobalSemaphore &operator=(GlobalSemaphore &&) noexcept = default;

    static std::unique_ptr<GlobalSemaphore> create(
        v1::DeviceHandle device, const CoreRangeSet &cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    static std::unique_ptr<GlobalSemaphore> create(
        v1::DeviceHandle device, CoreRangeSet &&cores, uint32_t initial_value, BufferType buffer_type = BufferType::L1);

    DeviceAddr address() const;

    void reset_semaphore_value();

   private:
    v1::BufferHandle setup_buffer(BufferType buffer_type);

    // GlobalSemaphore is implemented as a wrapper around a sharded buffer
    // This can be updated in the future to be its own container with optimized dispatch functions
    v1::DeviceHandle device_;
    CoreRangeSet cores_;
    uint32_t initial_value_ = 0;
    v1::BufferHandle buffer_;
    std::vector<uint32_t> host_buffer_;
};

}  // namespace v0

}  // namespace tt::tt_metal
