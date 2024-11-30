// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

namespace ttnn {

namespace device {

Device& open_device(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
    tt::DevicePool::initialize({device_id}, 1, l1_small_size, trace_region_size, dispatch_core_config, {});
    return *(tt::DevicePool::instance().get_active_device(device_id));
}

bool is_device_open(int device_id) { return tt::DevicePool::instance().is_device_active(device_id); }

void enable_program_cache(Device& device) { device.enable_program_cache(); }

void disable_and_clear_program_cache(Device& device) { device.disable_and_clear_program_cache(); }

void close_device(Device& device) { tt::DevicePool::instance().close_device(device.id()); }

bool is_wormhole_or_blackhole(tt::ARCH arch) { return arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE; }

void deallocate_buffers(Device* device) {
    device->push_work([device]() mutable { device->deallocate_buffers(); });
}

SubDeviceManagerId create_sub_device_manager(
    Device* device, tt::stl::Span<const SubDevice> sub_devices, DeviceAddr local_l1_size) {
    SubDeviceManagerId sub_device_manager_id;
    device->push_work(
        [device, sub_devices, local_l1_size, &sub_device_manager_id] {
            sub_device_manager_id = device->create_sub_device_manager(sub_devices, local_l1_size);
        },
        true);
    return sub_device_manager_id;
}

void load_sub_device_manager(Device* device, SubDeviceManagerId sub_device_manager_id) {
    device->push_work([device, sub_device_manager_id] { device->load_sub_device_manager(sub_device_manager_id); });
}

void clear_loaded_sub_device_manager(Device* device) {
    device->push_work([device] { device->clear_loaded_sub_device_manager(); });
}

void remove_sub_device_manager(Device* device, SubDeviceManagerId sub_device_manager_id) {
    device->push_work([device, sub_device_manager_id] { device->remove_sub_device_manager(sub_device_manager_id); });
}

}  // namespace device

using namespace device;

}  // namespace ttnn
