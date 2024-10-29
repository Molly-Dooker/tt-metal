// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device_coord.hpp"
#include "mesh_device.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/kernels/runtime_args_data.hpp"

namespace tt::tt_metal::distributed::experimental {

using RuntimeArgsPerCore = std::vector<std::vector<RuntimeArgsData>>;

// Define an“infinite” LogicalDevice
constexpr uint32_t inf = std::numeric_limits<uint32_t>::max();
constexpr tt_xyz_coord InfDeviceCoord = tt_xyz_coord(inf, inf, inf);
constexpr tt_xyz_coord ZeroDeviceCoord = tt_xyz_coord(0, 0, 0);
const LogicalDeviceRange InfDeviceRange = LogicalDeviceRange(ZeroDeviceCoord, InfDeviceCoord);

class MeshWorkload {
public:
    MeshWorkload() : programs_{}, runtime_args_{} {}

    void add_program(std::shared_ptr<Program>& program);

    void add_program(const LogicalDeviceRange& device_range, std::shared_ptr<Program>& program);

    const std::unordered_map<LogicalDeviceRange, std::shared_ptr<Program>>& get_programs() const;

    bool grid_configured() const;
private:
    std::unordered_map<LogicalDeviceRange, std::shared_ptr<Program>> programs_;
    std::unordered_map<KernelHandle, std::unordered_map<LogicalDeviceRange, RuntimeArgsPerCore>> runtime_args_;
    bool grid_configured_ = false;
};

void EnqueueMeshWorkload(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id, std::shared_ptr<MeshWorkload> mesh_workload, bool blocking);

} // namespace tt::tt_metal::distributed::experimental
