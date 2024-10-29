// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_workload.hpp"
#include "tt_metal/host_api.hpp"

namespace tt::tt_metal::distributed::experimental {

void MeshWorkload::add_program(std::shared_ptr<Program>& program) {
    // Broadcastable data-parallel program
    this->programs_.emplace(InfDeviceRange, program);
}

void MeshWorkload::add_program(const LogicalDeviceRange& device_range, std::shared_ptr<Program>& program) {
    // During enqueue we need to check device range
    this->grid_configured_ = true;
    this->programs_.emplace(device_range, program);
}

const std::unordered_map<LogicalDeviceRange, std::shared_ptr<Program>>& MeshWorkload::get_programs() const {
    return this->programs_;
}

bool MeshWorkload::grid_configured() const {
    return this->grid_configured_;
}

void EnqueueMeshWorkload(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id, std::shared_ptr<MeshWorkload> mesh_workload, bool blocking) {
    if (!mesh_workload->grid_configured()) {
        auto& broadcast_program = *(mesh_workload->get_programs().at(InfDeviceRange));
        const auto& [mesh_rows, mesh_cols] = mesh_device->shape();
        for (std::size_t row = 0; row < mesh_rows; row++) {
            for (std::size_t col = 0; col < mesh_cols; col++) {
                EnqueueProgram(mesh_device->get_device(row, col)->command_queue(cq_id), broadcast_program, false);
            }
        }
        if (blocking) {
            for (std::size_t row = 0; row < mesh_rows; row++) {
                for (std::size_t col = 0; col < mesh_cols; col++) {
                    Finish(mesh_device->get_device(row, col)->command_queue(cq_id));
                }
            }
        }
    }
}

} // namespace tt::tt_metal::distributed::experimental
