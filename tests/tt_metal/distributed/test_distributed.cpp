// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/distributed/mesh_device_view.hpp"
#include "tt_metal/distributed/mesh_workload.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt::tt_metal::distributed::test {

static inline void skip_test_if_not_t3000() {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    const auto arch = tt::Cluster::instance().arch();
    const size_t num_devices = tt::Cluster::instance().number_of_devices();

    if (slow_dispatch) {
        GTEST_SKIP() << "Skipping Multi-Device test suite, since it can only be run in Fast Dispatch Mode.";
    }
    if (num_devices < 8 or arch != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping T3K Multi-Device test suite on non T3K machine.";
    }
}
class MeshDevice_T3000 : public ::testing::Test {
   protected:
    void SetUp() override {
        skip_test_if_not_t3000();
        this->mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 2)));
    }

    void TearDown() override {
        mesh_device_->close_devices();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

struct ProgramTestConfig {
    CoreRangeSet cr_set;
    uint32_t num_sems;
};

void initialize_dummy_semaphores(std::shared_ptr<Program> program, const std::variant<CoreRange, CoreRangeSet>& core_ranges, const std::vector<uint32_t>& init_values)
{
    for (uint32_t i = 0; i < init_values.size(); i++)
    {
        CreateSemaphore(*program, core_ranges, init_values[i]);
    }
}

bool test_mesh_workload_with_sems(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id, std::shared_ptr<experimental::MeshWorkload> mesh_workload, const ProgramTestConfig& program_config, const std::vector<std::vector<uint32_t>>& expected_semaphore_vals) {
    TT_ASSERT(program_config.cr_set.size() == expected_semaphore_vals.size());

    bool are_all_semaphore_values_correct = true;

    EnqueueMeshWorkload(mesh_device, cq_id, mesh_workload, true);
    auto program = mesh_workload->get_programs().at(experimental::InfDeviceRange);

    for (auto device : mesh_device->get_devices()) {
        std::cout << "Testing device: " << device->id() << std::endl;
        uint32_t expected_semaphore_vals_idx = 0;
        for (const CoreRange& core_range : program_config.cr_set.ranges())
        {
            const std::vector<uint32_t>& expected_semaphore_vals_for_core = expected_semaphore_vals[expected_semaphore_vals_idx];
            TT_ASSERT(expected_semaphore_vals_for_core.size() == program_config.num_sems);
            expected_semaphore_vals_idx++;
            for (const CoreCoord& core_coord : core_range)
            {
                std::vector<uint32_t> semaphore_vals;
                uint32_t expected_semaphore_vals_for_core_idx = 0;
                const uint32_t semaphore_buffer_size = program_config.num_sems * hal.get_alignment(HalMemType::L1);
                uint32_t semaphore_base = program->get_sem_base_addr(device, core_coord, CoreType::WORKER);
                tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, semaphore_base, semaphore_buffer_size, semaphore_vals);
                for (uint32_t i = 0; i < semaphore_vals.size(); i += (hal.get_alignment(HalMemType::L1) / sizeof(uint32_t)))
                {
                    const bool is_semaphore_value_correct = semaphore_vals[i] == expected_semaphore_vals_for_core[expected_semaphore_vals_for_core_idx];
                    expected_semaphore_vals_for_core_idx++;
                    if (!is_semaphore_value_correct)
                    {
                        are_all_semaphore_values_correct = false;
                    }
                }
            }
        }
    }
    return are_all_semaphore_values_correct;
}

std::shared_ptr<experimental::MeshWorkload> create_mesh_workload_with_semaphores(const ProgramTestConfig& program_config, const std::vector<uint32_t> expected_semaphore_values) {
    // Create program and initialize semaphores to it
    std::shared_ptr<Program> semaphore_program = std::make_shared<Program>();
    initialize_dummy_semaphores(semaphore_program, program_config.cr_set, expected_semaphore_values);
    // Insert the program into a MeshWorkload
    std::shared_ptr<experimental::MeshWorkload> mesh_workload = std::make_shared<experimental::MeshWorkload>();
    mesh_workload->add_program(semaphore_program);
    return mesh_workload;
}

TEST_F(MeshDevice_T3000, SimpleMeshDeviceTest) {
    EXPECT_EQ(mesh_device_->num_devices(), 8);
    EXPECT_EQ(mesh_device_->num_rows(), 2);
    EXPECT_EQ(mesh_device_->num_cols(), 4);
}

TEST_F(MeshDevice_T3000, TestMeshProgramSemaphores) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});
    std::vector<uint32_t> expected_semaphore_values = {};
    ProgramTestConfig semaphore_program_config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

    for (uint32_t initial_sem_value = 0; initial_sem_value < semaphore_program_config.num_sems; initial_sem_value++) {
        expected_semaphore_values.push_back(initial_sem_value);
    }

    auto mesh_workload = create_mesh_workload_with_semaphores(semaphore_program_config, expected_semaphore_values);
    EXPECT_TRUE(test_mesh_workload_with_sems(this->mesh_device_, 0, mesh_workload, semaphore_program_config, {expected_semaphore_values}));

}

}  // namespace tt::tt_metal::distributed::test
