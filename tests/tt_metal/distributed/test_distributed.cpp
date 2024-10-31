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
        this->mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    }

    void TearDown() override {
        mesh_device_->close_devices();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

struct SemaphoreProgramTestConfig {
    CoreRangeSet cr_set;
    uint32_t num_sems;
};

struct CBConfig {
    uint32_t cb_id;
    uint32_t num_pages;
    uint32_t page_size;
    tt::DataFormat data_format;
};

struct CBProgramTestConfig {
    CoreRangeSet cr_set;
    std::vector<CBConfig> cb_config_vector;
};

void initialize_dummy_semaphores(std::shared_ptr<Program> program, const std::variant<CoreRange, CoreRangeSet>& core_ranges, const std::vector<uint32_t>& init_values)
{
    for (uint32_t i = 0; i < init_values.size(); i++)
    {
        CreateSemaphore(*program, core_ranges, init_values[i]);
    }
}

std::vector<CBHandle> initialize_dummy_circular_buffers(std::shared_ptr<Program> program, const CoreRangeSet& cr_set, const std::vector<CBConfig>& cb_configs)
{
    std::vector<CBHandle> cb_handles;
    for (uint32_t i = 0; i < cb_configs.size(); i++) {
        const CBConfig& cb_config = cb_configs[i];
        const uint32_t cb_id = cb_config.cb_id;
        const uint32_t cb_num_pages = cb_config.num_pages;
        const uint32_t page_size = cb_config.page_size;
        const uint32_t cb_size = cb_num_pages * page_size;
        const tt::DataFormat data_format = cb_config.data_format;
        const CircularBufferConfig circular_buffer_config = CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
        const CBHandle cb_handle = CreateCircularBuffer(*program, cr_set, circular_buffer_config);
        cb_handles.push_back(cb_handle);
    }
    return cb_handles;
}

bool test_mesh_workload_with_sems(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id, std::shared_ptr<experimental::MeshWorkload> mesh_workload, const SemaphoreProgramTestConfig& program_config, const std::vector<uint32_t>& expected_semaphore_vals) {
    TT_ASSERT(program_config.cr_set.size() == expected_semaphore_vals.size());

    bool are_all_semaphore_values_correct = true;

    EnqueueMeshWorkload(mesh_device, cq_id, mesh_workload, true);
    auto program = mesh_workload->get_programs().at(experimental::InfDeviceRange);

    for (auto device : mesh_device->get_devices()) {
        for (const CoreRange& core_range : program_config.cr_set.ranges())
        {
            const std::vector<uint32_t>& expected_semaphore_vals_for_core = expected_semaphore_vals;
            TT_ASSERT(expected_semaphore_vals_for_core.size() == program_config.num_sems);
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

std::shared_ptr<experimental::MeshWorkload> create_mesh_workload_with_semaphores(const SemaphoreProgramTestConfig& program_config, const std::vector<uint32_t> expected_semaphore_values) {
    // Create program and initialize semaphores on it
    std::shared_ptr<Program> semaphore_program = std::make_shared<Program>();
    initialize_dummy_semaphores(semaphore_program, program_config.cr_set, expected_semaphore_values);
    // Insert the program into a MeshWorkload
    std::shared_ptr<experimental::MeshWorkload> mesh_workload = std::make_shared<experimental::MeshWorkload>();
    mesh_workload->add_program(semaphore_program);
    return mesh_workload;
}

std::shared_ptr<experimental::MeshWorkload> create_mesh_workload_with_cbs(const CBProgramTestConfig& program_config) {
    // Create program and initialize CBs on it
    std::shared_ptr<Program> cb_program = std::make_shared<Program>();
    initialize_dummy_circular_buffers(cb_program, program_config.cr_set, program_config.cb_config_vector);
    // Insert the program into a MeshWorkload
    std::shared_ptr<experimental::MeshWorkload> mesh_workload = std::make_shared<experimental::MeshWorkload>();
    mesh_workload->add_program(cb_program);
    return mesh_workload;
}

bool cb_config_successful(std::shared_ptr<MeshDevice> mesh_device, std::shared_ptr<experimental::MeshWorkload> mesh_workload, const CBProgramTestConfig& program_config) {
    bool pass = true;

    std::vector<uint32_t> cb_config_vector;
    uint32_t cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);
    auto program = mesh_workload->get_programs().at(experimental::InfDeviceRange);

    for (auto device : mesh_device->get_devices()) {
        uint32_t l1_unreserved_base = device->get_base_allocator_addr(HalMemType::L1);
        for (const CoreRange& core_range : program_config.cr_set.ranges()) {
            for (const CoreCoord& core_coord : core_range) {
                tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord,
                    program->get_sem_base_addr(device, core_coord, CoreType::WORKER),
                    cb_config_buffer_size, cb_config_vector);

                uint32_t cb_addr = l1_unreserved_base;
                for (uint32_t i = 0; i < program_config.cb_config_vector.size(); i++) {
                    const uint32_t index = program_config.cb_config_vector[i].cb_id * sizeof(uint32_t);
                    const uint32_t cb_num_pages = program_config.cb_config_vector[i].num_pages;
                    const uint32_t cb_size = cb_num_pages * program_config.cb_config_vector[i].page_size;
                    const bool addr_match = cb_config_vector.at(index) == ((cb_addr) >> 4);
                    const bool size_match = cb_config_vector.at(index + 1) == (cb_size >> 4);
                    const bool num_pages_match = cb_config_vector.at(index + 2) == cb_num_pages;
                    pass &= (addr_match and size_match and num_pages_match);
                    cb_addr += cb_size;
                }
            }
        }
    }

    return pass;
}

bool test_mesh_workload_with_cbs(std::shared_ptr<MeshDevice> mesh_device, uint8_t cq_id, std::shared_ptr<experimental::MeshWorkload> mesh_workload, const CBProgramTestConfig& program_config) {
    EnqueueMeshWorkload(mesh_device, cq_id, mesh_workload, true);
    return cb_config_successful(mesh_device, mesh_workload, program_config);;

}

TEST_F(MeshDevice_T3000, SimpleMeshDeviceTest) {
    EXPECT_EQ(mesh_device_->num_devices(), 8);
    EXPECT_EQ(mesh_device_->num_rows(), 2);
    EXPECT_EQ(mesh_device_->num_cols(), 4);
}

TEST_F(MeshDevice_T3000, TestMeshProgramSemaphoresSingleCoreRange) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});
    std::vector<uint32_t> expected_semaphore_values = {};
    SemaphoreProgramTestConfig semaphore_program_config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

    for (uint32_t initial_sem_value = 0; initial_sem_value < semaphore_program_config.num_sems; initial_sem_value++) {
        expected_semaphore_values.push_back(initial_sem_value);
    }

    auto mesh_workload = create_mesh_workload_with_semaphores(semaphore_program_config, expected_semaphore_values);
    EXPECT_TRUE(test_mesh_workload_with_sems(this->mesh_device_, 0, mesh_workload, semaphore_program_config, expected_semaphore_values));

}

TEST_F(MeshDevice_T3000, TestMeshProgramSemaphoresMultiCoreRange) {
    // Define Core Range Sets to which the Semaphores will be written
    CoreRange first_cr({0, 0}, {1, 1});
    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    CoreRange second_cr({worker_grid_size.x - 2, worker_grid_size.y - 2}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set(std::vector{first_cr, second_cr});

    SemaphoreProgramTestConfig semaphore_program_config = {.cr_set = cr_set, .num_sems = NUM_SEMAPHORES};

    std::vector<uint32_t> expected_semaphore_values = {};
    for (uint32_t initial_sem_value = 0; initial_sem_value < semaphore_program_config.num_sems; initial_sem_value++) {
        expected_semaphore_values.push_back(initial_sem_value);
    }

    auto mesh_workload = create_mesh_workload_with_semaphores(semaphore_program_config, expected_semaphore_values);
    EXPECT_TRUE(test_mesh_workload_with_sems(this->mesh_device_, 0, mesh_workload, semaphore_program_config, expected_semaphore_values));

}

TEST_F(MeshDevice_T3000, TestMeshProgramSingleCBSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config = {.cb_id=0, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBProgramTestConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config} };

    auto mesh_workload = create_mesh_workload_with_cbs(config);
    EXPECT_TRUE(test_mesh_workload_with_cbs(this->mesh_device_, 0, mesh_workload, config));
}

TEST_F(MeshDevice_T3000, TestMeshProgramMultiCBSingleCore) {
    CoreRange cr({0, 0}, {0, 0});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 1, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 0, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 24, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 16, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    CBProgramTestConfig config = {.cr_set = cr_set, .cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3}};
    auto mesh_workload = create_mesh_workload_with_cbs(config);
    EXPECT_TRUE(test_mesh_workload_with_cbs(this->mesh_device_, 0, mesh_workload, config));
}

TEST_F(MeshDevice_T3000, TestMeshProgramMultiCBMultiCore) {
    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};

    std::vector <CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    CoreCoord worker_grid_size = this->mesh_device_->compute_with_storage_grid_size();
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    CBProgramTestConfig config = {.cr_set = cr_set, .cb_config_vector = cb_config_vector};

    auto mesh_workload = create_mesh_workload_with_cbs(config);
    EXPECT_TRUE(test_mesh_workload_with_cbs(this->mesh_device_, 0, mesh_workload, config));
}

std::shared_ptr<Program> create_dummy_program_0(std::shared_ptr<MeshDevice> mesh_device, std::vector<uint32_t>& rtas, CoreRangeSet& cr_set, KernelHandle& dm0_kernel, KernelHandle& dm1_kernel) {
    std::shared_ptr<Program> program = std::make_shared<Program>();

    CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();
    CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    cr_set = {cr};

    dm0_kernel = CreateKernel(
                                    *program,
                                    "tests/tt_metal/distributed/kernels/sem_update_dm0.cpp",
                                    cr_set,
                                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
                                );
    dm1_kernel = CreateKernel(
                                    *program,
                                    "tests/tt_metal/distributed/kernels/sem_update_dm1.cpp",
                                    cr_set,
                                    DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
                                );
    CreateSemaphore(*program, cr_set, 0);
    CreateSemaphore(*program, cr_set, 1);

    // Initialize Runtime Args
    std::vector<uint32_t> dm0_rtas = {rtas[0], rtas[1], rtas[2], rtas[3]};
    std::vector<uint32_t> dm1_rtas = {rtas[0], rtas[1], rtas[2], rtas[4], rtas[5]};
    SetRuntimeArgs(*program, dm0_kernel, cr_set, dm0_rtas);
    SetRuntimeArgs(*program, dm1_kernel, cr_set, dm1_rtas);
    return program;
}

void update_runtime_args_for_dummy_program_0(std::shared_ptr<Program> program, std::vector<uint32_t>& rtas, CoreRangeSet& cr_set, KernelHandle& dm0_kernel, KernelHandle& dm1_kernel) {
    SetRuntimeArgs(*program, dm0_kernel, cr_set, {rtas[0], rtas[1], rtas[2], rtas[3]});
    SetRuntimeArgs(*program, dm1_kernel, cr_set, {rtas[0], rtas[1], rtas[2], rtas[4], rtas[5]});
}

// std::shared_ptr<Program> create_dummy_program_1(std::shared_ptr<MeshDevice> mesh_device) {
//     return create_dummy_program_0(mesh_device);
// }

TEST_F(MeshDevice_T3000, TestMeshProgramBroadcast) {
    const uint32_t NUM_LOOPS = 5;

    uint32_t kernel_inc_val = 2;
    uint32_t kernel_dec_val = 1;
    uint32_t num_kernel_loops = 1;
    uint32_t sem_idx_dm_0 = 0;
    uint32_t sem_idx_dm_1 = 1;
    uint32_t num_loops_multiplier = 1;

    std::vector<uint32_t> all_rtas = {kernel_inc_val, kernel_dec_val, num_kernel_loops, sem_idx_dm_0, sem_idx_dm_1, num_loops_multiplier};
    CoreRangeSet program_cr_set;
    KernelHandle dm0_kernel_handle;
    KernelHandle dm1_kernel_handle;
    std::shared_ptr<Program> program = create_dummy_program_0(this->mesh_device_, all_rtas, program_cr_set, dm0_kernel_handle, dm1_kernel_handle);
    // Create Mesh Workload
    std::shared_ptr<experimental::MeshWorkload> mesh_workload = std::make_shared<experimental::MeshWorkload>();
    mesh_workload->add_program(program);

    for (auto loop = 0; loop < NUM_LOOPS; loop++) {
        EnqueueMeshWorkload(this->mesh_device_, 0, mesh_workload, true);
        for (auto device: this->mesh_device_->get_devices()) {
            for (const CoreRange& core_range : program_cr_set.ranges())
            {
                for (const CoreCoord& core_coord : core_range)
                {
                    std::vector<uint32_t> dm0_result = {};
                    std::vector<uint32_t> dm1_result = {};
                    uint32_t dm0_result_addr = program->get_sem_base_addr(device, core_coord, CoreType::WORKER);
                    uint32_t dm1_result_addr = program->get_sem_base_addr(device, core_coord, CoreType::WORKER) + hal.get_alignment(HalMemType::L1);
                    tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, dm0_result_addr, sizeof(uint32_t), dm0_result);
                    tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, dm1_result_addr, sizeof(uint32_t), dm1_result);
                    EXPECT_EQ(dm0_result.at(0), (all_rtas.at(0) - all_rtas.at(1)) * num_kernel_loops);
                    EXPECT_EQ(dm1_result.at(0), (all_rtas.at(0) -  all_rtas.at(1)) * num_kernel_loops * all_rtas.at(5) + 1);
                }
            }
        }
        // Update RTAs
        all_rtas.at(0) *= 2;
        all_rtas.at(1) *= 2;
        all_rtas.at(5) *= 2;
        update_runtime_args_for_dummy_program_0(program, all_rtas, program_cr_set, dm0_kernel_handle, dm1_kernel_handle);
    }
}

TEST_F(MeshDevice_T3000, TestMeshProgramMultiBroadcast) {
    LogicalDeviceRange grid_0(tt_xyz_coord(0, 0, 0), tt_xyz_coord(1, 1, 0));
    LogicalDeviceRange grid_1(tt_xyz_coord(2, 0, 0), tt_xyz_coord(3, 1, 0));

    const uint32_t NUM_LOOPS = 5;

    // Initialize Runtime Args for grid 0
    uint32_t grid_0_kernel_inc_val = 2;
    uint32_t grid_0_kernel_dec_val = 1;
    uint32_t grid_0_num_kernel_loops = 1;
    uint32_t grid_0_sem_idx_dm_0 = 0;
    uint32_t grid_0_sem_idx_dm_1 = 1;
    uint32_t grid_0_num_loops_multiplier = 1;

    // Initialize Runtime Args for grid 1
    uint32_t grid_1_kernel_inc_val = 3;
    uint32_t grid_1_kernel_dec_val = 1;
    uint32_t grid_1_num_kernel_loops = 2;
    uint32_t grid_1_sem_idx_dm_0 = 0;
    uint32_t grid_1_sem_idx_dm_1 = 1;
    uint32_t grid_1_num_loops_multiplier = 1;

    std::vector<uint32_t> grid_0_rtas = {grid_0_kernel_inc_val, grid_0_kernel_dec_val, grid_0_num_kernel_loops, grid_0_sem_idx_dm_0, grid_0_sem_idx_dm_1, grid_0_num_loops_multiplier};
    std::vector<uint32_t> grid_1_rtas = {grid_1_kernel_inc_val, grid_1_kernel_dec_val, grid_1_num_kernel_loops, grid_1_sem_idx_dm_0, grid_1_sem_idx_dm_1, grid_1_num_loops_multiplier};

    CoreRangeSet grid_0_program_cr_set;
    CoreRangeSet grid_1_program_cr_set;

    KernelHandle grid_0_dm0_kernel_handle;
    KernelHandle grid_0_dm1_kernel_handle;
    KernelHandle grid_1_dm0_kernel_handle;
    KernelHandle grid_1_dm1_kernel_handle;

    std::shared_ptr<Program> grid0_program = create_dummy_program_0(this->mesh_device_, grid_0_rtas, grid_0_program_cr_set, grid_0_dm0_kernel_handle, grid_0_dm1_kernel_handle);
    std::shared_ptr<Program> grid1_program = create_dummy_program_0(this->mesh_device_, grid_1_rtas, grid_1_program_cr_set, grid_1_dm0_kernel_handle, grid_1_dm1_kernel_handle);

    // Create Mesh Workload, running 2 programs
    std::shared_ptr<experimental::MeshWorkload> mesh_workload = std::make_shared<experimental::MeshWorkload>();
    mesh_workload->add_program(grid_0, grid0_program);
    mesh_workload->add_program(grid_1, grid1_program);

    for (auto loop = 0; loop < NUM_LOOPS; loop++) {
        EnqueueMeshWorkload(this->mesh_device_, 0, mesh_workload, true);

        for (std::size_t row = grid_0.start_coord.y; row < grid_0.end_coord.y + 1; row++) {
            for (std::size_t col = grid_0.start_coord.x; col < grid_0.end_coord.x + 1; col++) {
                auto device = this->mesh_device_->get_device(row, col);
                for (const CoreRange& core_range : grid_0_program_cr_set.ranges())
                {
                    for (const CoreCoord& core_coord : core_range)
                    {
                        std::vector<uint32_t> dm0_result = {};
                        std::vector<uint32_t> dm1_result = {};
                        uint32_t dm0_result_addr = grid0_program->get_sem_base_addr(device, core_coord, CoreType::WORKER);
                        uint32_t dm1_result_addr = grid0_program->get_sem_base_addr(device, core_coord, CoreType::WORKER) + hal.get_alignment(HalMemType::L1);
                        tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, dm0_result_addr, sizeof(uint32_t), dm0_result);
                        tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, dm1_result_addr, sizeof(uint32_t), dm1_result);
                        EXPECT_EQ(dm0_result.at(0), (grid_0_rtas.at(0) - grid_0_rtas.at(1)) * grid_0_num_kernel_loops);
                        EXPECT_EQ(dm1_result.at(0), (grid_0_rtas.at(0) - grid_0_rtas.at(1)) * grid_0_num_kernel_loops * grid_0_rtas.at(5) + 1);
                    }
                }
            }
        }

        for (std::size_t row = grid_1.start_coord.y; row < grid_1.end_coord.y + 1; row++) {
            for (std::size_t col = grid_1.start_coord.x; col < grid_1.end_coord.x + 1; col++) {
                auto device = this->mesh_device_->get_device(row, col);
                for (const CoreRange& core_range : grid_1_program_cr_set.ranges())
                {
                    for (const CoreCoord& core_coord : core_range)
                    {
                        std::vector<uint32_t> dm0_result = {};
                        std::vector<uint32_t> dm1_result = {};
                        uint32_t dm0_result_addr = grid1_program->get_sem_base_addr(device, core_coord, CoreType::WORKER);
                        uint32_t dm1_result_addr = grid1_program->get_sem_base_addr(device, core_coord, CoreType::WORKER) + hal.get_alignment(HalMemType::L1);
                        tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, dm0_result_addr, sizeof(uint32_t), dm0_result);
                        tt::tt_metal::detail::ReadFromDeviceL1(device, core_coord, dm1_result_addr, sizeof(uint32_t), dm1_result);
                        EXPECT_EQ(dm0_result.at(0), (grid_1_rtas.at(0) - grid_1_rtas.at(1)) * grid_1_num_kernel_loops);
                        EXPECT_EQ(dm1_result.at(0), (grid_1_rtas.at(0) - grid_1_rtas.at(1)) * grid_1_num_kernel_loops * grid_1_rtas.at(5) + 1);
                    }
                }
            }
        }

        // Update RTAs grid_0
        grid_0_rtas.at(0) *= 2;
        grid_0_rtas.at(1) *= 2;
        grid_0_rtas.at(5) *= 2;
        // Update RTAs grid_1
        grid_1_rtas.at(0) *= 3;
        grid_1_rtas.at(1) *= 3;
        grid_1_rtas.at(5) *= 3;

        update_runtime_args_for_dummy_program_0(grid0_program, grid_0_rtas, grid_0_program_cr_set, grid_0_dm0_kernel_handle, grid_0_dm1_kernel_handle);
        update_runtime_args_for_dummy_program_0(grid1_program, grid_1_rtas, grid_1_program_cr_set, grid_1_dm0_kernel_handle, grid_1_dm1_kernel_handle);
    }
}

}  // namespace tt::tt_metal::distributed::test
