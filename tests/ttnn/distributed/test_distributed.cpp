// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <ttnn/core.hpp>
#include <ttnn/distributed/api.hpp>

namespace ttnn::distributed::test {

class DistributedTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DistributedTest, TestSystemMeshTearDownWithoutClose) {
    auto& sys = tt::tt_metal::distributed::SystemMesh::instance();
    auto mesh = ttnn::distributed::open_mesh_device(
        {2, 4}, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, tt::tt_metal::DispatchCoreType::WORKER);

    auto [rows, cols] = sys.get_shape();
    EXPECT_GT(rows, 0);
    EXPECT_GT(cols, 0);
}

}  // namespace ttnn::distributed::test
