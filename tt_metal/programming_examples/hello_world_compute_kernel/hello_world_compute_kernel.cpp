// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/metal.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {

    // Initialize Program and Device

    auto core = CoreRange({0, 0});
    int device_id = 0;
    auto device = v1::CreateDevice(device_id);
    auto cq = GetDefaultCommandQueue(device);
    auto program = v1::CreateProgram();

    // Configure and Create Void Kernel

    std::vector<uint32_t> compute_kernel_args = {};
    auto void_compute_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args
        }
    );

    // Configure Program and Start Program Execution on Device

    SetRuntimeArgs(program, void_compute_kernel_id, core, {});
    EnqueueProgram(cq, program, false);
    printf("Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    CloseDevice(device);

    return 0;

}
