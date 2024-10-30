// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "dprint.h"

void kernel_main() {
    uint32_t inc_val = get_arg_val<uint32_t>(0);
    uint32_t dec_val = get_arg_val<uint32_t>(1);
    uint32_t num_loops = get_arg_val<uint32_t>(2);
    uint32_t sem_idx = get_arg_val<uint32_t>(3);
    uint32_t num_loops_multiplier = get_arg_val<uint32_t>(4);

    uint32_t biased_inc_val = inc_val - dec_val;

    for (uint32_t i = 0; i < num_loops_multiplier * num_loops; i++) {
        // Increment sem at idx 0
        noc_semaphore_inc(get_noc_addr(get_semaphore(sem_idx)), biased_inc_val);
    }
    noc_async_atomic_barrier();
}
