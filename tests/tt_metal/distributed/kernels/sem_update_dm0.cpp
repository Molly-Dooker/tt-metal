// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "dprint.h"

void kernel_main() {
    for (uint32_t i = 0; i < 20; i++) {
        volatile tt_l1_ptr uint32_t* sem_addr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
        // Increment sem at idx 0
        noc_semaphore_inc(get_noc_addr(get_semaphore(0)), 1);
    }
    noc_async_atomic_barrier();
}
