// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "noc.h"
#include "noc_nonblocking_api.h"
#include "noc_overlay_parameters.h"
#include "risc_attribs.h"
#include "risc_common.h"
#include "stream_io_map.h"
#include "tensix.h"
#include "tensix_types.h"
#ifdef PERF_DUMP
#include "risc_perf.h"
#endif
#include "firmware_common.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tensix_functions.h"
#include "c_tensix_core.h"

#include "kernel_includes.hpp"


uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_nonposted_atomics_acked[NUM_NOCS];
uint32_t noc_posted_writes_num_issued[NUM_NOCS];

void kernel_launch(uint32_t kernel_base_addr) {
    uint8_t go_message_signal;
    tt_l1_ptr mailboxes_t *const mailboxes = (tt_l1_ptr mailboxes_t *)(MEM_MAILBOX_BASE);

    while ((go_message_signal = mailboxes->go_message.signal) != RUN_MSG_GO) {
    }
  DeviceZoneScopedMainChildN("NCRISC-KERNEL");
#if defined(DEBUG_NULL_KERNELS) && !defined(DISPATCH_KERNEL)
#ifdef KERNEL_RUN_TIME
    uint64_t end_time = c_tensix_core::read_wall_clock() + KERNEL_RUN_TIME;
    while (c_tensix_core::read_wall_clock() < KERNEL_RUN_TIME);
#endif
#else
  extern uint32_t __kernel_init_local_l1_base[];
  extern uint32_t __fw_export_end_text[];
  do_crt1((
      uint32_t tt_l1_ptr *)(kernel_base_addr + (uint32_t)__kernel_init_local_l1_base - (uint32_t)__fw_export_end_text));

  if constexpr (NOC_MODE == DM_DEDICATED_NOC) {
      noc_local_state_init(NOC_INDEX);
    } else {
        noc_local_state_init(NOC_0);
        noc_local_state_init(NOC_1);
    }

    kernel_main();
#endif
}
