// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t cos_sin_offset = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);
    uint32_t Wbytes = get_arg_val<uint32_t>(5);

    DPRINT << "cs " << cos_sin_offset << " Wt " << Wt << " Wbytes " << Wbytes;

    constexpr uint32_t untilized_sin_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t untilized_sin_sync_cb_id = get_compile_time_arg_val(5);
    cb_wait_front(untilized_sin_cb_id, Wt);
    cb_reserve_back(untilized_sin_sync_cb_id, Wt);
    uint64_t sin_l1_read_addr = get_noc_addr(get_read_ptr(untilized_sin_cb_id)) + cos_sin_offset;
    uint32_t sin_l1_write_addr = get_read_ptr(untilized_sin_cb_id);
    noc_async_read(sin_l1_read_addr, sin_l1_write_addr, Wbytes);
    noc_async_read_barrier();
    cb_push_back(untilized_sin_sync_cb_id, Wt);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_id_out, onetile);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        noc_async_write_tile(i, s, l1_read_addr);

        noc_async_write_barrier();

        cb_pop_front(cb_id_out, onetile);
    }
}
