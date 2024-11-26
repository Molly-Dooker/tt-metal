// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(3);
    uint32_t start_id = get_arg_val<uint32_t>(4);

    constexpr uint32_t input_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(4);
    constexpr bool input_is_dram = get_compile_time_arg_val(5) == 1;
    constexpr uint16_t scalar_value = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);

    constexpr uint32_t onetile = 1;
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const DataFormat input_data_format = get_dataformat(input_cb_id);

    const InterleavedAddrGenFast<input_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = input_tile_bytes,
        .data_format = input_data_format
    };

    // Fill tile with zeros
    const uint32_t scalar_tile_bytes = get_tile_size(scalar_cb_id);
    cb_reserve_back(scalar_cb_id, onetile);
    uint32_t l1_zeros_addr_in_scalar = get_write_ptr(scalar_cb_id);
    volatile tt_l1_ptr uint16_t* scalar_buffer = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_zeros_addr_in_scalar);
    scalar_buffer[0] = scalar_value;
    cb_push_back(scalar_cb_id, onetile);

    uint32_t input_curr_id = start_id;
    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = 0; i<num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            cb_reserve_back(input_cb_id, onetile);
            uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
            noc_async_read_tile(input_curr_id, s0, input_l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(input_cb_id, onetile);
            input_curr_id++;
       }
    }
}
