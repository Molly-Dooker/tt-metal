// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t col_start_tile_id =
        get_arg_val<uint32_t>(1);  // Start id in column major order. This should be the start of a column
    uint32_t curr_col_in_batch = get_arg_val<uint32_t>(2);
    uint32_t num_cols = get_arg_val<uint32_t>(3);  // number of cols to read

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt = get_compile_time_arg_val(2);
    constexpr uint32_t HtWt = get_compile_time_arg_val(3);
    constexpr uint32_t row_chunk = get_compile_time_arg_val(4);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

#ifdef REDUCE_SCALER
    constexpr uint32_t cb_id_in2 = 2;
    constexpr uint32_t scalar = get_compile_time_arg_val(5);
    generate_reduce_scaler(cb_id_in2, scalar);
#endif

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t w = curr_col_in_batch;

    for (uint32_t i = 0; i < num_cols; i += row_chunk) {
        uint32_t chunk_end = std::min(i + row_chunk, num_cols);
        uint32_t curr_id = col_start_tile_id;
        uint32_t reset_curr_id = curr_id;
        uint32_t reset_w = w;
        uint32_t reset_col_start = col_start_tile_id;

        // row wise read for a chunk of columns
        for (uint32_t j = 0; j < Ht; ++j) {
            w = reset_w;
            col_start_tile_id = reset_col_start;
            for (uint32_t k = i; k < chunk_end; ++k) {


                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read_tile(curr_id, s, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);

                ++w;

                if (w == Wt) {
                    col_start_tile_id = curr_id + (Ht - j - 1) * Wt + 1;
                    curr_id = col_start_tile_id + j * Wt;
                    w = 0;
                } else {
                    ++curr_id;
                    ++col_start_tile_id;
                }
            }
            curr_id = reset_curr_id + (j + 1) * Wt;  // stride in H
        }
    }
}
