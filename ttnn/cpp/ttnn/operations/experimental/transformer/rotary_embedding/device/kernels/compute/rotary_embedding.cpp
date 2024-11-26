// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

#include "debug/assert.h"
#include "debug/dprint.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    PACK( DPRINT << "======" << ENDL() );
    for (uint8_t r = 0; r < 32; ++ r) {
        SliceRange sr = SliceRange{.h0 = r, .h1 = (uint8_t)(r+1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        PACK( DPRINT << (uint)r << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL() );
    }
    PACK( DPRINT << "++++++" << ENDL() );
}

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(3);
    constexpr uint32_t scalar_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t Wt = get_compile_time_arg_val(10);

    cb_wait_front(scalar_cb, onetile);
    constexpr uint32_t untilized_sin_cb = get_compile_time_arg_val(14);
    constexpr uint32_t untilized_sin_sync_cb = get_compile_time_arg_val(15);
    constexpr uint32_t retilized_sin_cb = get_compile_time_arg_val(17);

    binary_op_init_common(in_cb, scalar_cb, out_cb);

    /* We replace the TILIZE_ROWS function with just the init and uninit functions.
       Moreover, we replace the circular buffers with compatible data types, for the
       subsequent operations, still the issue appears. Thus the functions should be
       buggy and no amount of tensix sync helps, so this is not a race either.
    */

    //TILIZE_ROWS(untilized_sin_cb, untilized_sin_sync_cb, retilized_sin_cb, Wt);
    tilize_init_short(in_cb, Wt);
    tilize_uninit(in_cb);
    //tensix_sync();

    uint32_t in1_idx = 0;
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
                reconfig_data_format(in_cb, scalar_cb);
                pack_reconfig_data_format(out_cb);
                cb_wait_front(in_cb, onetile);
                cb_reserve_back(out_cb, onetile);
                ACQ();
                mul_tiles_bcast_scalar_init_short();
                mul_tiles_bcast_scalar(in_cb, scalar_cb, 0, 0, 0);
                pack_tile(0, out_cb);
                REL();
                cb_push_back(out_cb, onetile);
                cb_pop_front(in_cb, onetile);
        }
    }


    /* This is a contrived issue, not related to the current bug. If we remove the tensix_sync(),
       the whole output data is corrupted, and only happens when the circular buffer data types
       are different from the previous operation. This one seems to be a race.
    */
    //tensix_sync();
    //binary_op_init_common(sin_cb, scalar_cb, untilized_sin_cb);

}
} // NAMESPACE
