// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ethernet/dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/cq_helpers.hpp"
#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"

#define NUM_BIDIR_TUNNELS 1
#define NUM_TUNNEL_QUEUES (NUM_BIDIR_TUNNELS * 2)

packet_input_queue_state_t input_queues[NUM_TUNNEL_QUEUES];
packet_output_queue_state_t output_queues[NUM_TUNNEL_QUEUES];

constexpr uint32_t endpoint_id_start_index = get_compile_time_arg_val(0);
constexpr uint32_t tunnel_lanes = get_compile_time_arg_val(1);
constexpr uint32_t in_queue_start_addr_words = get_compile_time_arg_val(2);
constexpr uint32_t in_queue_size_words = get_compile_time_arg_val(3);
constexpr uint32_t in_queue_size_bytes = in_queue_size_words * PACKET_WORD_SIZE_BYTES;
static_assert(is_power_of_2(in_queue_size_words), "in_queue_size_words must be a power of 2");
static_assert(tunnel_lanes <= NUM_TUNNEL_QUEUES, "cannot have more than 2 tunnel directions.");
static_assert(tunnel_lanes, "tunnel directions cannot be 0. 1 => Unidirectional. 2 => Bidirectional");

constexpr uint32_t remote_receiver_x[NUM_TUNNEL_QUEUES] = {
    (get_compile_time_arg_val(4) & 0xFF), (get_compile_time_arg_val(5) & 0xFF)};

constexpr uint32_t remote_receiver_y[NUM_TUNNEL_QUEUES] = {
    (get_compile_time_arg_val(4) >> 8) & 0xFF, (get_compile_time_arg_val(5) >> 8) & 0xFF};

constexpr uint32_t remote_receiver_queue_id[NUM_TUNNEL_QUEUES] = {
    (get_compile_time_arg_val(4) >> 16) & 0xFF, (get_compile_time_arg_val(5) >> 16) & 0xFF};

static_assert(remote_receiver_queue_id[0] < PACKET_QUEUE_MAX_ID, "remote_receiver_queue_id[0] has exceeded the maximum supported queue id");
static_assert(remote_receiver_queue_id[1] < PACKET_QUEUE_MAX_ID, "remote_receiver_queue_id[1] has exceeded the maximum supported queue id");

constexpr DispatchRemoteNetworkType remote_receiver_network_type[NUM_TUNNEL_QUEUES] = {
    static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(4) >> 24) & 0xFF),
    static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(5) >> 24) & 0xFF)};

constexpr uint32_t remote_receiver_queue_start_addr_words[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(6), get_compile_time_arg_val(8)};

constexpr uint32_t remote_receiver_queue_size_words[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(7), get_compile_time_arg_val(9)};

static_assert(
    is_power_of_2(remote_receiver_queue_size_words[0]), "remote_receiver_queue_size_words must be a power of 2");
static_assert(
    is_power_of_2(remote_receiver_queue_size_words[1]), "remote_receiver_queue_size_words must be a power of 2");

constexpr uint32_t remote_sender_x[NUM_TUNNEL_QUEUES] = {
    (get_compile_time_arg_val(10) & 0xFF), (get_compile_time_arg_val(11) & 0xFF)};

constexpr uint32_t remote_sender_y[NUM_TUNNEL_QUEUES] = {
    (get_compile_time_arg_val(10) >> 8) & 0xFF, (get_compile_time_arg_val(11) >> 8) & 0xFF};

constexpr uint32_t remote_sender_queue_id[NUM_TUNNEL_QUEUES] = {
    (get_compile_time_arg_val(10) >> 16) & 0xFF, (get_compile_time_arg_val(11) >> 16) & 0xFF};

static_assert(remote_sender_queue_id[0] < PACKET_QUEUE_MAX_ID, "remote_sender_queue_id[0] has exceeded the maximum supported queue id");
static_assert(remote_sender_queue_id[1] < PACKET_QUEUE_MAX_ID, "remote_sender_queue_id[1] has exceeded the maximum supported queue id");

constexpr DispatchRemoteNetworkType remote_sender_network_type[NUM_TUNNEL_QUEUES] = {
    static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(10) >> 24) & 0xFF),
    static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(11) >> 24) & 0xFF)};

constexpr uint32_t test_results_buf_addr_arg = get_compile_time_arg_val(12);
constexpr uint32_t test_results_buf_size_bytes = get_compile_time_arg_val(13);

// careful, may be null
tt_l1_ptr uint32_t* const test_results = reinterpret_cast<tt_l1_ptr uint32_t*>(test_results_buf_addr_arg);

constexpr uint32_t timeout_cycles = get_compile_time_arg_val(14);
constexpr uint32_t inner_stop_mux_d_bypass = get_compile_time_arg_val(15);

// Eth tunneler inputs
constexpr uint32_t eth_tunneler_in_local_wptr_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(16),
    get_compile_time_arg_val(17)};

constexpr uint32_t eth_tunneler_in_remote_rptr_sent_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(18),
    get_compile_time_arg_val(19)};

constexpr uint32_t eth_tunneler_in_remote_rptr_cleared_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(20),
    get_compile_time_arg_val(21)};

// Eth tunneler outputs
constexpr uint32_t eth_tunneler_out_local_rptr_sent_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(22),
    get_compile_time_arg_val(23)};

constexpr uint32_t eth_tunneler_out_local_rptr_cleared_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(24),
    get_compile_time_arg_val(25)};

constexpr uint32_t eth_tunneler_out_remote_wptr_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(26),
    get_compile_time_arg_val(27)};

void kernel_main() {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    write_buffer_to_l1(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX, 0xff000000);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX + 1, 0xbb000000);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX + 2, 0xAABBCCDD);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX + 3, 0xDDCCBBAA);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX + 4, endpoint_id_start_index);

    for (uint32_t i = 0; i < tunnel_lanes; i++) {
        input_queues[i].init(
            i,
            in_queue_start_addr_words + i * in_queue_size_words,
            in_queue_size_words,
            remote_sender_x[i],
            remote_sender_y[i],
            remote_sender_queue_id[i],
            remote_sender_network_type[i],
            eth_tunneler_in_local_wptr_addr[i],
            eth_tunneler_in_remote_rptr_sent_addr[i],
            eth_tunneler_in_remote_rptr_cleared_addr[i]);

        output_queues[i].init(
            i + NUM_TUNNEL_QUEUES,
            remote_receiver_queue_start_addr_words[i],
            remote_receiver_queue_size_words[i],
            remote_receiver_x[i],
            remote_receiver_y[i],
            remote_receiver_queue_id[i],
            remote_receiver_network_type[i],
            &input_queues[i],
            1,
            eth_tunneler_out_local_rptr_sent_addr[i],
            eth_tunneler_out_local_rptr_cleared_addr[i],
            eth_tunneler_out_remote_wptr_addr[i]);
    }

    input_queues[0].tunneler_local_recv_ptr=
            reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(2, STREAM_REMOTE_SRC_REG_INDEX));
    input_queues[0].tunneler_local_sent_ptr =
        reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(3, STREAM_REMOTE_SRC_REG_INDEX));

    input_queues[0].remote_tunneler_recv_addr = STREAM_REG_ADDR(4, STREAM_REMOTE_SRC_REG_INDEX);
    input_queues[0].remote_tunneler_sent_addr = STREAM_REG_ADDR(5, STREAM_REMOTE_SRC_REG_INDEX);

    output_queues[0].tunneler_local_recv_ptr =
        reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(4, STREAM_REMOTE_SRC_REG_INDEX));
    output_queues[0].tunneler_local_sent_ptr =
        reinterpret_cast<volatile uint32_t*>(STREAM_REG_ADDR(5, STREAM_REMOTE_SRC_REG_INDEX));

    output_queues[0].remote_tunneler_recv_addr = STREAM_REG_ADDR(2, STREAM_REMOTE_SRC_REG_INDEX);
    output_queues[0].remote_tunneler_sent_addr = STREAM_REG_ADDR(3, STREAM_REMOTE_SRC_REG_INDEX);

    *input_queues[0].tunneler_local_recv_ptr = 0;
    *input_queues[0].tunneler_local_sent_ptr = 0;
    *output_queues[0].tunneler_local_recv_ptr = 0;
    *output_queues[0].tunneler_local_sent_ptr = 0;

    if (!wait_all_src_dest_ready(input_queues, tunnel_lanes, output_queues, tunnel_lanes, timeout_cycles)) {
        write_buffer_to_l1(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX, 0xff000001);

    bool timeout = false;
    bool all_outputs_finished = false;
    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t start_timestamp = get_timestamp();
    uint32_t progress_timestamp = start_timestamp & 0xFFFFFFFF;
    uint32_t heartbeat = 0;
    while (!all_outputs_finished && !timeout) {
        IDLE_ERISC_HEARTBEAT_AND_RETURN(heartbeat);
        iter++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_progress = get_timestamp_32b() - progress_timestamp;
            if (cycles_since_progress > timeout_cycles) {
                timeout = true;
                break;
            }
        }

        all_outputs_finished = true;
        for (uint32_t i = 0; i < tunnel_lanes; i++) {
            input_queues[i].handle_recv();
            output_queues[i].handle_recv();

            if (input_queues[i].is_waiting_for_ack() || output_queues[i].is_waiting_for_ack()) {
                all_outputs_finished = false;
                continue;
            }

            if (input_queues[i].get_curr_packet_valid()) {
                bool full_packet_sent;
                uint32_t words_sent =
                    output_queues[i].forward_data_from_input(0, full_packet_sent, input_queues[i].get_end_of_cmd());
                progress_timestamp = get_timestamp_32b();
            }
            output_queues[i].prev_words_in_flight_check_flush();
            bool output_finished = output_queues[i].is_remote_finished();
            if (output_finished) {
                if ((i == 1) && (inner_stop_mux_d_bypass != 0)) {
                    input_queues[1].remote_x = inner_stop_mux_d_bypass & 0xFF;
                    input_queues[1].remote_y = (inner_stop_mux_d_bypass >> 8) & 0xFF;
                    input_queues[1].remote_ready_status_addr = (inner_stop_mux_d_bypass >> 16) & 0xFF;
                }
                input_queues[i].send_remote_finished_notification();
            }
            all_outputs_finished &= output_finished;
        }
        uint32_t launch_msg_rd_ptr = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
        tt_l1_ptr launch_msg_t * const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_msg_rd_ptr]);
        if (launch_msg->kernel_config.exit_erisc_kernel) {
            return;
        }
        run_routing();
    }

    if (!timeout) {
        write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX, 0xff000002);
        for (uint32_t i = 0; i < tunnel_lanes; i++) {
            if (!output_queues[i].output_barrier(timeout_cycles)) {
                timeout = true;
                break;
            }
        }
    }

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    if (!timeout) {
        write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX, 0xff000003);
    }

    set_64b_result(test_results, data_words_sent, PQ_TEST_WORD_CNT_INDEX);
    set_64b_result(test_results, cycles_elapsed, PQ_TEST_CYCLES_INDEX);
    set_64b_result(test_results, iter, PQ_TEST_ITER_INDEX);

    if (timeout) {
        write_buffer_to_l1(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
    } else {
        write_buffer_to_l1(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
        write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX, 0xff00005);
    }
}
