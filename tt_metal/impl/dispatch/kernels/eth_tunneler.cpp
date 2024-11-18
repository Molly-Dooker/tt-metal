// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include "ethernet/dataflow_api.h"
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

// Device id of the chip running this eth tunneler
constexpr uint32_t device_id = get_compile_time_arg_val(16);
// Device id of the other chip running this eth tunneler
constexpr uint32_t remote_device_id = get_compile_time_arg_val(17);

// Initialize the handshake address. The address can be the same for the input/output
// queues as in the kernel main loop, only one queue will use that space at a time.
// Even though both eth tunnelers will be communicating with each other, we establish that the
// "sender" will be the one with the lower device id and the "receiver" is the one with the
// higher device id.
// The lower device id will use ERISC_UNRESERVED_BASE as their ack address
// The higher device id will use ERISC_UNRESERVED_BASE + 16.
//
// If the same address is used then a deadlock could occur. Both ack values will be 1
// and they will be waiting for each other
constexpr uint32_t local_ack_addr =
    device_id < remote_device_id ? PACKET_QUEUE_ACK_LOW_DEVICE_ADDR:
    PACKET_QUEUE_ACK_HIGH_DEVICE_ADDR;

constexpr uint32_t remote_ack_addr =
    device_id < remote_device_id ? PACKET_QUEUE_ACK_HIGH_DEVICE_ADDR :
   PACKET_QUEUE_ACK_LOW_DEVICE_ADDR;

// Eth tunneler inputs
constexpr uint32_t eth_tunneler_in_local_wptr_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(18),
    get_compile_time_arg_val(19)};

constexpr uint32_t eth_tunneler_in_remote_rptr_sent_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(20),
    get_compile_time_arg_val(21)};

constexpr uint32_t eth_tunneler_in_remote_rptr_cleared_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(22),
    get_compile_time_arg_val(23)};

// Eth tunneler outputs
constexpr uint32_t eth_tunneler_out_local_rptr_sent_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(24),
    get_compile_time_arg_val(25)};

constexpr uint32_t eth_tunneler_out_local_rptr_cleared_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(26),
    get_compile_time_arg_val(27)};

constexpr uint32_t eth_tunneler_out_remote_wptr_addr[NUM_TUNNEL_QUEUES] = {
    get_compile_time_arg_val(28),
    get_compile_time_arg_val(29)};


void kernel_main() {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    write_buffer_to_l1(test_results, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX, 0xff000000);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX + 1, 0xbb000000);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX + 2, 0xAABBCCDD);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX + 3, 0xDDCCBBAA);
    write_buffer_to_l1(test_results, PQ_TEST_MISC_INDEX + 4, endpoint_id_start_index);

    // Ack for data leaving this core
    auto local_ack = reinterpret_cast<volatile eth_channel_sync_t*>(local_ack_addr);
    // Ack for data coming into this core
    auto remote_ack = reinterpret_cast<volatile eth_channel_sync_t*>(remote_ack_addr);

    local_ack->bytes_sent = 0;
    remote_ack->bytes_sent = 0;

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

        input_queues[i].staging_area_ = reinterpret_cast<volatile uint32_t*>(PACKET_QUEUE_ETH_STAGE_ADDR);
        input_queues[i].local_ack_ = local_ack;
        input_queues[i].remote_ack_ = remote_ack;

        output_queues[i].staging_area_ = reinterpret_cast<volatile uint32_t*>(PACKET_QUEUE_ETH_STAGE_ADDR);
        output_queues[i].local_ack_ = local_ack;
        output_queues[i].remote_ack_ = remote_ack;
    }

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
    while (!all_outputs_finished && !timeout) {
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
            // Acknowledge any data received
            if (local_ack->bytes_sent != 0) {
                local_ack->bytes_sent = 0;
                while (eth_txq_is_busy()) {}
                internal_::eth_send_packet(
                    0,
                    (uint32_t)(local_ack) >> 4,
                    (uint32_t)(local_ack) >> 4,
                    1
                );
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
