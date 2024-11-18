// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define is_power_of_2(x) (((x) > 0) && (((x) & ((x) - 1)) == 0))

#define is_16b_aligned(x) (x % 16 == 0)

constexpr uint32_t PACKET_WORD_SIZE_BYTES = 16;

constexpr uint32_t MAX_SWITCH_FAN_IN = 4;
constexpr uint32_t MAX_SWITCH_FAN_OUT = 4;
constexpr uint32_t MAX_TUNNEL_LANES = 10;

constexpr uint32_t MAX_SRC_ENDPOINTS = 32;
constexpr uint32_t MAX_DEST_ENDPOINTS = 32;
constexpr uint32_t PACKET_QUEUE_MAX_ID = std::max(MAX_SRC_ENDPOINTS, MAX_DEST_ENDPOINTS);

constexpr uint32_t INPUT_QUEUE_START_ID = 0;
constexpr uint32_t OUTPUT_QUEUE_START_ID = MAX_SWITCH_FAN_IN;

constexpr uint32_t PACKET_QUEUE_REMOTE_READY_FLAG = 0xA;
constexpr uint32_t PACKET_QUEUE_REMOTE_FINISHED_FLAG = 0xB;

constexpr uint32_t PACKET_QUEUE_STAUS_MASK = 0xabc00000;
constexpr uint32_t PACKET_QUEUE_TEST_STARTED = PACKET_QUEUE_STAUS_MASK | 0x0;
constexpr uint32_t PACKET_QUEUE_TEST_PASS = PACKET_QUEUE_STAUS_MASK | 0x1;
constexpr uint32_t PACKET_QUEUE_TEST_TIMEOUT = PACKET_QUEUE_STAUS_MASK | 0x2;
constexpr uint32_t PACKET_QUEUE_TEST_DATA_MISMATCH = PACKET_QUEUE_STAUS_MASK | 0x3;

// Scratch buffer. All addresses are made to be 16B aligned.
// Extra space is provided in the slots for remote shadow values. The offsets are below.
// TODO: make a struct/union for the buffer slots
constexpr uint32_t PACKET_QUEUE_SCRATCH_BUFFER_SLOT_BYTES = 64;
constexpr uint32_t PACKET_QUEUE_SCRATCH_BUFFER_SHADOW_WTPR_OFFSET = PACKET_WORD_SIZE_BYTES;
constexpr uint32_t PACKET_QUEUE_SCRATCH_BUFFER_SHADOW_RTPR_SENT_OFFSET = PACKET_WORD_SIZE_BYTES * 2;
constexpr uint32_t PACKET_QUEUE_SCRATCH_BUFFER_SHADOW_RTPR_CLEARED_OFFSET = PACKET_WORD_SIZE_BYTES * 3;

// indexes of return values in test results buffer
constexpr uint32_t PQ_TEST_STATUS_INDEX = 0;
constexpr uint32_t PQ_TEST_WORD_CNT_INDEX = 2;
constexpr uint32_t PQ_TEST_CYCLES_INDEX = 4;
constexpr uint32_t PQ_TEST_ITER_INDEX = 6;
constexpr uint32_t PQ_TEST_MISC_INDEX = 16;


constexpr uint32_t NUM_WR_CMD_BUFS = 4;
constexpr uint32_t DEFAULT_MAX_NOC_SEND_WORDS =
    (NUM_WR_CMD_BUFS - 1) * (NOC_MAX_BURST_WORDS * NOC_WORD_BYTES) / PACKET_WORD_SIZE_BYTES;
constexpr uint32_t DEFAULT_MAX_ETH_SEND_WORDS = 2 * 1024;

enum DispatchPacketFlag : uint32_t {
    PACKET_CMD_START = (0x1 << 1),
    PACKET_CMD_END = (0x1 << 2),
    PACKET_MULTI_CMD = (0x1 << 3),
    PACKET_TEST_LAST = (0x1 << 15),  // test only
};

enum DispatchRemoteNetworkType : uint8_t {
    NOC0 = 0,
    NOC1_RESERVED = 1,
    ETH = 2,
    NONE = 3
};

struct dispatch_packet_header_t {
    uint32_t packet_size_bytes;
    uint16_t packet_src;
    uint16_t packet_dest;
    uint16_t packet_flags;
    uint16_t num_cmds;
    uint32_t tag;
};

static_assert(sizeof(dispatch_packet_header_t) == PACKET_WORD_SIZE_BYTES);

static_assert(MAX_DEST_ENDPOINTS <= 32,
              "MAX_DEST_ENDPOINTS must be <= 32 for the packing functions below to work");

static_assert(MAX_SWITCH_FAN_OUT <= 4,
              "MAX_SWITCH_FAN_OUT must be <= 4 for the packing functions below to work");

uint32_t packet_switch_4B_pack(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3) {
    return (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
}

uint64_t packet_switch_dest_pack(uint32_t* dest_output_map_array, uint32_t num_dests) {
    uint64_t result = 0;
    for (uint32_t i = 0; i < num_dests; i++) {
        result |= ((uint64_t)(dest_output_map_array[i])) << (2*i);
    }
    return result;
}

// Ethernet ack and staging addresses
#if defined(COMPILE_FOR_ERISC)
constexpr uint32_t PACKET_QUEUE_ETH_STAGE_ADDR = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
constexpr uint32_t PACKET_QUEUE_ACK_BASE_ADDR = PACKET_QUEUE_ETH_STAGE_ADDR + PACKET_WORD_SIZE_BYTES;

constexpr uint32_t PACKET_QUEUE_ACK_LOW_DEVICE_ADDR = PACKET_QUEUE_ACK_BASE_ADDR;
constexpr uint32_t PACKET_QUEUE_ACK_HIGH_DEVICE_ADDR = PACKET_QUEUE_ACK_LOW_DEVICE_ADDR + PACKET_WORD_SIZE_BYTES;

// Checking last item - first item is within the l1 range
static_assert(PACKET_QUEUE_ACK_HIGH_DEVICE_ADDR - PACKET_QUEUE_ETH_STAGE_ADDR <= eth_l1_mem::address_map::ERISC_L1_UNRESERVED_SIZE, "Packet queue ethernet buffer in ERISC_L1_UNRESERVED_BASE has overflowed");

#endif
