// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_descriptor.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/llrt/get_platform_architecture.hpp"
#include <list>

namespace tt::tt_metal {

enum DispatchWorkerType : uint32_t {
    PREFETCH = 0,
    PREFETCH_HD = 1,
    PREFETCH_H = 2,
    PREFETCH_D = 3,
    DISPATCH = 4,
    DISPATCH_HD = 5,
    DISPATCH_H = 6,
    DISPATCH_D = 7,
    DISPATCH_S = 8,
    MUX = 9,
    MUX_D = 10,
    DEMUX = 11,
    DEMUX_D = 12,
    US_TUNNELER_LOCAL = 13,
    US_TUNNELER_REMOTE = 14,
    PACKET_ROUTER_MUX = 15,
    PACKET_ROUTER_DEMUX = 16,
    COUNT = 17
};

enum DispatchCoreType : uint32_t {
    WORKER = 0,
    ETH = 1,
};

enum class DispatchCoreAxis {
    ROW,
    COL,
};

class DispatchCoreConfig {
private:
    DispatchCoreType type_;
    DispatchCoreAxis axis_;

    static DispatchCoreAxis get_default_axis() {
        return (tt::tt_metal::get_platform_architecture() == tt::ARCH::BLACKHOLE) ? DispatchCoreAxis::COL
                                                                                  : DispatchCoreAxis::ROW;
    }

public:
    DispatchCoreConfig() : type_(DispatchCoreType::WORKER), axis_(get_default_axis()) {}

    DispatchCoreConfig(DispatchCoreType type) : type_(type), axis_(get_default_axis()) {}

    DispatchCoreConfig(DispatchCoreType type, DispatchCoreAxis axis) : type_(type), axis_(axis) {}

    CoreType get_core_type() const {
        switch (type_) {
            case DispatchCoreType::WORKER: return CoreType::WORKER;
            case DispatchCoreType::ETH: return CoreType::ETH;
            default: TT_THROW("invalid dispatch core type");
        }
    }

    DispatchCoreType get_dispatch_core_type() const { return type_; }

    void set_dispatch_core_type(DispatchCoreType new_type) { type_ = new_type; }

    DispatchCoreAxis get_dispatch_core_axis() const { return axis_; }

    void set_dispatch_core_axis(DispatchCoreAxis new_axis) { axis_ = new_axis; }

    bool operator==(const DispatchCoreConfig& other) const { return (type_ == other.type_) && (axis_ == other.axis_); }
};

}  // namespace tt::tt_metal
