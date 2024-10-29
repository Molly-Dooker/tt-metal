// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bitset>

struct tt_xyz_coord {
    constexpr tt_xyz_coord() : x{}, y{}, z{} {}
    constexpr tt_xyz_coord(std::size_t x, std::size_t y, std::size_t z) : x(x), y(y), z(z) {}
    constexpr tt_xyz_coord(std::size_t x, std::size_t y) : x(x), y(y), z(0) {}

    std::size_t x;
    std::size_t y;
    std::size_t z;
};

constexpr inline bool operator==(const tt_xyz_coord &a, const tt_xyz_coord& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

using LogicalDeviceCoord = tt_xyz_coord;

struct LogicalDeviceRange {
    LogicalDeviceCoord start_coord;
    LogicalDeviceCoord end_coord;

    LogicalDeviceRange(const LogicalDeviceCoord& point) {
        this->start_coord = point;
        this->end_coord = point;
    }

    LogicalDeviceRange(const LogicalDeviceCoord& start_coord, const LogicalDeviceCoord& end_coord) {
        this->start_coord = start_coord;
        this->end_coord = end_coord;
    }
};

constexpr bool operator==(const LogicalDeviceRange &a, const LogicalDeviceRange &b) {
    return a.start_coord == b.start_coord && a.end_coord == b.end_coord;
}

namespace std {

template <>
struct hash<tt_xyz_coord> {
    std::size_t operator()(tt_xyz_coord const &o) const {
        return std::hash<std::size_t>()(o.x) ^ (std::hash<std::size_t>()(o.y) << 1) ^ (std::hash<std::size_t>()(o.y) << 2);
    }
};

template <>
struct hash<LogicalDeviceRange> {
    std::size_t operator()(const LogicalDeviceRange &device_range) const {
        std::size_t seed = 0;
        seed = std::hash<LogicalDeviceCoord>{}(device_range.start_coord) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed = std::hash<LogicalDeviceCoord>{}(device_range.end_coord) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

}  // namespace std
