// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>


namespace tt::tt_metal {
inline namespace v0 {

bool executeLightMetalBinary(const std::string& filename);

}  // namespace v0
}  // namespace tt::tt_metal
