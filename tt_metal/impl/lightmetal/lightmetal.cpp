// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "lightmetal.hpp"
#include "binary_generated.h"
#include "tt_metal/impl/trace/trace_buffer.hpp"
#include "tt_metal/common/logger.hpp"

namespace tt::tt_metal {
inline namespace v0 {

//////////////////////////////////////
// Serialization Functions          //
//////////////////////////////////////

TraceDescriptorByTraceIdOffset toFlatBuffer(flatbuffers::FlatBufferBuilder& builder, const detail::TraceDescriptor& traceDesc, const uint32_t trace_id) {
    auto trace_data_offset = builder.CreateVector(traceDesc.data);

    return tt::target::lightmetal::CreateTraceDescriptorByTraceId(
        builder,
        trace_id,
        tt::target::lightmetal::CreateTraceDescriptor(
            builder,
            trace_data_offset,
            traceDesc.num_completion_worker_cores,
            traceDesc.num_traced_programs_needing_go_signal_multicast,
            traceDesc.num_traced_programs_needing_go_signal_unicast
        )
    );
}


}  // namespace v0
}  // namespace tt::tt_metal
