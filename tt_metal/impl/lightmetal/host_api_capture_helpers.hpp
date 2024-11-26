#pragma once

#include <cstdint>
#include "lightmetal_capture_context.hpp"
#include "command_generated.h"
#include "tt_metal/common/logger.hpp"

// KCM - Temporary hack for bringup.
#define ENABLE_TRACING 1

#ifdef ENABLE_TRACING
    #define TRACE_FUNCTION_CALL(capture_func, ...) \
        do { \
            if (LightMetalCaptureContext::getInstance().isTracing()) { \
                capture_func(__VA_ARGS__); \
            } \
        } while (0)
#else
    #define TRACE_FUNCTION_CALL(capture_func, ...) do { } while (0)
#endif

inline void captureReplayTrace(Device *device, uint8_t cq_id, uint32_t tid, bool blocking) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    // FIXME - Handle device_id.
    log_info(tt::LogMetalTrace, "captureReplayTrace: cq_id: {}, tid: {}, blocking: {}", cq_id, tid, blocking);
    if (!ctx.isTracing()) return;

    auto& builder = ctx.getBuilder();
    // If complex types, convert to flatbuffer format here.
    auto command_variant = tt::target::CreateReplayTrace(builder, cq_id, tid, blocking);
    auto command = tt::target::CreateCommand(
        builder,
        tt::target::CommandUnion::CommandUnion_ReplayTrace,
        command_variant.Union()
    );
    ctx.getCmdsVector().push_back(command);
}

inline void captureEnqueueTrace(CommandQueue& cq, uint32_t trace_id, bool blocking) {
    auto& ctx = LightMetalCaptureContext::getInstance();

    uint8_t cq_id = cq.id();
    // FIXME - Handle device_id.

    log_info(tt::LogMetalTrace, "captureEnqueueTrace: cq_id: {}, trace_id: {}, blocking: {}", cq_id, trace_id, blocking);
    if (!ctx.isTracing()) return;

    auto& builder = ctx.getBuilder();
    // If complex types, convert to flatbuffer format here.
    auto command_variant = tt::target::CreateEnqueueTrace(builder, cq_id, trace_id, blocking);
    auto command = tt::target::CreateCommand(
        builder,
        tt::target::CommandUnion::CommandUnion_EnqueueTrace,
        command_variant.Union()
    );
    ctx.getCmdsVector().push_back(command);
}

inline void captureLightMetalLoadTraceId(Device *device, const uint32_t tid, const uint8_t cq_id) {
    auto& ctx = LightMetalCaptureContext::getInstance();
    // FIXME - Handle device_id.
    log_info(tt::LogMetalTrace, "{}: cq_id: {}, tid: {}", __FUNCTION__, cq_id, tid);
    if (!ctx.isTracing()) return;

    auto& builder = ctx.getBuilder();
    auto command_variant = tt::target::CreateLightMetalLoadTraceId(builder, tid, cq_id);
    auto command = tt::target::CreateCommand(
        builder,
        tt::target::CommandUnion::CommandUnion_LightMetalLoadTraceId,
        command_variant.Union()
    );
    ctx.getCmdsVector().push_back(command);
}
