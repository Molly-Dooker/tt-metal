// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "global_semaphore.hpp"

#include "tt_metal/impl/buffers/global_semaphore.hpp"
#include "ttnn/cpp/ttnn/global_semaphore.hpp"
#include "pybind11/pybind11.h"

namespace ttnn::global_semaphore {

void py_module_types(py::module& module) {
    py::class_<tt::tt_metal::GlobalSemaphore, std::shared_ptr<tt::tt_metal::GlobalSemaphore>>(
        module, "global_sempahore");
}

void py_module(py::module& module) {
    // Single Device APIs
    module.def(
        "create_global_semaphore",
        &create_global_semaphore,
        py::arg("device"),
        py::arg("cores"),
        py::arg("initial_value"),
        py::arg("buffer_type") = tt::tt_metal::BufferType::L1,
        R"doc(
            Create an GlobalCircularBuffer Object on a single device.

            Args:
                device (Device): The device on which to create the global semaphore.
                cores (CoreRangeSet): The cores on which the global semaphore will be used for synchronization.
                initial_value (int): The initial value of the global semaphore.
                buffer_type (BufferType): The type of buffer to use for the global semaphore.
            )doc");

    module.def(
        "get_global_semaphore_address",
        &get_global_semaphore_address,
        py::arg("global_semaphore"),
        R"doc(
            Get the address of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
            )doc");

    module.def(
        "reset_global_semaphore_value",
        &reset_global_semaphore_value,
        py::arg("global_semaphore"),
        R"doc(
            Reset the value of the global semaphore.

            Args:
                global_semaphore (GlobalSemaphore): The global semaphore object.
            )doc");
}

}  // namespace ttnn::global_semaphore
