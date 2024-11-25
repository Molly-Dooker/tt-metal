set(ARCHS
    grayskull
    wormhole
    blackhole
)
set(PROCS
    brisc
    ncrisc
    trisc0
    trisc1
    trisc2
    ierisc
    slave_ierisc
)
set(TYPES firmware)

include(FetchContent)
FetchContent_Declare(
    sfpi
    URL
        https://github.com/tenstorrent/sfpi/releases/download/v5.0.0/sfpi-release.tgz
    URL_HASH MD5=10cacabe92846076cb4d7596e46d06be
    SOURCE_DIR
    ${PROJECT_SOURCE_DIR}/runtime/sfpi
)
FetchContent_MakeAvailable(sfpi)

foreach(ARCH IN LISTS ARCHS)
    set(FW_INCLUDES "${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${ARCH}")
    set(FW_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/runtime/hw/firmware/${ARCH}")
    foreach(PROC IN LISTS PROCS)
        foreach(TYPE IN LISTS TYPES)
            set(FW_OUTPUT_FILE "${FW_OUTPUT_DIR}/${TYPE}_${PROC}.ld")
            #string(TOUPPER ${PROC} PROC_DEFINE)
            #string(TOUPPER ${TYPE} TYPE_DEFINE)

            # custom command to preprocess/generate the output file
            add_custom_command(
                OUTPUT
                    ${FW_OUTPUT_FILE}
                COMMAND
                    ${CMAKE_COMMAND} -E make_directory ${FW_OUTPUT_DIR}
                COMMAND
                    ${CMAKE_CXX_COMPILER} -DLD_TARGET=${PROC_DEFINE} -DLD_TYPE=${TYPE_DEFINE} -DTARGET_${PROC_DEFINE}
                    -DTYPE_${TYPE_DEFINE} -DCOMPILE_FOR_${PROC_DEFINE} -I${FW_INCLUDES} -E -P -x c -o ${FW_OUTPUT_FILE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
                DEPENDS
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/main.ld
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/memory.ld
                    ${CMAKE_CURRENT_SOURCE_DIR}/toolchain/sections.ld
                    ${DEV_MEM_MAP}
                COMMENT "Preprocessing toolchain/${PROC}.ld"
                VERBATIM
            )

            # add output file to the custom target
            list(APPEND PREPROCESSED_LD_FILES ${FW_OUTPUT_FILE})
        endforeach()
    endforeach()
endforeach()

# Set project name
project(FirmwareProject)

# Define the toolchain file for cross-compiling (make sure to replace with your specific toolchain file)
# Example: toolchain-arm-none-eabi.cmake (This is specific for ARM Cortex-M)
#set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/toolchain-arm-none-eabi.cmake")

# Specify the MCU (example for STM32, change this for your specific MCU)
#set(MCU "STM32F407VG")

# Define the compiler command
set(GPP_CMD ${PROJECT_SOURCE_DIR}/runtime/sfpi/compiler/bin/riscv32-unknown-elf-g++)

set(GPP_DEFINES -DTENSIX_FIRMWARE)

#set(CMAKE_C_FLAGS "-mcpu=cortex-m4 -mthumb -std=c99 -Wall -Wextra")
set(GPP_FLAGS_grayskull
    -mgrayskull
    -march=rv32iy
    -mtune=rvtt-b1
    -mabi=ilp32
)
set(GPP_FLAGS_wormhole
    -mwormhole
    -march=rv32imw
    -mtune=rvtt-b1
    -mabi=ilp32
)
set(GPP_FLAGS_blackhole
    -mblackhole
    -march=rv32iml
    -mtune=rvtt-b1
    -mabi=ilp32
)
# Define common flags for all architectures
set(GPP_FLAGS_common
    -std=c++17
    -flto
    -ffast-math
    -fno-use-cxa-atexit
    -fno-exceptions
    -Wall
    -Werror
    -Wno-unknown-pragmas
    -Wno-error=multistatement-macros
    -Wno-error=parentheses
    -Wno-error=unused-but-set-variable
    -Wno-unused-variable
    -Wno-unused-function
    -O3
)
set(GPP_FLAGS_brisc
    -Os
    -fno-tree-loop-distribute-patterns
)

set(GPP_FLAGS_ncrisc
    -Os
    -fno-tree-loop-distribute-patterns
)

set(GPP_FLAGS_trisc0 -O3)
set(GPP_FLAGS_trisc1 -O3)
set(GPP_FLAGS_trisc2 -O3)
set(GPP_FLAGS_erisc
    -Os
    -fno-delete-null-pointer-checks
)
set(GPP_FLAGS_ierisc
    -Os
    -fno-delete-null-pointer-checks
)
set(GPP_FLAGS_slave_ierisc
    -Os
    -fno-delete-null-pointer-checks
)

# Set GPP_FLAGS based on ARCH
set(GPP_FLAGS
    ${GPP_FLAGS_${ARCH}}
    ${GPP_FLAGS_${PROCS}}
    ${GPP_FLAGS_common}
)
# Dump object files to this directory
set(FW_LIB_DIR ${PROJECT_SOURCE_DIR}/runtime/hw/lib/${ARCH})

# Set the build type to Release or Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set the output directory for the built firmware (bin/ or build/)
set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

# Specify the source directory and include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# Define the source files for the firmware
file(
    GLOB SOURCES
    ${CMAKE_SOURCE_DIR}/src/*.c
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)

# Create the executable target (firmware binary)
add_executable(${PROJECT_NAME}.elf ${SOURCES})

# Linker script (make sure to specify your MCU-specific linker script)
set(LINKER_SCRIPT "${CMAKE_SOURCE_DIR}/ldscripts/${MCU}.ld")

# Link the firmware with the appropriate startup code and linker script
target_link_libraries(
    ${PROJECT_NAME}.elf
    -T${LINKER_SCRIPT}
    -L${CMAKE_SOURCE_DIR}/libs
)

# Set the output directory for the firmware binary
set_target_properties(
    ${PROJECT_NAME}.elf
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}
)

# Convert ELF to BIN or HEX (optional, can be used for flashing)
add_custom_command(
    TARGET ${PROJECT_NAME}.elf
    POST_BUILD
    COMMAND
        ${CMAKE_OBJCOPY} -O binary ${PROJECT_NAME}.elf ${PROJECT_NAME}.bin
    COMMAND
        ${CMAKE_OBJCOPY} -O ihex ${PROJECT_NAME}.elf ${PROJECT_NAME}.hex
    COMMENT "Converting ELF to BIN and HEX"
)

# Specify post-build commands to flash the MCU (optional, example for OpenOCD)
add_custom_target(
    flash
    COMMAND
        openocd -f interface/stlink-v2.cfg -f target/stm32f4x.cfg -c "program ${PROJECT_NAME}.bin verify reset exit"
    DEPENDS
        ${PROJECT_NAME}.bin
)
