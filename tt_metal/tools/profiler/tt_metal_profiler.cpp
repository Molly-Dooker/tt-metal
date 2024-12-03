// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <thread>
#include <cmath>

#include "llrt/hal.hpp"
#include "tt_metal/host_api.hpp"
#include "impl/debug/dprint_server.hpp"

#include "tools/profiler/profiler.hpp"
#include "hostdevcommon/profiler_common.h"

#include "tt_metal/detail/tt_metal.hpp"

#include "tt_metal/third_party/tracy/public/tracy/TracyTTDevice.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/impl/device/device_pool.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

namespace tt {

namespace tt_metal {
inline namespace v0 {

void DumpDeviceProfileResults(Device* device, const Program& program) {
#if defined(TRACY_ENABLE)
    std::vector<CoreCoord> worker_cores_in_program;
    std::vector<CoreCoord> eth_cores_in_program;

    std::vector<std::vector<CoreCoord>>logical_cores = program.logical_cores();
    for (uint32_t index = 0; index < hal.get_programmable_core_type_count(); index++) {
        if (hal.get_core_type(index) == CoreType::WORKER) {
            worker_cores_in_program = device->worker_cores_from_logical_cores(logical_cores[index]);
        }
        if (hal.get_core_type(index) == CoreType::ETH) {
            eth_cores_in_program = device->ethernet_cores_from_logical_cores(logical_cores[index]);
        }
    }

    std::vector<CoreCoord> cores_in_program;
    cores_in_program.reserve(worker_cores_in_program.size() + eth_cores_in_program.size());
    std::copy(worker_cores_in_program.begin(), worker_cores_in_program.end(), std::back_inserter(cores_in_program));
    std::copy(eth_cores_in_program.begin(), eth_cores_in_program.end(), std::back_inserter(cores_in_program));

    detail::DumpDeviceProfileResults(device, cores_in_program);
#endif
}

}  // namespace v0

namespace detail {

std::map <uint32_t, DeviceProfiler> tt_metal_device_profiler_map;

std::unordered_map <chip_id_t, std::vector <std::pair<uint64_t,uint64_t>>> deviceHostTimePair;
std::unordered_map <chip_id_t, uint64_t> smallestHostime;

std::unordered_map <chip_id_t, std::unordered_map <chip_id_t, std::vector <std::pair<uint64_t,uint64_t>>>> deviceDeviceTimePair;
std::mutex device_mutex;

constexpr CoreCoord SYNC_CORE = {0,0};

std::set<tracy::TTDeviceEvent> ret = {};
std::set<tracy::TTDeviceEvent>& getSyncDeviceData(int device_id){
    if (tt_metal_device_profiler_map.find(device_id) != tt_metal_device_profiler_map.end())
    {
        return tt_metal_device_profiler_map.at(device_id).device_sync_events;
    }
    return ret;
}
void setControlBuffer(chip_id_t device_id, std::vector<uint32_t>& control_buffer)
{
#if defined(TRACY_ENABLE)
    const metal_SocDescriptor& soc_d = tt::Cluster::instance().get_soc_desc(device_id);

    control_buffer[kernel_profiler::CORE_COUNT_PER_DRAM] = soc_d.profiler_ceiled_core_count_perf_dram_bank;

    auto ethCores = soc_d.get_physical_ethernet_cores() ;
    for (auto &core : soc_d.physical_routing_to_profiler_flat_id)
    {
        profiler_msg_t *profiler_msg;
        // TODO: clean this up when HAL is more complete (one lookup w/ type)
        if (std::find(ethCores.begin(), ethCores.end(), core.first) == ethCores.end())
        {
            //Tensix
            profiler_msg = hal.get_dev_addr<profiler_msg_t *>(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER);
        }
        else
        {
            //ETH
            profiler_msg = hal.get_dev_addr<profiler_msg_t *>(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::PROFILER);
        }

        control_buffer[kernel_profiler::FLAT_ID] = core.second;
        tt::llrt::write_hex_vec_to_core(
            device_id,
            core.first,
            control_buffer,
            reinterpret_cast<uint64_t>(profiler_msg->control_vector));
    }
#endif
}

void syncDeviceHost(Device *device, CoreCoord logical_core, std::shared_ptr<tt_metal::Program> &sync_program, bool doHeader)
{
    if (!tt::llrt::OptionsG.get_profiler_sync_enabled()) return;
    auto device_id = device->id();
    if (device_id > 0) return;
    ZoneScopedC(tracy::Color::Tomato3);
    auto core = device->worker_core_from_logical_core(logical_core);

    deviceHostTimePair.emplace(device_id, (std::vector <std::pair<uint64_t,uint64_t>>){});
    smallestHostime.emplace(device_id, 0);

    constexpr uint16_t sampleCount = 249;
    if (sync_program == nullptr) {
        sync_program = std::make_shared<tt_metal::Program>();

        std::map<string, string> kernel_defines = {
            {"SAMPLE_COUNT", std::to_string(sampleCount)},
        };

        tt_metal::KernelHandle brisc_kernel = tt_metal::CreateKernel(
            *sync_program, "tt_metal/tools/profiler/sync/sync_kernel.cpp",
            logical_core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .defines = kernel_defines}
            );
    }

    EnqueueProgram(device->command_queue(), *sync_program, false);

    std::filesystem::path output_dir = std::filesystem::path(get_profiler_logs_dir());
    std::filesystem::path log_path = output_dir / "sync_device_info.csv";
    std::ofstream log_file;

    int64_t writeSum = 0;

    constexpr int millisecond_wait = 4;

    const double tracyToSecRatio = TracyGetTimerMul();
    const int64_t tracyBaseTime = TracyGetBaseTime();
    const int64_t hostStartTime = TracyGetCpuTime();
    std::vector<int64_t> writeTimes(sampleCount);

    profiler_msg_t *profiler_msg = device->get_dev_addr<profiler_msg_t *>(core, HalL1MemAddrType::PROFILER);
    uint64_t control_addr = reinterpret_cast<uint64_t>(&profiler_msg->control_vector[kernel_profiler::FW_RESET_L]);
    for (int i = 0; i < sampleCount; i++)
    {
        ZoneScopedC(tracy::Color::Tomato2);
        std::this_thread::sleep_for(std::chrono::milliseconds(millisecond_wait));
        int64_t writeStart = TracyGetCpuTime();
        uint32_t sinceStart = writeStart - hostStartTime;

        tt::Cluster::instance().write_reg(&sinceStart, tt_cxy_pair(device_id, core), control_addr);
        writeTimes[i] = (TracyGetCpuTime() - writeStart);
    }

    Finish(device->command_queue());

    log_info ("SYNC PROGRAM FINISH IS DONE ON {}",device_id);
    if ((smallestHostime[device_id] == 0) || (smallestHostime[device_id] > hostStartTime))
    {
        smallestHostime[device_id] = hostStartTime;
    }

    for (auto writeTime : writeTimes)
    {
        writeSum += writeTime;
    }
    double writeOverhead = (double)writeSum / sampleCount;

    constexpr uint32_t briscIndex = 0;
    uint64_t addr = reinterpret_cast<uint64_t>(&profiler_msg->buffer[briscIndex][kernel_profiler::CUSTOM_MARKERS]);

    std::vector<std::uint32_t> sync_times = tt::llrt::read_hex_vec_from_core(
            device_id,
            core,
            addr,
            (sampleCount + 1) * 2 * sizeof(uint32_t));

    uint32_t preDeviceTime = 0;
    uint32_t preHostTime = 0;
    bool firstSample = true;

    uint64_t deviceStartTime = (uint64_t(sync_times[0] & 0xFFF) << 32) | sync_times[1];
    uint32_t deviceStartTime_H = sync_times[0] & 0xFFF;
    uint32_t deviceStartTime_L = sync_times[1];
    preDeviceTime = deviceStartTime_L;

    uint32_t hostStartTime_H = 0;

    uint64_t preDeviceTimeLarge = 0;
    uint64_t preHostTimeLarge = 0;
    uint64_t firstDeviceTimeLarge = 0;
    uint64_t firstHostTimeLarge = 0;

    for (int i = 2; i < 2 * (sampleCount + 1); i += 2)
    {

        uint32_t deviceTime = sync_times[i];
        if (deviceTime < preDeviceTime) deviceStartTime_H ++;
        preDeviceTime = deviceTime;
        uint64_t deviceTimeLarge = (uint64_t(deviceStartTime_H) << 32) | deviceTime;

        uint32_t hostTime = sync_times[i + 1] + writeTimes[i/2 - 1];
        if (hostTime < preHostTime) hostStartTime_H ++;
        preHostTime = hostTime;
        uint64_t hostTimeLarge = hostStartTime - smallestHostime[device_id] + ((uint64_t(hostStartTime_H) << 32) | hostTime);

        deviceHostTimePair[device_id].push_back(std::pair<uint64_t,uint64_t> {deviceTimeLarge,hostTimeLarge});

        if (firstSample)
        {
            firstDeviceTimeLarge = deviceTimeLarge;
            firstHostTimeLarge = hostTimeLarge;
            firstSample = false;
        }

        preDeviceTimeLarge = deviceTimeLarge;
        preHostTimeLarge = hostTimeLarge;
    }

    double hostSum = 0;
    double deviceSum = 0;
    double hostSquaredSum = 0;
    double hostDeviceProductSum = 0;

    for (auto& deviceHostTime : deviceHostTimePair[device_id])
    {
        double deviceTime = deviceHostTime.first;
        double hostTime = deviceHostTime.second;

        deviceSum += deviceTime;
        hostSum += hostTime;
        hostSquaredSum += (hostTime * hostTime);
        hostDeviceProductSum += (hostTime * deviceTime);
    }

    uint16_t accumulateSampleCount = deviceHostTimePair[device_id].size();

    double frequencyFit = (hostDeviceProductSum * accumulateSampleCount - hostSum * deviceSum)  / ((hostSquaredSum * accumulateSampleCount - hostSum * hostSum) * tracyToSecRatio);

    double delay = (deviceSum - frequencyFit * hostSum * tracyToSecRatio) / accumulateSampleCount;

    log_file.open(log_path, std::ios_base::app);
    if (doHeader)
    {
        log_file << fmt::format("device id,core_x, core_y,device,host_tracy,host_real,write_overhead,host_start,delay,frequency,tracy_ratio,tracy_base_time") << std::endl;
    }
    int init = deviceHostTimePair[device_id].size() - sampleCount;
    for (int i = init ;i < deviceHostTimePair[device_id].size(); i++)
    {
        log_file << fmt::format(
                "{:5},{:5},{:5},{:20},{:20},{:20.2f},{:20},{:20},{:20.2f},{:20.15f},{:20.15f},{:20}",
                device_id,
                core.x,
                core.y,
                deviceHostTimePair[device_id][i].first,
                deviceHostTimePair[device_id][i].second,
                (double) deviceHostTimePair[device_id][i].second  * tracyToSecRatio,
                writeTimes[i - init],
                smallestHostime[device_id],
                delay,
                frequencyFit,
                tracyToSecRatio,
                tracyBaseTime
                )
            << std::endl;
    }

    log_info("Sync data for device: {}, c:{}, d:{}, f:{}",device_id, smallestHostime[device_id], delay, frequencyFit);

    tt_metal_device_profiler_map.at(device_id).device_core_sync_info.emplace(core, std::make_tuple(smallestHostime[device_id], delay, frequencyFit));
}
void setShift(int device_id, int64_t shift, double scale){
    if (tt_metal_device_profiler_map.find(device_id) != tt_metal_device_profiler_map.end()){
        tt_metal_device_profiler_map.at(device_id).shift = shift;
        tt_metal_device_profiler_map.at(device_id).freqScale = scale;
    }
}

void syncDeviceDevice(chip_id_t device_id_sender, chip_id_t device_id_receiver)
{
    ZoneScopedC(tracy::Color::Tomato4);
    std::string zoneName = fmt::format("sync_device_device_{}->{}",device_id_sender, device_id_receiver);
    ZoneName(zoneName.c_str(), zoneName.size());
    if (!tt::llrt::OptionsG.get_profiler_sync_enabled()) return;

    Device* device_sender = nullptr;
    Device* device_receiver = nullptr;

    if (tt::DevicePool::instance().is_device_active(device_id_receiver)){
        device_receiver = tt::DevicePool::instance().get_active_device(device_id_receiver);
    }

    if (tt::DevicePool::instance().is_device_active(device_id_sender)){
        device_sender = tt::DevicePool::instance().get_active_device(device_id_sender);
    }

    if (device_sender != nullptr and device_receiver != nullptr){
        constexpr std::uint16_t sample_count = 120;
        constexpr std::uint16_t sample_size = 16;
        constexpr std::uint16_t channel_count = 1;

        auto const& active_eth_cores = device_sender->get_active_ethernet_cores(true);
        auto eth_sender_core_iter = active_eth_cores.begin();
        tt_xy_pair eth_receiver_core;
        tt_xy_pair eth_sender_core;

        chip_id_t device_id_receiver_curr = std::numeric_limits<chip_id_t>::max();
        do {
            eth_sender_core = *eth_sender_core_iter;
            std::tie(device_id_receiver_curr, eth_receiver_core) = device_sender->get_connected_ethernet_core(eth_sender_core);
            eth_sender_core_iter++;
        } while ((device_id_receiver != device_id_receiver_curr) and (eth_sender_core_iter != active_eth_cores.end()));

        if (device_id_receiver != device_id_receiver_curr){
            log_warning("No eth connection could be found between device {} and {}", device_id_sender, device_id_receiver);
            return;
        }

        std::cout << "Connection :"  << device_id_sender << ":" << eth_sender_core.x << "," << eth_sender_core.y;
        std::cout << "->" << device_id_receiver << ":" << eth_receiver_core.x << "," << eth_receiver_core.y << std::endl;

        std::vector<uint32_t> const& ct_args = {channel_count};
        std::vector<uint32_t> const& rt_args = {
            eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE,
            static_cast<uint32_t>(sample_count),
            static_cast<uint32_t>(sample_size)
        };

        Program program_sender;
        Program program_receiver;

        auto local_kernel = tt_metal::CreateKernel(
            program_sender,
            "tt_metal/tools/profiler/sync/sync_device_kernel_sender.cpp",
            eth_sender_core,
            tt_metal::EthernetConfig {
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = ct_args});
        tt_metal::SetRuntimeArgs(program_sender, local_kernel, eth_sender_core, rt_args);

        auto remote_kernel = tt_metal::CreateKernel(
            program_receiver,
            "tt_metal/tools/profiler/sync/sync_device_kernel_receiver.cpp",
            eth_receiver_core,
            tt_metal::EthernetConfig {
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = ct_args});
        tt_metal::SetRuntimeArgs(program_receiver, remote_kernel, eth_receiver_core, rt_args);


        try {
            tt::tt_metal::detail::CompileProgram(device_sender, program_sender);
            tt::tt_metal::detail::CompileProgram(device_receiver, program_receiver);
        } catch (std::exception& e) {
            log_error("Failed compile: {}", e.what());
            throw e;
        }

        tt_metal::SetRuntimeArgs(program_sender, local_kernel, eth_sender_core, rt_args);
        tt_metal::SetRuntimeArgs(program_receiver, remote_kernel, eth_receiver_core, rt_args);

        tt_metal::EnqueueProgram(device_sender->command_queue(), program_sender, false);
        tt_metal::EnqueueProgram(device_receiver->command_queue(), program_receiver, false);

        tt_metal::Finish(device_sender->command_queue());
        tt_metal::Finish(device_receiver->command_queue());

        //constexpr uint32_t eriscIndex = 0;
        //profiler_msg_t *profiler_msg = device_sender->get_dev_addr<profiler_msg_t *>(eth_sender_core, HalL1MemAddrType::PROFILER);
        //uint64_t addr = reinterpret_cast<uint64_t>(&profiler_msg->buffer[eriscIndex][kernel_profiler::CUSTOM_MARKERS]);
        //std::vector<std::uint32_t> sync_times = tt::llrt::read_hex_vec_from_core(
                //device_id_sender,
                //eth_sender_core,
                //addr,
                //120 * 2 * sizeof(uint32_t));
        //for (const auto& sync_time: sync_times)
        //{
            //std::cout <<  sync_time << std::endl;
        //}
    }
}


void ProfilerSync()
{
#if defined(TRACY_ENABLE)
    auto ethernet_connections = tt::Cluster::instance().get_ethernet_connections();

    std::set<chip_id_t> visited_devices = {};
    std::vector<std::pair<chip_id_t,chip_id_t>> ping_pairs;
    for (const auto& device : ethernet_connections) {
        for (const auto& connection : device.second) {
            chip_id_t receiver_device = std::get<0>(connection.second);

            if (visited_devices.find(device.first) == visited_devices.end() or\
                    visited_devices.find(receiver_device) == visited_devices.end())
            {
                visited_devices.insert(device.first);
                visited_devices.insert(receiver_device);
                std::pair<chip_id_t,chip_id_t> ping_pair = {device.first, receiver_device};
                ping_pairs.push_back(ping_pair);

                deviceDeviceTimePair.emplace(device.first,(std::unordered_map <chip_id_t, std::vector <std::pair<uint64_t,uint64_t>>>){});
                deviceDeviceTimePair.at(device.first).emplace(receiver_device, (std::vector <std::pair<uint64_t,uint64_t>>){});
            }
        }
    }

    for (const auto& ping_pair: ping_pairs){
        syncDeviceDevice (std::get<0>(ping_pair),std::get<1>(ping_pair));
    }


#endif
}

void ClearProfilerControlBuffer(Device *device){
#if defined(TRACY_ENABLE)
    auto device_id = device->id();
    std::vector<uint32_t> control_buffer(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
    setControlBuffer (device_id, control_buffer);
#endif
}


void InitDeviceProfiler(Device *device){
#if defined(TRACY_ENABLE)
    ZoneScoped;
    auto device_id = device->id();
    CoreCoord logical_grid_size = device->logical_grid_size();
    TracySetCpuTime (TracyGetCpuTime());

    bool doSync = false;
    if (getDeviceProfilerState())
    {
        static std::atomic<bool> firstInit = true;
        bool doHeader = firstInit;

        auto device_id = device->id();

        if (tt_metal_device_profiler_map.find(device_id) == tt_metal_device_profiler_map.end())
        {
            doSync = true;
            if (firstInit.exchange(false))
            {
                tt_metal_device_profiler_map.emplace(device_id, DeviceProfiler(true));
            }
            else
            {
                tt_metal_device_profiler_map.emplace(device_id, DeviceProfiler(false));
            }
        }

        uint32_t dramBankCount = tt::Cluster::instance().get_soc_desc(device_id).get_num_dram_channels();
        uint32_t coreCountPerDram = tt::Cluster::instance().get_soc_desc(device_id).profiler_ceiled_core_count_perf_dram_bank;

        uint32_t pageSize =
            PROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC * PROFILER_RISC_COUNT * coreCountPerDram;


        if (tt_metal_device_profiler_map.at(device_id).output_dram_buffer == nullptr )
        {
            tt::tt_metal::InterleavedBufferConfig dram_config{
                        .device= device,
                        .size = pageSize * dramBankCount,
                        .page_size =  pageSize,
                        .buffer_type = tt::tt_metal::BufferType::DRAM
            };
            tt_metal_device_profiler_map.at(device_id).output_dram_buffer = tt_metal::CreateBuffer(dram_config);
        }

        std::vector<uint32_t> control_buffer(kernel_profiler::PROFILER_L1_CONTROL_VECTOR_SIZE, 0);
        control_buffer[kernel_profiler::DRAM_PROFILER_ADDRESS] = tt_metal_device_profiler_map.at(device_id).output_dram_buffer->address();
        setControlBuffer (device_id, control_buffer);

        std::vector<uint32_t> inputs_DRAM(tt_metal_device_profiler_map.at(device_id).output_dram_buffer->size()/sizeof(uint32_t), 0);
        tt_metal::detail::WriteToBuffer(tt_metal_device_profiler_map.at(device_id).output_dram_buffer, inputs_DRAM);
        if (doSync)
        {
            syncDeviceHost (device, SYNC_CORE, tt_metal_device_profiler_map.at(device_id).sync_program, doHeader);
        }
    }
#endif
}

void DumpDeviceProfileResults(Device *device, bool lastDump) {
#if defined(TRACY_ENABLE)
    ZoneScoped;
    static bool device_sync_done = false;
    std::vector<CoreCoord> workerCores;
    auto device_id = device->id();
    if (device_id == 1 and !device_sync_done)
    {
        //detail::syncDeviceDevice (1,0);
        device_sync_done = true;
    }
    auto device_num_hw_cqs = device->num_hw_cqs();
    CoreType dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device_id);
    for (const CoreCoord& core : tt::get_logical_compute_cores(device_id, device_num_hw_cqs, dispatch_core_type)) {
        const CoreCoord curr_core = device->worker_core_from_logical_core(core);
        workerCores.push_back(curr_core);
    }
    for (const CoreCoord& core : device->get_active_ethernet_cores(true)){
        auto physicalCore = device->physical_core_from_logical_core(core, CoreType::ETH);
        workerCores.push_back(physicalCore);
    }
    device->push_work([device, workerCores, lastDump] () mutable {
        DumpDeviceProfileResults(device, workerCores, lastDump);
        //auto device_id = device->id();
        //if (device_id == 1 and !lastDump)
        //{
            //auto device_0 = tt::DevicePool::instance().get_active_device(0);
            //DumpDeviceProfileResults(device_0);
            //std::vector<tracy::TTDeviceEvent> sync_data_sender;
            //std::vector<tracy::TTDeviceEvent> sync_data_receiver;
            //for (int dev_id = 0; dev_id < 2; dev_id ++) {
                //ZoneScopedN("Fetching_device_sync_event");
                //std::set<tracy::TTDeviceEvent>& sync_data = detail::getSyncDeviceData (dev_id);
                //for (auto& event: sync_data){
                    //if (event.zone_name.find("SENDER") != std::string::npos)
                    //{
                        //sync_data_sender.push_back(event);
                    //}
                    //else if (event.zone_name.find("RECEIVER") != std::string::npos)
                    //{
                        //sync_data_receiver.push_back(event);
                    //}
                //}
            //}

            //TT_ASSERT (sync_data_sender.size () == 120 * 2 * 2, "Wrong sync data sample count.");
            //TT_ASSERT (sync_data_receiver.size () == 120 * 2 * 2, "Wrong sync data sample count.");
            //double total_delta = 0;
            //double total = 0.0;
            //for ( int index = 0 ; index < 120 * 2 ; index += 2){

                //ZoneScopedN("TESTS");
                //int64_t sender_point_0 = (int64_t) (sync_data_sender[index].timestamp+ sync_data_sender[index+1].timestamp)/2 ;
                //int64_t receiver_point_0 = (int64_t)sync_data_receiver[index].timestamp;

                //int64_t sender_point_1 = (int64_t) (sync_data_sender[index + 240].timestamp+ sync_data_sender[index + 241].timestamp)/2 ;
                //int64_t receiver_point_1 = (int64_t)sync_data_receiver[index + 240].timestamp;

                //double diff_0 = receiver_point_0 - sender_point_0;
                //double diff_1 = sender_point_1 - receiver_point_1;
                //double scale = diff_0/diff_1;


                //std::cout << "Sync points: ";
                //std::cout << sender_point_0 << ",";
                //std::cout << receiver_point_0 << ",";
                //std::cout << sender_point_1 << ",";
                //std::cout << receiver_point_1 << std::endl;
                //total += scale;
            //}
            //double freqScale = total/120;

            //std::cout << std::fixed;
            //std::cout << std::setprecision(20);
            //std::cout << "FreqScale";
            //std::cout << freqScale << std::endl;

            //for ( int index = 0 ; index < 120 * 2 ; index += 2){
                //double sender_point_0 = (double) (sync_data_sender[index].timestamp+ sync_data_sender[index+1].timestamp)/2 ;
                //double receiver_point_0 = (double)sync_data_receiver[index].timestamp;

                //double sender_point_1 = (double)(sync_data_sender[index + 240].timestamp+ sync_data_sender[index + 241].timestamp)/2 ;
                //double receiver_point_1 = (double)sync_data_receiver[index + 240].timestamp;

                //double midpoint = ((receiver_point_0 + sender_point_0) - freqScale * (receiver_point_1 + sender_point_1))/2;
                //std::cout << midpoint << std::endl;
                //total_delta += midpoint;
            //}
            //double shift = total_delta/120;
            //std::cout << shift << " , " << freqScale << std::endl;
            //detail::setShift(1,int(shift), freqScale);
        //}
    });

#endif
}


void DumpDeviceProfileResults(Device *device, std::vector<CoreCoord> &worker_cores, bool lastDump){
#if defined(TRACY_ENABLE)
    ZoneScoped;
    std::string name = fmt::format("Device Dump {}", device->id());
    ZoneName(name.c_str(), name.size());
    std::scoped_lock<std::mutex> lock(device_mutex);
    auto dispatch_core_type = dispatch_core_manager::instance().get_dispatch_core_type(device->id());
    if (tt::llrt::OptionsG.get_profiler_do_dispatch_cores()) {
        auto device_id = device->id();
        auto device_num_hw_cqs = device->num_hw_cqs();
        for (const CoreCoord& core : tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs, dispatch_core_type)) {
            const auto curr_core = device->physical_core_from_logical_core(core, dispatch_core_type);
            worker_cores.push_back(curr_core);
        }
        for (const CoreCoord& core : tt::Cluster::instance().get_soc_desc(device_id).physical_ethernet_cores){
            worker_cores.push_back(core);
        }
    }
    if (getDeviceProfilerState())
    {
	if (!lastDump)
	{
	    const auto USE_FAST_DISPATCH = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr;
	    if (USE_FAST_DISPATCH)
	    {
		Finish(device->command_queue());
	    }
	}
        else
        {
            if (tt::llrt::OptionsG.get_profiler_do_dispatch_cores())
            {
                bool waitForDispatch = true;
                uint8_t loopCount = 0;
                CoreCoord unfinishedCore = {0,0};
                constexpr uint8_t maxLoopCount = 10;
                constexpr uint32_t loopDuration_us = 10000;
                while (waitForDispatch)
                {
                    waitForDispatch = false;
                    std::this_thread::sleep_for(std::chrono::microseconds(loopDuration_us));
                    auto device_id = device->id();
                    auto device_num_hw_cqs = device->num_hw_cqs();
                    loopCount++;
                    if (loopCount > maxLoopCount)
                    {
                        std::string msg = fmt::format(
                                "Device profiling never finished on device {}, worker core {}, {}",
                                device_id, unfinishedCore.x, unfinishedCore.y);
                        TracyMessageC(msg.c_str(), msg.size(), tracy::Color::Tomato3);
                        log_warning(msg.c_str());
                        break;
                    }
                    for (const CoreCoord& core :
                         tt::get_logical_dispatch_cores(device_id, device_num_hw_cqs, dispatch_core_type)) {
                        const auto curr_core = device->physical_core_from_logical_core(core, dispatch_core_type);
                        profiler_msg_t *profiler_msg = device->get_dev_addr<profiler_msg_t *>(curr_core, HalL1MemAddrType::PROFILER);
                        std::vector<std::uint32_t> control_buffer = tt::llrt::read_hex_vec_from_core(
                                device_id,
                                curr_core,
                                reinterpret_cast<uint64_t>(profiler_msg->control_vector),
                                kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);
                        if (control_buffer[kernel_profiler::PROFILER_DONE] == 0)
                        {
                            unfinishedCore = curr_core;
                            waitForDispatch = true;
                            continue;
                        }
                    }
                    if (waitForDispatch)
                    {
                        continue;
                    }
                    for (const CoreCoord& core : tt::Cluster::instance().get_soc_desc(device_id).physical_ethernet_cores)
                    {
                        profiler_msg_t *profiler_msg = device->get_dev_addr<profiler_msg_t *>(core, HalL1MemAddrType::PROFILER);
                        std::vector<std::uint32_t> control_buffer = tt::llrt::read_hex_vec_from_core(
                                device_id,
                                core,
                                reinterpret_cast<uint64_t>(profiler_msg->control_vector),
                                kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE);
                        if (control_buffer[kernel_profiler::PROFILER_DONE] == 0)
                        {
                            unfinishedCore = core;
                            waitForDispatch = true;
                            continue;
                        }
                    }

                }
            }
        }
	TT_FATAL(DprintServerIsRunning() == false, "Debug print server is running, cannot dump device profiler data");
        auto device_id = device->id();

        if (tt_metal_device_profiler_map.find(device_id) != tt_metal_device_profiler_map.end())
        {
            if (!lastDump)
            {
                syncDeviceHost (device, SYNC_CORE, tt_metal_device_profiler_map.at(device_id).sync_program, false);
            }
            tt_metal_device_profiler_map.at(device_id).setDeviceArchitecture(device->arch());
            tt_metal_device_profiler_map.at(device_id).dumpResults(device, worker_cores, lastDump);

            if (!lastDump)
            {
                for (auto& event: tt_metal_device_profiler_map.at(device_id).device_events){
                    if (event.zone_name.find("SYNC-ZONE") != std::string::npos)
                    {
                        ZoneScopedN("Adding_device_sync_event");
                        tt_metal_device_profiler_map.at(device_id).device_sync_events.insert(event);
                    }
                }
            }

            if (lastDump)
            {
                // Process is ending, no more device dumps are coming, reset your ref on the buffer so deallocate is the last
                // owner. Sync program also contains a buffer so it is safter to release it here
                tt_metal_device_profiler_map.at(device_id).output_dram_buffer.reset();
                tt_metal_device_profiler_map.at(device_id).sync_program.reset();
            }
            else
            {
                InitDeviceProfiler(device);
            }
        }
    }
#endif
}

void SetDeviceProfilerDir(std::string output_dir){
#if defined(TRACY_ENABLE)
    for (auto& device_id : tt_metal_device_profiler_map)
    {
        tt_metal_device_profiler_map.at(device_id.first).setOutputDir(output_dir);
    }
#endif
}

void FreshProfilerDeviceLog(){
#if defined(TRACY_ENABLE)
    for (auto& device_id : tt_metal_device_profiler_map)
    {
        tt_metal_device_profiler_map.at(device_id.first).freshDeviceLog();
    }
#endif
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
