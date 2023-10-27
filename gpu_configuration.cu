//
// Created by Shujian Qian on 2023-10-11.
//

#include "util_gpu_error_check.cuh"
#include "util_log.h"

namespace epic {

static const int gpu_configuration = []() -> int {
    auto &logger = Logger::GetInstance();
    logger.Info("Configuring GPU...");
//    gpu_err_check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
//    gpu_err_check(cudaSetDeviceFlags(cudaDeviceScheduleAuto));
    logger.Info("GPU configuration succeeded.");
    return 0;
}();

}