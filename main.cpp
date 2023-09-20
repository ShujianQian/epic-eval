//
// Created by Shujian Qian on 2023-07-18.
//

#include "epic_gpu.cuh"

#include "util_log.h"
#include "util_math.h"

int main(int argc, char **argv)
{

#ifdef CUDA_AVAILABLE
    epic_gpu(argc, argv);

    run_thrust_experiment(argc, argv);
#endif

    epic::Logger &logger = epic::Logger::GetInstance();
    logger.Trace("help {}", epic::formatSizeBytes(1024 * 1024 * 1024));

    return 0;
}