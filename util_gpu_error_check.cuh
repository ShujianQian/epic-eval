//
// Created by Shujian Qian on 2023-08-13.
//

#ifndef UTIL_GPU_ERROR_CHECK_CUH
#define UTIL_GPU_ERROR_CHECK_CUH

#include <cstdio>

#ifdef EPIC_CUDA_AVAILABLE
#    ifdef __CUDACC__

#        define gpu_err_check(ans) gpu_err_check_impl((ans), __FILE__, __LINE__)
inline void gpu_err_check_impl(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            fflush(stderr);
            exit(code);
        }
    }
}

#    endif // __CUDACC__
#endif     // EPIC_CUDA_AVAILABLE

#endif // UTIL_GPU_ERROR_CHECK_CUH
