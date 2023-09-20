//
// Created by Shujian Qian on 2023-08-30.
//

#ifndef COMMON_H
#define COMMON_H

#ifdef __host__
#define EPIC_HOST __host__
#else
#define EPIC_HOST
#endif

#ifdef __device__
#define EPIC_DEVICE __device__
#else
#define EPIC_DEVICE
#endif

#endif // COMMON_H
