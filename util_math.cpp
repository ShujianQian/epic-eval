//
// Created by Shujian Qian on 2023-08-18.
//

#include "util_math.h"

#include <array>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace epic {

std::string formatSizeBytes(uint64_t bytes)
{
    static const char *suffixes[] = {"B", "KB", "MB", "GB", "TB", "PB"};
    static const size_t suffixesSize = sizeof(suffixes) / sizeof(suffixes[0]);
    const double kilobyte = 1024.0;

    size_t i = 0;
    double size = static_cast<double>(bytes);
    while (size > kilobyte && i < suffixesSize - 1)
    {
        size /= kilobyte;
        ++i;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << suffixes[i];
    return ss.str();
}

} // namespace epic