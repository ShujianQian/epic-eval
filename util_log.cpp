//
// Created by Shujian Qian on 2023-08-13.
//

#include "util_log.h"

#include <mutex>

#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace {
std::mutex g_logger_mutex;
}

namespace epic {
Logger::Logger()
{
    spdlog::init_thread_pool(8192, 1);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%f] [%^%L%$] [thread %t] %v");

    logger = spdlog::stdout_color_mt<spdlog::async_factory>("epic");
    logger->set_level(spdlog::level::trace);
}

Logger &Logger::GetInstance()
{
    if (instance == nullptr)
    {
        std::lock_guard<std::mutex> lock(g_logger_mutex);
        if (instance == nullptr)
        {
            instance = std::make_unique<Logger>();
        }
    }
    return *instance;
}

std::unique_ptr<Logger> Logger::instance = nullptr;
} // namespace epic
