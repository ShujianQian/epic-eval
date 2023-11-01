//
// Created by Shujian Qian on 2023-08-13.
//

#ifndef UTIL_LOG_H
#define UTIL_LOG_H

#include <memory>
#include <string_view>

#include "spdlog/spdlog.h"

namespace epic {
class Logger
{
public:
    Logger(Logger &) = delete;
    void operator=(const Logger &) = delete;

    static Logger &GetInstance();

    template<typename... Args>
    void Trace(std::string_view fmt, Args... args)
    {
        logger->trace(fmt, args...);
    }

    template<typename... Args>
    void Debug(std::string_view fmt, Args... args)
    {
        logger->debug(fmt, args...);
    }

    template<typename... Args>
    void Info(std::string_view fmt, Args... args)
    {
        logger->info(fmt, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void Warn(std::string_view fmt, Args... args)
    {
        logger->warn(fmt, args...);
    }

    template<typename... Args>
    void Error(std::string_view fmt, Args... args)
    {
        logger->error(fmt, args...);
    }

    void flush() {
        logger->flush();
    }

    std::shared_ptr<spdlog::logger> logger = nullptr;

private:
    static std::unique_ptr<Logger> instance;
    Logger();

    friend std::unique_ptr<Logger> std::make_unique<Logger>();
};
} // namespace epic

#endif // UTIL_LOG_H
