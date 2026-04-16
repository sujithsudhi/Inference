#pragma once

#include <string>

namespace inference::core
{

enum class StatusCode
{
    Ok = 0,
    InvalidArgument,
    NotFound,
    NotImplemented,
    InternalError,
};

class Status
{
public:
    Status();

    Status(StatusCode  code,
           std::string message);

    static Status Ok();

    static Status InvalidArgument(std::string message);
    static Status NotFound(std::string message);
    static Status NotImplemented(std::string message);
    static Status InternalError(std::string message);

    bool ok() const noexcept;

    StatusCode code() const noexcept;

    const std::string& message() const noexcept;

private:
    StatusCode  code_;
    std::string message_;
};

}  // namespace inference::core
