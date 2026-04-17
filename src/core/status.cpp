/// \file
/// \brief Lightweight status helper implementation.

#include "inference/core/status.hpp"

#include <utility>

namespace inference::core
{

Status::Status()
: code_(StatusCode::Ok),
  message_()
{
}

Status::Status(StatusCode  code,
               std::string message)
: code_(code),
  message_(std::move(message))
{
}

Status Status::Ok()
{
    return Status();
}

Status Status::InvalidArgument(std::string message)
{
    return Status(StatusCode::InvalidArgument, std::move(message));
}

Status Status::NotFound(std::string message)
{
    return Status(StatusCode::NotFound, std::move(message));
}

Status Status::NotImplemented(std::string message)
{
    return Status(StatusCode::NotImplemented, std::move(message));
}

Status Status::InternalError(std::string message)
{
    return Status(StatusCode::InternalError, std::move(message));
}

bool Status::ok() const noexcept
{
    return code_ == StatusCode::Ok;
}

StatusCode Status::code() const noexcept
{
    return code_;
}

const std::string& Status::message() const noexcept
{
    return message_;
}

}  // namespace inference::core
