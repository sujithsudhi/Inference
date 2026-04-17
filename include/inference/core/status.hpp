#pragma once

/// \file
/// \brief Lightweight status codes and status objects used by runtime APIs.

#include <string>

namespace inference::core
{

/// \brief Stable status codes used by lightweight runtime APIs.
enum class StatusCode
{
    /// Operation completed successfully.
    Ok = 0,
    /// Caller supplied invalid inputs or incompatible metadata.
    InvalidArgument,
    /// Requested artifact or resource could not be found.
    NotFound,
    /// Code path is recognized but not implemented yet.
    NotImplemented,
    /// Unexpected internal failure occurred while processing the request.
    InternalError,
};

/// \brief Lightweight status object returned by non-throwing runtime APIs.
class Status
{
public:
    /// \brief Construct an OK status with an empty message.
    Status();

    /// \brief Construct a status from one explicit code/message pair.
    Status(StatusCode  code,
           std::string message);

    /// \brief Return an OK status.
    static Status Ok();

    /// \brief Return an invalid-argument status.
    static Status InvalidArgument(std::string message);
    /// \brief Return a not-found status.
    static Status NotFound(std::string message);
    /// \brief Return a not-implemented status.
    static Status NotImplemented(std::string message);
    /// \brief Return an internal-error status.
    static Status InternalError(std::string message);

    /// \brief Indicate whether the status represents success.
    bool ok() const noexcept;

    /// \brief Return the stored status code.
    StatusCode code() const noexcept;

    /// \brief Return the stored explanatory message.
    const std::string& message() const noexcept;

private:
    StatusCode  code_;
    std::string message_;
};

}  // namespace inference::core
