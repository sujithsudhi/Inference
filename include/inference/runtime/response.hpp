#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "inference/core/status.hpp"

namespace inference::runtime
{

struct Response
{
    core::Status         status = core::Status::Ok();
    std::string          text;
    std::vector<int32_t> prompt_token_ids;
    std::vector<int32_t> generated_token_ids;
};

}  // namespace inference::runtime
