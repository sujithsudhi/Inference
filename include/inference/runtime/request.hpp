#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace inference::runtime
{

enum class InputKind
{
    Text,
    Image,
};

struct InputSegment
{
    InputKind              kind       = InputKind::Text;
    std::string            text;
    std::filesystem::path  image_path;
};

struct GenerationConfig
{
    int32_t max_new_tokens = 128;
    float   temperature    = 1.0F;
    int32_t top_k          = 0;
};

struct Request
{
    std::vector<InputSegment> segments;
    GenerationConfig          generation;
    bool                      return_token_ids = false;
};

}  // namespace inference::runtime
