#pragma once

/// \file
/// \brief Generic runtime request types shared by adapter-backed flows.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace inference::runtime
{

/// \brief Supported multimodal segment kinds accepted by runtime requests.
enum class InputKind
{
    /// Plain-text input segment.
    Text,
    /// Image-path input segment.
    Image,
};

/// \brief One ordered request segment passed to an adapter-backed runtime.
struct InputSegment
{
    /// Segment modality.
    InputKind              kind       = InputKind::Text;
    /// Text payload used when `kind == InputKind::Text`.
    std::string            text;
    /// Filesystem path used when `kind == InputKind::Image`.
    std::filesystem::path  image_path;
};

/// \brief Generation-time controls shared by adapter-backed text runtimes.
struct GenerationConfig
{
    /// Maximum number of tokens the runtime should generate.
    int32_t max_new_tokens = 128;
    /// Sampling temperature applied by stochastic decoders.
    float   temperature    = 1.0F;
    /// Optional top-k truncation applied during sampling. Zero disables it.
    int32_t top_k          = 0;
};

/// \brief Generic runtime request consumed by `ModelAdapter` implementations and `Session`.
struct Request
{
    /// Ordered multimodal input segments.
    std::vector<InputSegment> segments;
    /// Optional generation controls for decoder-style runtimes.
    GenerationConfig          generation;
    /// Whether the runtime should surface prompt token ids in the response when available.
    bool                      return_token_ids = false;
};

}  // namespace inference::runtime
