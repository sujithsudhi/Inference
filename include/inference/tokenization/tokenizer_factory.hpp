#pragma once

/// \file
/// \brief Tokenizer factory helpers and backend capability probes.

#include <filesystem>

#include "inference/tokenization/tokenizer.hpp"

namespace inference::tokenization
{

/// \brief Load one tokenizer artifact through the configured backend.
TokenizerPtr LoadTokenizer(const std::filesystem::path& path);

/// \brief Construct the built-in whitespace tokenizer fallback.
TokenizerPtr CreateWhitespaceTokenizer();

/// \brief Indicate whether the `tokenizers-cpp` backend is available in the current build.
bool HasTokenizersCppBackend() noexcept;

}  // namespace inference::tokenization
