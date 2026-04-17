#pragma once

/// \file
/// \brief Tokenizer factory helpers and backend capability probes.

#include <filesystem>

#include "inference/tokenization/tokenizer.hpp"

namespace inference::tokenization
{

/// \brief Load one tokenizer artifact through the configured backend.
TokenizerPtr LoadTokenizer(const std::filesystem::path& path);

}  // namespace inference::tokenization
