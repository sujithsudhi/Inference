#pragma once

#include <filesystem>

#include "inference/tokenization/tokenizer.hpp"

namespace inference::tokenization
{

TokenizerPtr LoadTokenizer(const std::filesystem::path& path);

TokenizerPtr CreateWhitespaceTokenizer();

bool HasTokenizersCppBackend() noexcept;

}  // namespace inference::tokenization
