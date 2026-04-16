#pragma once

#include "inference/tokenization/tokenizer.hpp"

namespace inference::tokenization
{

class WhitespaceTokenizer final : public Tokenizer
{
public:
    std::string Name() const override;

    std::vector<int32_t> Encode(std::string_view text) const override;

    std::string Decode(const std::vector<int32_t>& token_ids) const override;
};

}  // namespace inference::tokenization
