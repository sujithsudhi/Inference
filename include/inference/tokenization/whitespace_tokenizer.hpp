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

    std::optional<int32_t> TokenToId(std::string_view token) const override;

    std::optional<std::string> IdToToken(int32_t token_id) const override;

    std::optional<std::size_t> VocabSize() const override;
};

}  // namespace inference::tokenization
