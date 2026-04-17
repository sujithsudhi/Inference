#pragma once

/// \file
/// \brief Whitespace-tokenizer fallback implementation.

#include "inference/tokenization/tokenizer.hpp"

namespace inference::tokenization
{

/// \brief Development tokenizer that splits text on ASCII whitespace boundaries.
class WhitespaceTokenizer final : public Tokenizer
{
public:
    /// \brief Return the tokenizer name.
    std::string Name() const override;

    /// \brief Encode whitespace-separated text into a deterministic synthetic vocabulary.
    std::vector<int32_t> Encode(std::string_view text) const override;

    /// \brief Decode synthetic token ids back into a whitespace-joined string when possible.
    std::string Decode(const std::vector<int32_t>& token_ids) const override;

    /// \brief Resolve a token string to its synthetic id when known.
    std::optional<int32_t> TokenToId(std::string_view token) const override;

    /// \brief Resolve a synthetic token id back to its token string when known.
    std::optional<std::string> IdToToken(int32_t token_id) const override;

    /// \brief Return the known vocabulary size when it has been materialized.
    std::optional<std::size_t> VocabSize() const override;
};

}  // namespace inference::tokenization
