#pragma once

/// \file
/// \brief Abstract tokenizer interface used across runtime components.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace inference::tokenization
{

/// \brief Abstract tokenizer boundary used by runtime components and apps.
class Tokenizer
{
public:
    /// \brief Virtual destructor for tokenizer implementations.
    virtual ~Tokenizer() = default;

    /// \brief Return one stable tokenizer/backend name.
    virtual std::string Name() const = 0;

    /// \brief Encode text into integer token ids.
    virtual std::vector<int32_t> Encode(std::string_view text) const = 0;

    /// \brief Decode token ids back into text.
    virtual std::string Decode(const std::vector<int32_t>& token_ids) const = 0;

    /// \brief Resolve one token string to its integer id when the backend supports lookup.
    virtual std::optional<int32_t> TokenToId(std::string_view token) const = 0;

    /// \brief Resolve one token id back to a token string when the backend supports lookup.
    virtual std::optional<std::string> IdToToken(int32_t token_id) const = 0;

    /// \brief Return the vocabulary size when it is known by the backend.
    virtual std::optional<std::size_t> VocabSize() const = 0;
};

/// \brief Shared tokenizer pointer used across runtime APIs.
using TokenizerPtr = std::shared_ptr<Tokenizer>;

}  // namespace inference::tokenization
