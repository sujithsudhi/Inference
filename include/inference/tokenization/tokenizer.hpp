#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace inference::tokenization
{

class Tokenizer
{
public:
    virtual ~Tokenizer() = default;

    virtual std::string Name() const = 0;

    virtual std::vector<int32_t> Encode(std::string_view text) const = 0;

    virtual std::string Decode(const std::vector<int32_t>& token_ids) const = 0;

    virtual std::optional<int32_t> TokenToId(std::string_view token) const = 0;

    virtual std::optional<std::string> IdToToken(int32_t token_id) const = 0;

    virtual std::optional<std::size_t> VocabSize() const = 0;
};

using TokenizerPtr = std::shared_ptr<Tokenizer>;

}  // namespace inference::tokenization
