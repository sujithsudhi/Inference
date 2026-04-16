#pragma once

#include <cstdint>
#include <memory>
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
};

using TokenizerPtr = std::shared_ptr<Tokenizer>;

}  // namespace inference::tokenization
