/// \file
/// \brief Whitespace-tokenizer fallback implementation.

#include "inference/tokenization/whitespace_tokenizer.hpp"

#include <cstdint>
#include <sstream>
#include <string>

namespace inference::tokenization
{
namespace
{

int32_t stable_token_id(const std::string& token)
{
    uint32_t value = 2166136261u;

    for (const unsigned char c : token)
    {
        value ^= c;
        value *= 16777619u;
    }

    return static_cast<int32_t>(value & 0x7fffffff);
}

}  // namespace

std::string WhitespaceTokenizer::Name() const
{
    return "whitespace";
}

std::vector<int32_t> WhitespaceTokenizer::Encode(std::string_view text) const
{
    std::vector<int32_t> token_ids;
    std::istringstream   stream{std::string(text)};
    std::string          token;

    while (stream >> token)
    {
        token_ids.push_back(stable_token_id(token));
    }

    return token_ids;
}

std::string WhitespaceTokenizer::Decode(const std::vector<int32_t>& token_ids) const
{
    std::ostringstream stream;

    for (size_t idx = 0; idx < token_ids.size(); ++idx)
    {
        if (idx > 0)
        {
            stream << ' ';
        }

        stream << '<' << token_ids[idx] << '>';
    }

    return stream.str();
}

std::optional<int32_t> WhitespaceTokenizer::TokenToId(std::string_view token) const
{
    return stable_token_id(std::string(token));
}

std::optional<std::string> WhitespaceTokenizer::IdToToken(int32_t token_id) const
{
    return std::to_string(token_id);
}

std::optional<std::size_t> WhitespaceTokenizer::VocabSize() const
{
    return std::nullopt;
}

}  // namespace inference::tokenization

