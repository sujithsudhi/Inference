/// \file
/// \brief Tokenizer factory and backend-loading implementation.

#include "inference/tokenization/tokenizer_factory.hpp"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#if defined(INFERENCE_HAS_TOKENIZERS_CPP)
#include <tokenizers_cpp.h>
#endif

namespace inference::tokenization
{

namespace
{

#if defined(INFERENCE_HAS_TOKENIZERS_CPP)

class TokenizersCppTokenizer final : public Tokenizer
{
public:
    TokenizersCppTokenizer(std::string                                name,
                           std::unique_ptr<tokenizers::Tokenizer>     impl)
        : name_(std::move(name)),
          impl_(std::move(impl))
    {
        if (!impl_)
        {
            throw std::invalid_argument("TokenizersCppTokenizer requires a valid tokenizer instance.");
        }
    }

    std::string Name() const override
    {
        return name_;
    }

    std::vector<int32_t> Encode(std::string_view text) const override
    {
        const std::vector<int> encoded = impl_->Encode(std::string(text));
        return std::vector<int32_t>(encoded.begin(), encoded.end());
    }

    std::string Decode(const std::vector<int32_t>& token_ids) const override
    {
        return impl_->Decode(std::vector<int>(token_ids.begin(), token_ids.end()));
    }

    std::optional<int32_t> TokenToId(std::string_view token) const override
    {
        const int32_t token_id = impl_->TokenToId(std::string(token));
        return token_id >= 0 ? std::optional<int32_t>(token_id) : std::nullopt;
    }

    std::optional<std::string> IdToToken(int32_t token_id) const override
    {
        const std::string token = impl_->IdToToken(token_id);
        return token.empty() ? std::nullopt : std::optional<std::string>(token);
    }

    std::optional<std::size_t> VocabSize() const override
    {
        return impl_->GetVocabSize();
    }

private:
    std::string                            name_;
    std::unique_ptr<tokenizers::Tokenizer> impl_;
};

std::string ReadBlob(const std::filesystem::path& path)
{
    std::ifstream handle(path, std::ios::binary);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open tokenizer file: " + path.string());
    }

    return std::string(std::istreambuf_iterator<char>(handle), std::istreambuf_iterator<char>());
}

TokenizerPtr LoadTokenizerWithTokenizersCpp(const std::filesystem::path& path)
{
    const std::string blob = ReadBlob(path);
    const auto        extension = path.extension().string();

    if (extension == ".json")
    {
        return std::make_shared<TokenizersCppTokenizer>("tokenizers-cpp[hf-json]",
                                                        tokenizers::Tokenizer::FromBlobJSON(blob));
    }

    if (extension == ".model")
    {
        return std::make_shared<TokenizersCppTokenizer>("tokenizers-cpp[sentencepiece]",
                                                        tokenizers::Tokenizer::FromBlobSentencePiece(blob));
    }

    throw std::runtime_error("Unsupported tokenizer file extension: " + path.string());
}

#endif

}  // namespace

TokenizerPtr LoadTokenizer(const std::filesystem::path& path)
{
    if (path.empty())
    {
        throw std::invalid_argument("Tokenizer path is required.");
    }

    if (!std::filesystem::exists(path))
    {
        throw std::runtime_error("Tokenizer file not found: " + path.string());
    }

#if defined(INFERENCE_HAS_TOKENIZERS_CPP)
    return LoadTokenizerWithTokenizersCpp(path);
#else
    throw std::runtime_error("This build does not include tokenizers-cpp support. "
                             "Reconfigure with Rust/Cargo available so INFERENCE_ENABLE_TOKENIZERS_CPP can build.");
#endif
}

}  // namespace inference::tokenization
