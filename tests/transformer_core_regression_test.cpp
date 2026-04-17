/// \file
/// \brief Regression tests for transformer-core behavior and model surfaces.

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference/transformer_core.hpp"

namespace
{

using namespace inference::transformer_core;

void Expect(bool               condition,
            const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

StateDict BuildStateDict(const std::vector<TensorSpec>& specs)
{
    StateDict state_dict;
    float     next_value = 0.01F;

    for (const auto& spec : specs)
    {
        Tensor tensor(spec.shape, 0.0F);
        for (std::size_t index = 0; index < tensor.numel(); ++index)
        {
            tensor.flat(index) = next_value;
            next_value += 0.01F;
        }
        state_dict.emplace(spec.name, std::move(tensor));
    }

    return state_dict;
}

void ExpectTensorsClose(const Tensor&       lhs,
                        const Tensor&       rhs,
                        float               tolerance,
                        const std::string&  label)
{
    Expect(lhs.shape() == rhs.shape(),
           label + " shape mismatch.");

    for (std::size_t index = 0; index < lhs.numel(); ++index)
    {
        const float diff = std::fabs(lhs.flat(index) - rhs.flat(index));
        if (diff > tolerance)
        {
            throw std::runtime_error(label + " mismatch at flat index "
                                     + std::to_string(index)
                                     + ": diff=" + std::to_string(diff));
        }
    }
}

Tensor SliceSequence(const Tensor&   x,
                     std::int64_t    start,
                     std::int64_t    length)
{
    Expect(x.rank() == 3,
           "SliceSequence expects a rank-3 tensor.");

    Tensor out({x.dim(0), length, x.dim(2)}, 0.0F);
    for (std::int64_t batch = 0; batch < x.dim(0); ++batch)
    {
        for (std::int64_t seq = 0; seq < length; ++seq)
        {
            for (std::int64_t dim = 0; dim < x.dim(2); ++dim)
            {
                out.at({batch, seq, dim}) = x.at({batch, start + seq, dim});
            }
        }
    }

    return out;
}

bool RowsDiffer(const Tensor& x,
                std::int64_t  lhs_row,
                std::int64_t  rhs_row)
{
    for (std::int64_t dim = 0; dim < x.dim(1); ++dim)
    {
        if (std::fabs(x.at({lhs_row, dim}) - x.at({rhs_row, dim})) > 1e-6F)
        {
            return true;
        }
    }

    return false;
}

void TestPaddingMaskNormalization()
{
    MultiHeadSelfAttention attention(8, 2);
    attention.LoadParameters(BuildStateDict(attention.ParameterSpecs()));

    Tensor x({2, 2, 8}, 0.0F);
    for (std::size_t index = 0; index < x.numel(); ++index)
    {
        x.flat(index) = 0.1F + static_cast<float>(index) * 0.01F;
    }

    Tensor padding_mask({2, 2}, 0.0F);
    padding_mask.at({0, 0}) = 1.0F;
    padding_mask.at({1, 1}) = 1.0F;

    Tensor explicit_mask({2, 1, 1, 2}, 0.0F);
    explicit_mask.at({0, 0, 0, 0}) = 1.0F;
    explicit_mask.at({1, 0, 0, 1}) = 1.0F;

    const AttentionResult with_padding = attention.Forward(x,
                                                           padding_mask,
                                                           std::nullopt,
                                                           false,
                                                           false,
                                                           true);
    const AttentionResult with_explicit = attention.Forward(x,
                                                            explicit_mask,
                                                            std::nullopt,
                                                            false,
                                                            false,
                                                            true);

    ExpectTensorsClose(with_padding.output,
                       with_explicit.output,
                       1e-5F,
                       "Padding-mask normalization");
    ExpectTensorsClose(with_padding.attention_weights,
                       with_explicit.attention_weights,
                       1e-5F,
                       "Padding-mask attention weights");
}

void TestCachedChunkedDecoderMatchesFullSequence()
{
    TransformerDecoderLayer decoder(8, 2);
    decoder.LoadParameters(BuildStateDict(decoder.ParameterSpecs()));

    Tensor full_input({1, 4, 8}, 0.0F);
    for (std::size_t index = 0; index < full_input.numel(); ++index)
    {
        full_input.flat(index) = 0.05F + static_cast<float>(index) * 0.02F;
    }

    const DecoderResult full = decoder.Forward(full_input, std::nullopt, std::nullopt, false);

    const Tensor prefix = SliceSequence(full_input, 0, 2);
    const Tensor chunk  = SliceSequence(full_input, 2, 2);

    const DecoderResult first = decoder.Forward(prefix, std::nullopt, std::nullopt, true);
    Expect(first.cache.has_value(),
           "Decoder should return a cache for the prefix chunk.");

    const DecoderResult second = decoder.Forward(chunk, std::nullopt, first.cache, false);
    const Tensor        expected_chunk = SliceSequence(full.output, 2, 2);

    ExpectTensorsClose(second.output,
                       expected_chunk,
                       1e-5F,
                       "Cached decoder chunk output");
}

void TestHighLevelTransformerModels()
{
    VisionTransformerConfig vision_config;
    vision_config.image_size  = 4;
    vision_config.patch_size  = 2;
    vision_config.in_channels = 1;
    vision_config.embed_dim   = 8;
    vision_config.num_layers  = 1;
    vision_config.num_heads   = 2;
    vision_config.mlp_ratio   = 2.0F;

    VisionTransformer vision_model(vision_config);
    vision_model.LoadParameters(BuildStateDict(vision_model.ParameterSpecs()));

    Tensor images({2, 1, 4, 4}, 0.0F);
    for (std::size_t index = 0; index < images.numel(); ++index)
    {
        images.flat(index) = static_cast<float>(index) / 16.0F;
    }

    const Tensor vision_output = vision_model.Forward(images);
    Expect(vision_output.shape() == std::vector<std::int64_t>({2, 8}),
           "VisionTransformer batch forward shape mismatch.");
    Expect(RowsDiffer(vision_output, 0, 1),
           "VisionTransformer should preserve per-sample differences across the batch.");

    TextTransformerConfig text_config;
    text_config.vocab_size  = 32;
    text_config.max_length  = 4;
    text_config.embed_dim   = 8;
    text_config.depth       = 1;
    text_config.num_heads   = 2;
    text_config.mlp_ratio   = 2.0F;
    text_config.num_classes = 3;

    TextTransformer text_model(text_config);
    text_model.LoadParameters(BuildStateDict(text_model.ParameterSpecs()));

    Tensor token_ids({2, 4}, 0.0F);
    token_ids.at({0, 0}) = 1.0F;
    token_ids.at({0, 1}) = 2.0F;
    token_ids.at({0, 2}) = 3.0F;
    token_ids.at({0, 3}) = 4.0F;
    token_ids.at({1, 0}) = 4.0F;
    token_ids.at({1, 1}) = 3.0F;
    token_ids.at({1, 2}) = 2.0F;
    token_ids.at({1, 3}) = 1.0F;

    const Tensor text_output = text_model.Forward(token_ids);
    Expect(text_output.shape() == std::vector<std::int64_t>({2, 3}),
           "TextTransformer forward shape mismatch.");
}

}  // namespace

int main()
{
    try
    {
        TestPaddingMaskNormalization();
        TestCachedChunkedDecoderMatchesFullSequence();
        TestHighLevelTransformerModels();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
