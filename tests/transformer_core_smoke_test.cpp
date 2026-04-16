#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference/transformer_core.hpp"

using namespace inference::transformer_core;

namespace
{

void Expect(bool condition,
            const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void ExpectNames(const std::vector<std::string>& actual,
                 const std::vector<std::string>& expected,
                 const std::string&              label)
{
    Expect(actual.size() == expected.size(),
           label + " size mismatch.");

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        Expect(actual[i] == expected[i],
               label + " mismatch at index " + std::to_string(i)
               + ": expected '" + expected[i] + "' but received '" + actual[i] + "'.");
    }
}

StateDict BuildStateDict(const std::vector<TensorSpec>& specs)
{
    StateDict state_dict;
    float     next_value = 0.01F;

    for (const auto& spec : specs)
    {
        Tensor tensor(spec.shape, 0.0F);
        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            tensor.flat(i) = next_value;
            next_value += 0.01F;
        }
        state_dict.emplace(spec.name, tensor);
    }

    return state_dict;
}

void TestParameterNames()
{
    TokenEmbedding token_embedding(32, 8);
    ExpectNames(SpecNames(token_embedding.ParameterSpecs()),
                {"embedding.weight"},
                "TokenEmbedding");

    PositionalEncoding normal_pe(16, 8, 0.0F, PositionalEncodingMethod::Normal);
    Expect(normal_pe.ParameterSpecs().empty(),
           "Sinusoidal positional encoding should not expose state_dict parameters.");

    PositionalEncoding trainable_pe(16, 8, 0.0F, PositionalEncodingMethod::Trainable);
    ExpectNames(SpecNames(trainable_pe.ParameterSpecs()),
                {"positional_table"},
                "Trainable PositionalEncoding");

    TransformerEncoderLayer encoder(8, 2);
    ExpectNames(SpecNames(encoder.ParameterSpecs()),
                {"residual_attention.module.w_q.weight",
                 "residual_attention.module.w_q.bias",
                 "residual_attention.module.w_k.weight",
                 "residual_attention.module.w_k.bias",
                 "residual_attention.module.w_v.weight",
                 "residual_attention.module.w_v.bias",
                 "residual_attention.module.w_o.weight",
                 "residual_attention.module.w_o.bias",
                 "residual_attention.norm.weight",
                 "residual_attention.norm.bias",
                 "residual_mlp.module.fc1.weight",
                 "residual_mlp.module.fc1.bias",
                 "residual_mlp.module.fc2.weight",
                 "residual_mlp.module.fc2.bias",
                 "residual_mlp.norm.weight",
                 "residual_mlp.norm.bias"},
                "TransformerEncoderLayer");

    TransformerDecoderLayer decoder(8, 2);
    ExpectNames(SpecNames(decoder.ParameterSpecs()),
                SpecNames(encoder.ParameterSpecs()),
                "TransformerDecoderLayer");

    PatchEmbedding patch_embedding(8, 4, 3, 6);
    ExpectNames(SpecNames(patch_embedding.ParameterSpecs()),
                {"proj.weight", "proj.bias"},
                "PatchEmbedding");

    ViTEncoderLayer vit(8, 2);
    ExpectNames(SpecNames(vit.ParameterSpecs()),
                {"attention.w_q.weight",
                 "attention.w_q.bias",
                 "attention.w_k.weight",
                 "attention.w_k.bias",
                 "attention.w_v.weight",
                 "attention.w_v.bias",
                 "attention.w_o.weight",
                 "attention.w_o.bias",
                 "residual_attention.module.w_q.weight",
                 "residual_attention.module.w_q.bias",
                 "residual_attention.module.w_k.weight",
                 "residual_attention.module.w_k.bias",
                 "residual_attention.module.w_v.weight",
                 "residual_attention.module.w_v.bias",
                 "residual_attention.module.w_o.weight",
                 "residual_attention.module.w_o.bias",
                 "residual_attention.norm.weight",
                 "residual_attention.norm.bias",
                 "feed_forward.fc1.weight",
                 "feed_forward.fc1.bias",
                 "feed_forward.fc2.weight",
                 "feed_forward.fc2.bias",
                 "residual_mlp.module.fc1.weight",
                 "residual_mlp.module.fc1.bias",
                 "residual_mlp.module.fc2.weight",
                 "residual_mlp.module.fc2.bias",
                 "residual_mlp.norm.weight",
                 "residual_mlp.norm.bias"},
                "ViTEncoderLayer");
}

void TestForwardShapesAndLoading()
{
    DropPath drop_path(0.2F);
    Tensor residual_branch({1, 2, 4}, 1.0F);
    const Tensor drop_path_out = drop_path.Forward(residual_branch);
    Expect(drop_path.drop_prob() == 0.2F,
           "DropPath drop_prob accessor mismatch.");
    Expect(drop_path_out.shape() == residual_branch.shape(),
           "DropPath forward shape mismatch.");

    Linear linear(4, 3, true, "proj");
    linear.LoadParameters(BuildStateDict(linear.ParameterSpecs()));
    Expect(linear.weight().shape() == std::vector<std::int64_t>({3, 4}),
           "Linear weight accessor shape mismatch.");
    Expect(linear.bias().has_value()
           && linear.bias()->shape() == std::vector<std::int64_t>({3}),
           "Linear bias accessor shape mismatch.");
    const Tensor linear_out = linear.Forward(Tensor({1, 2, 4}, 1.0F));
    Expect(linear_out.shape() == std::vector<std::int64_t>({1, 2, 3}),
           "Linear forward shape mismatch.");

    LayerNorm layer_norm(4);
    layer_norm.LoadParameters(BuildStateDict(layer_norm.ParameterSpecs()));
    const Tensor normalized = layer_norm.Forward(Tensor({1, 2, 4}, 1.0F));
    Expect(normalized.shape() == std::vector<std::int64_t>({1, 2, 4}),
           "LayerNorm forward shape mismatch.");

    TokenEmbedding token_embedding(16, 4);
    token_embedding.LoadParameters(BuildStateDict(token_embedding.ParameterSpecs()));

    IndexTensor tokens({2, 3}, 0);
    tokens.at({0, 0}) = 1;
    tokens.at({0, 1}) = 2;
    tokens.at({0, 2}) = 3;
    tokens.at({1, 0}) = 4;
    tokens.at({1, 1}) = 5;
    tokens.at({1, 2}) = 6;

    const Tensor embedded = token_embedding.Forward(tokens);
    Expect(embedded.shape() == std::vector<std::int64_t>({2, 3, 4}),
           "TokenEmbedding forward shape mismatch.");

    PositionalEncoding trainable_pe(8, 4, 0.0F, PositionalEncodingMethod::Trainable);
    trainable_pe.LoadParameters(BuildStateDict(trainable_pe.ParameterSpecs()));
    const Tensor positioned = trainable_pe.Forward(embedded, 1);
    Expect(positioned.shape() == embedded.shape(),
           "PositionalEncoding forward shape mismatch.");

    RotaryEmbedding rope(4, 10000, 2);
    Tensor q({1, 2, 3, 4}, 1.0F);
    Tensor k({1, 2, 3, 4}, 2.0F);
    const auto rotated = rope.Forward(q, k, 2);
    Expect(rotated.first.shape() == q.shape() && rotated.second.shape() == k.shape(),
           "RoPE forward shape mismatch.");
    Expect(rope.max_seq_len() >= 5,
           "RoPE cache should grow when position_offset + seq_len exceeds the initial cache.");

    MultiHeadSelfAttention attention(8, 2);
    attention.LoadParameters(BuildStateDict(attention.ParameterSpecs()));
    attention.SetTrace(true, true);
    AttentionResult traced_attention = attention.Forward(Tensor({1, 3, 8}, 0.5F),
                                                         std::nullopt,
                                                         std::nullopt,
                                                         true,
                                                         false,
                                                         true);
    Expect(traced_attention.output.shape() == std::vector<std::int64_t>({1, 3, 8}),
           "Attention forward shape mismatch.");
    Expect(traced_attention.cache.has_value()
           && traced_attention.cache->key.shape() == std::vector<std::int64_t>({1, 2, 3, 4}),
           "Attention cache shape mismatch.");
    Expect(attention.last_attention_weights().has_value()
           && attention.last_q().has_value()
           && attention.last_k().has_value()
           && attention.last_v().has_value(),
           "Attention tracing outputs should be captured.");
    attention.ClearTrace();
    Expect(!attention.last_attention_weights().has_value()
           && !attention.last_q().has_value()
           && !attention.last_k().has_value()
           && !attention.last_v().has_value(),
           "Attention trace clear should reset captured tensors.");

    FeedForward feed_forward(8, 16, 8);
    feed_forward.LoadParameters(BuildStateDict(feed_forward.ParameterSpecs()));
    const Tensor feed_forward_out = feed_forward.Forward(Tensor({1, 2, 8}, 0.25F));
    Expect(feed_forward_out.shape() == std::vector<std::int64_t>({1, 2, 8}),
           "FeedForward forward shape mismatch.");

    TransformerDecoderLayer decoder(8, 2);
    decoder.LoadParameters(BuildStateDict(decoder.ParameterSpecs()));

    Tensor x0({1, 2, 8}, 0.5F);
    DecoderResult first = decoder.Forward(x0, std::nullopt, std::nullopt, true);
    Expect(first.output.shape() == std::vector<std::int64_t>({1, 2, 8}),
           "Decoder first forward output shape mismatch.");
    Expect(first.cache.has_value(),
           "Decoder should return cache when use_cache is enabled.");
    Expect(first.cache->key.shape() == std::vector<std::int64_t>({1, 2, 2, 4}),
           "Decoder cache key shape mismatch after first step.");

    Tensor x1({1, 1, 8}, 0.25F);
    DecoderResult second = decoder.Forward(x1, std::nullopt, first.cache, true);
    Expect(second.output.shape() == std::vector<std::int64_t>({1, 1, 8}),
           "Decoder cached forward output shape mismatch.");
    Expect(second.cache.has_value()
           && second.cache->key.shape() == std::vector<std::int64_t>({1, 2, 3, 4}),
           "Decoder cache should grow by one token.");

    PatchEmbedding patch_embedding(4, 2, 1, 3, true);
    patch_embedding.LoadParameters(BuildStateDict(patch_embedding.ParameterSpecs()));
    Tensor image({1, 1, 4, 4}, 1.0F);
    const Tensor patches = patch_embedding.Forward(image);
    Expect(patches.shape() == std::vector<std::int64_t>({1, 4, 3}),
           "PatchEmbedding flattened output shape mismatch.");

    ViTEncoderLayer vit(8, 2);
    vit.LoadParameters(BuildStateDict(vit.ParameterSpecs()));
    Tensor vit_input({1, 4, 8}, 0.1F);
    const Tensor vit_output = vit.Forward(vit_input, std::nullopt);
    Expect(vit_output.shape() == vit_input.shape(),
           "ViTEncoderLayer forward shape mismatch.");
}

}  // namespace

int main()
{
    try
    {
        TestParameterNames();
        TestForwardShapesAndLoading();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
