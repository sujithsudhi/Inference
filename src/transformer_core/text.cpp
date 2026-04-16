#include "inference/transformer_core/text.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "detail.hpp"

namespace inference::transformer_core
{

namespace
{

void FillSinusoidalTable(Tensor&      table,
                         std::int64_t max_len,
                         std::int64_t embed_dim)
{
    for (std::int64_t position = 0; position < max_len; ++position)
    {
        for (std::int64_t dim = 0; dim < embed_dim; dim += 2)
        {
            const float exponent = -std::log(10000.0F) * static_cast<float>(dim) / static_cast<float>(embed_dim);
            const float angle    = static_cast<float>(position) * std::exp(exponent);

            table.at({0, position, dim}) = std::sin(angle);
            if (dim + 1 < embed_dim)
            {
                table.at({0, position, dim + 1}) = std::cos(angle);
            }
        }
    }
}

void FillTrainableTable(Tensor&      table,
                        std::int64_t max_len,
                        std::int64_t embed_dim)
{
    for (std::int64_t position = 0; position < max_len; ++position)
    {
        for (std::int64_t dim = 0; dim < embed_dim; ++dim)
        {
            const float angle = static_cast<float>((position + 1) * (dim + 1)) * 0.017F;
            const float value = 0.02F * std::sin(angle);
            table.at({0, position, dim}) = std::max(-0.04F, std::min(0.04F, value));
        }
    }
}

}  // namespace

TokenEmbedding::TokenEmbedding(std::int64_t                vocab_size,
                               std::int64_t                embed_dim,
                               std::optional<std::int64_t> padding_idx)
    : vocab_size_(vocab_size),
      embed_dim_(embed_dim),
      padding_idx_(padding_idx),
      embedding_weight_({vocab_size, embed_dim}, 0.0F)
{
    if (vocab_size <= 0 || embed_dim <= 0)
    {
        throw std::invalid_argument("TokenEmbedding dimensions must be positive.");
    }
}

Tensor TokenEmbedding::Forward(const IndexTensor& tokens) const
{
    if (tokens.rank() == 0)
    {
        throw std::invalid_argument("TokenEmbedding expects a tensor with rank >= 1.");
    }

    std::vector<std::int64_t> output_shape = tokens.shape();
    output_shape.push_back(embed_dim_);
    Tensor out(output_shape, 0.0F);

    for (std::size_t index = 0; index < tokens.numel(); ++index)
    {
        const auto token_id = tokens.data()[index];
        if (token_id < 0 || token_id >= vocab_size_)
        {
            throw std::out_of_range("TokenEmbedding token id out of range.");
        }

        const std::size_t output_offset = index * static_cast<std::size_t>(embed_dim_);
        const std::size_t weight_offset = static_cast<std::size_t>(token_id) * static_cast<std::size_t>(embed_dim_);
        for (std::int64_t dim = 0; dim < embed_dim_; ++dim)
        {
            out.flat(output_offset + static_cast<std::size_t>(dim)) =
                embedding_weight_.flat(weight_offset + static_cast<std::size_t>(dim));
        }
    }

    return out;
}

void TokenEmbedding::LoadParameters(const StateDict&   state_dict,
                                    const std::string& prefix)
{
    embedding_weight_ = detail::RequireTensor(state_dict,
                                              detail::JoinKey(prefix, "embedding.weight"),
                                              {vocab_size_, embed_dim_});
}

std::vector<TensorSpec> TokenEmbedding::ParameterSpecs(const std::string& prefix) const
{
    return {
        {detail::JoinKey(prefix, "embedding.weight"), {vocab_size_, embed_dim_}},
    };
}

PositionalEncoding::PositionalEncoding(std::int64_t             max_len,
                                       std::int64_t             embed_dim,
                                       float                    dropout,
                                       PositionalEncodingMethod method)
    : max_len_(max_len),
      embed_dim_(embed_dim),
      dropout_(dropout),
      method_(method),
      positional_table_({1, max_len, embed_dim}, 0.0F)
{
    if (max_len <= 0 || embed_dim <= 0)
    {
        throw std::invalid_argument("PositionalEncoding max_len and embed_dim must be positive.");
    }

    if (method_ == PositionalEncodingMethod::Trainable)
    {
        FillTrainableTable(positional_table_, max_len_, embed_dim_);
    }
    else
    {
        FillSinusoidalTable(positional_table_, max_len_, embed_dim_);
    }
}

Tensor PositionalEncoding::Forward(const Tensor& x,
                                   std::int64_t  offset) const
{
    if (x.rank() != 3 || x.dim(2) != embed_dim_)
    {
        throw std::invalid_argument("PositionalEncoding expects [batch, seq, embed_dim] inputs.");
    }
    if (offset < 0 || offset + x.dim(1) > max_len_)
    {
        throw std::invalid_argument("Requested positional window exceeds max_len.");
    }

    Tensor out(x.shape(), 0.0F);
    for (std::int64_t batch = 0; batch < x.dim(0); ++batch)
    {
        for (std::int64_t seq = 0; seq < x.dim(1); ++seq)
        {
            for (std::int64_t dim = 0; dim < embed_dim_; ++dim)
            {
                out.at({batch, seq, dim}) =
                    x.at({batch, seq, dim}) + positional_table_.at({0, offset + seq, dim});
            }
        }
    }

    return out;
}

void PositionalEncoding::LoadParameters(const StateDict&   state_dict,
                                        const std::string& prefix)
{
    if (method_ != PositionalEncodingMethod::Trainable)
    {
        return;
    }

    positional_table_ = detail::RequireTensor(state_dict,
                                              detail::JoinKey(prefix, "positional_table"),
                                              {1, max_len_, embed_dim_});
}

std::vector<TensorSpec> PositionalEncoding::ParameterSpecs(const std::string& prefix) const
{
    if (method_ != PositionalEncodingMethod::Trainable)
    {
        return {};
    }

    return {
        {detail::JoinKey(prefix, "positional_table"), {1, max_len_, embed_dim_}},
    };
}

TransformerEncoderLayer::TransformerEncoderLayer(std::int64_t                 embed_dim,
                                                 std::int64_t                 num_heads,
                                                 float                        mlp_ratio,
                                                 ActivationType               activation,
                                                 float                        attention_dropout,
                                                 float                        dropout,
                                                 bool                         norm_first,
                                                 bool                         flash_attention,
                                                 bool                         qkv_bias,
                                                 bool                         use_rope,
                                                 std::int64_t                 rope_base,
                                                 std::optional<std::int64_t>  mlp_hidden_dim,
                                                 float                        layer_norm_eps,
                                                 float                        drop_path)
    : attention_(std::make_shared<MultiHeadSelfAttention>(embed_dim,
                                                          num_heads,
                                                          attention_dropout,
                                                          flash_attention,
                                                          qkv_bias,
                                                          use_rope,
                                                          rope_base)),
      feed_forward_(std::make_shared<FeedForward>(embed_dim,
                                                 mlp_hidden_dim.value_or(static_cast<std::int64_t>(embed_dim * mlp_ratio)),
                                                 embed_dim,
                                                 activation,
                                                 dropout)),
      residual_attention_(embed_dim,
                          attention_,
                          dropout,
                          norm_first,
                          layer_norm_eps,
                          drop_path),
      residual_mlp_(embed_dim,
                    feed_forward_,
                    dropout,
                    norm_first,
                    layer_norm_eps,
                    drop_path)
{
}

Tensor TransformerEncoderLayer::Forward(const Tensor&                x,
                                        const std::optional<Tensor>& mask)
{
    AttentionResult attention_out = residual_attention_.Forward(x,
                                                                mask,
                                                                std::nullopt,
                                                                false,
                                                                false);
    return residual_mlp_.Forward(attention_out.output);
}

void TransformerEncoderLayer::LoadParameters(const StateDict&   state_dict,
                                             const std::string& prefix)
{
    residual_attention_.LoadParameters(state_dict, detail::JoinKey(prefix, "residual_attention."));
    residual_mlp_.LoadParameters(state_dict, detail::JoinKey(prefix, "residual_mlp."));
}

std::vector<TensorSpec> TransformerEncoderLayer::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs,
                        residual_attention_.ParameterSpecs(detail::JoinKey(prefix, "residual_attention.")));
    detail::AppendSpecs(specs,
                        residual_mlp_.ParameterSpecs(detail::JoinKey(prefix, "residual_mlp.")));
    return specs;
}

TransformerDecoderLayer::TransformerDecoderLayer(std::int64_t                 embed_dim,
                                                 std::int64_t                 num_heads,
                                                 float                        mlp_ratio,
                                                 ActivationType               activation,
                                                 float                        attention_dropout,
                                                 float                        dropout,
                                                 bool                         norm_first,
                                                 bool                         flash_attention,
                                                 bool                         qkv_bias,
                                                 bool                         use_rope,
                                                 std::int64_t                 rope_base,
                                                 std::optional<std::int64_t>  mlp_hidden_dim,
                                                 float                        layer_norm_eps,
                                                 float                        drop_path)
    : attention_(std::make_shared<MultiHeadSelfAttention>(embed_dim,
                                                          num_heads,
                                                          attention_dropout,
                                                          flash_attention,
                                                          qkv_bias,
                                                          use_rope,
                                                          rope_base)),
      feed_forward_(std::make_shared<FeedForward>(embed_dim,
                                                 mlp_hidden_dim.value_or(static_cast<std::int64_t>(embed_dim * mlp_ratio)),
                                                 embed_dim,
                                                 activation,
                                                 dropout)),
      residual_attention_(embed_dim,
                          attention_,
                          dropout,
                          norm_first,
                          layer_norm_eps,
                          drop_path),
      residual_mlp_(embed_dim,
                    feed_forward_,
                    dropout,
                    norm_first,
                    layer_norm_eps,
                    drop_path)
{
}

DecoderResult TransformerDecoderLayer::Forward(const Tensor&                      x,
                                               const std::optional<Tensor>&       mask,
                                               const std::optional<KeyValueCache>& past_kv,
                                               bool                               use_cache)
{
    const std::int64_t past_len = past_kv.has_value() ? past_kv->key.dim(2) : 0;

    std::optional<Tensor> attn_mask = std::nullopt;
    bool                  is_causal = false;

    if (!(use_cache && past_kv.has_value() && x.dim(1) == 1))
    {
        if (!mask.has_value())
        {
            is_causal = true;
        }
        else
        {
            attn_mask = BuildCausalMask(x, *mask, past_len);
        }
    }

    AttentionResult attention_out = residual_attention_.Forward(x,
                                                                attn_mask,
                                                                past_kv,
                                                                use_cache,
                                                                is_causal);

    DecoderResult result;
    result.output = residual_mlp_.Forward(attention_out.output);
    result.cache  = attention_out.cache;
    return result;
}

void TransformerDecoderLayer::LoadParameters(const StateDict&   state_dict,
                                             const std::string& prefix)
{
    residual_attention_.LoadParameters(state_dict, detail::JoinKey(prefix, "residual_attention."));
    residual_mlp_.LoadParameters(state_dict, detail::JoinKey(prefix, "residual_mlp."));
}

std::vector<TensorSpec> TransformerDecoderLayer::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs,
                        residual_attention_.ParameterSpecs(detail::JoinKey(prefix, "residual_attention.")));
    detail::AppendSpecs(specs,
                        residual_mlp_.ParameterSpecs(detail::JoinKey(prefix, "residual_mlp.")));
    return specs;
}

Tensor TransformerDecoderLayer::BuildCausalMask(const Tensor& x,
                                                const Tensor& mask,
                                                std::int64_t  past_len) const
{
    if (x.rank() != 3)
    {
        throw std::invalid_argument("TransformerDecoderLayer expects [batch, seq, embed_dim] inputs.");
    }

    const auto batch_size = x.dim(0);
    const auto seq_len    = x.dim(1);
    const auto total_len  = past_len + seq_len;

    Tensor causal({batch_size, seq_len, total_len}, 0.0F);
    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t query = 0; query < seq_len; ++query)
        {
            const std::int64_t absolute_query = total_len - seq_len + query;
            for (std::int64_t key = 0; key < total_len; ++key)
            {
                causal.at({batch, query, key}) = key <= absolute_query ? 1.0F : 0.0F;
            }
        }
    }

    if (mask.rank() == 2)
    {
        if (mask.dim(0) != batch_size || mask.dim(1) != total_len)
        {
            throw std::invalid_argument("Padding mask length does not match total sequence length.");
        }

        for (std::int64_t batch = 0; batch < batch_size; ++batch)
        {
            for (std::int64_t query = 0; query < seq_len; ++query)
            {
                for (std::int64_t key = 0; key < total_len; ++key)
                {
                    causal.at({batch, query, key}) *= mask.at({batch, key}) > 0.0F ? 1.0F : 0.0F;
                }
            }
        }
        return causal;
    }

    if (mask.rank() == 3)
    {
        if (mask.dim(0) != batch_size
            || mask.dim(1) != seq_len
            || mask.dim(2) != total_len)
        {
            throw std::invalid_argument("Explicit decoder mask shape does not match [batch, seq, total_len].");
        }

        for (std::int64_t batch = 0; batch < batch_size; ++batch)
        {
            for (std::int64_t query = 0; query < seq_len; ++query)
            {
                for (std::int64_t key = 0; key < total_len; ++key)
                {
                    causal.at({batch, query, key}) *= mask.at({batch, query, key}) > 0.0F ? 1.0F : 0.0F;
                }
            }
        }
        return causal;
    }

    throw std::invalid_argument("Unsupported attention mask rank.");
}

}  // namespace inference::transformer_core
