/// \file
/// \brief Shared transformer-core layer and attention implementation.

#include "inference/transformer_core/common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "detail.hpp"

namespace inference::transformer_core
{

namespace
{

void ValidateLastDim(const Tensor&      tensor,
                     std::int64_t       expected_dim,
                     const std::string& context)
{
    if (tensor.rank() == 0)
    {
        throw std::invalid_argument(context + " expects a tensor with rank >= 1.");
    }

    if (tensor.dim(tensor.rank() - 1) != expected_dim)
    {
        throw std::invalid_argument(context + " expects the last dimension to be "
                                    + std::to_string(expected_dim) + ".");
    }
}

Tensor Add(const Tensor& lhs,
           const Tensor& rhs)
{
    if (!detail::ShapesEqual(lhs.shape(), rhs.shape()))
    {
        throw std::invalid_argument("Tensor add expects matching shapes.");
    }

    Tensor out(lhs.shape());
    for (std::size_t i = 0; i < lhs.numel(); ++i)
    {
        out.flat(i) = lhs.flat(i) + rhs.flat(i);
    }
    return out;
}

float Activate(float          value,
               ActivationType activation)
{
    switch (activation)
    {
        case ActivationType::Identity:
            return value;
        case ActivationType::Gelu:
            return 0.5F * value * (1.0F + std::erf(value / std::sqrt(2.0F)));
        case ActivationType::Relu:
            return std::max(0.0F, value);
        case ActivationType::Silu:
            return value / (1.0F + std::exp(-value));
    }

    throw std::invalid_argument("Unsupported activation type.");
}

Tensor ApplyActivation(const Tensor& x,
                       ActivationType activation)
{
    Tensor out(x.shape());
    for (std::size_t i = 0; i < x.numel(); ++i)
    {
        out.flat(i) = Activate(x.flat(i), activation);
    }
    return out;
}

Tensor ConcatSequence(const Tensor& lhs,
                      const Tensor& rhs)
{
    if (lhs.rank() != 4 || rhs.rank() != 4)
    {
        throw std::invalid_argument("KV cache concatenation expects rank-4 tensors.");
    }

    if (lhs.dim(0) != rhs.dim(0)
        || lhs.dim(1) != rhs.dim(1)
        || lhs.dim(3) != rhs.dim(3))
    {
        throw std::invalid_argument("KV cache tensors must match on batch, head, and head_dim.");
    }

    const auto batch_size = lhs.dim(0);
    const auto num_heads  = lhs.dim(1);
    const auto lhs_seq    = lhs.dim(2);
    const auto rhs_seq    = rhs.dim(2);
    const auto head_dim   = lhs.dim(3);

    Tensor out({batch_size, num_heads, lhs_seq + rhs_seq, head_dim});

    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t head = 0; head < num_heads; ++head)
        {
            for (std::int64_t seq = 0; seq < lhs_seq; ++seq)
            {
                for (std::int64_t dim = 0; dim < head_dim; ++dim)
                {
                    out.at({batch, head, seq, dim}) = lhs.at({batch, head, seq, dim});
                }
            }

            for (std::int64_t seq = 0; seq < rhs_seq; ++seq)
            {
                for (std::int64_t dim = 0; dim < head_dim; ++dim)
                {
                    out.at({batch, head, lhs_seq + seq, dim}) = rhs.at({batch, head, seq, dim});
                }
            }
        }
    }

    return out;
}

Tensor NormalizeAttentionMask(const Tensor& mask,
                              std::int64_t  batch_size,
                              std::int64_t  query_len,
                              std::int64_t  key_len)
{
    if (mask.rank() == 2)
    {
        if (mask.dim(0) == batch_size && mask.dim(1) == key_len)
        {
            Tensor out({batch_size, 1, 1, key_len});
            for (std::int64_t batch = 0; batch < batch_size; ++batch)
            {
                for (std::int64_t key = 0; key < key_len; ++key)
                {
                    out.at({batch, 0, 0, key}) = mask.at({batch, key});
                }
            }
            return out;
        }

        if (mask.dim(0) == query_len && mask.dim(1) == key_len)
        {
            Tensor out({1, 1, query_len, key_len});
            for (std::int64_t query = 0; query < query_len; ++query)
            {
                for (std::int64_t key = 0; key < key_len; ++key)
                {
                    out.at({0, 0, query, key}) = mask.at({query, key});
                }
            }
            return out;
        }

        throw std::invalid_argument("Rank-2 attention masks must be shaped as [batch, key_len] "
                                    "or [query_len, key_len].");
    }

    if (mask.rank() == 3)
    {
        if (mask.dim(0) != batch_size
            || mask.dim(1) != query_len
            || mask.dim(2) != key_len)
        {
            throw std::invalid_argument("Rank-3 attention masks must be shaped as "
                                        "[batch, query_len, key_len].");
        }

        Tensor out({mask.dim(0), 1, mask.dim(1), mask.dim(2)});
        for (std::int64_t batch = 0; batch < mask.dim(0); ++batch)
        {
            for (std::int64_t query = 0; query < mask.dim(1); ++query)
            {
                for (std::int64_t key = 0; key < mask.dim(2); ++key)
                {
                    out.at({batch, 0, query, key}) = mask.at({batch, query, key});
                }
            }
        }
        return out;
    }

    if (mask.rank() == 4)
    {
        const bool batch_ok = mask.dim(0) == 1 || mask.dim(0) == batch_size;
        const bool query_ok = mask.dim(2) == 1 || mask.dim(2) == query_len;
        const bool key_ok   = mask.dim(3) == 1 || mask.dim(3) == key_len;

        if (!batch_ok || !query_ok || !key_ok)
        {
            throw std::invalid_argument("Rank-4 attention masks must broadcast to "
                                        "[batch, heads, query_len, key_len].");
        }

        return mask;
    }

    throw std::invalid_argument("Unsupported attention mask rank.");
}

bool MaskAllows(const Tensor& mask,
                std::int64_t  batch,
                std::int64_t  head,
                std::int64_t  query,
                std::int64_t  key)
{
    const std::int64_t batch_index = mask.dim(0) == 1 ? 0 : batch;
    const std::int64_t head_index  = mask.dim(1) == 1 ? 0 : head;
    const std::int64_t query_index = mask.dim(2) == 1 ? 0 : query;
    const std::int64_t key_index   = mask.dim(3) == 1 ? 0 : key;
    return mask.at({batch_index, head_index, query_index, key_index}) > 0.0F;
}

}  // namespace

DropPath::DropPath(float drop_prob)
    : drop_prob_(drop_prob)
{
    if (drop_prob < 0.0F || drop_prob > 1.0F)
    {
        throw std::invalid_argument("drop_prob must be in [0, 1].");
    }
}

Tensor DropPath::Forward(const Tensor& x) const
{
    return x;
}

float DropPath::drop_prob() const noexcept
{
    return drop_prob_;
}

Linear::Linear(std::int64_t input_dim,
               std::int64_t output_dim,
               bool         bias,
               std::string  name)
    : input_dim_(input_dim),
      output_dim_(output_dim),
      weight_({output_dim, input_dim}, 0.0F),
      name_(std::move(name))
{
    if (input_dim <= 0 || output_dim <= 0)
    {
        throw std::invalid_argument("Linear dimensions must be positive.");
    }

    if (bias)
    {
        bias_.emplace(std::vector<std::int64_t>{output_dim}, 0.0F);
    }
}

Tensor Linear::Forward(const Tensor& x) const
{
    ValidateLastDim(x, input_dim_, "Linear::Forward");

    Tensor out(detail::ReplaceLastDim(x.shape(), output_dim_), 0.0F);

    const std::size_t rows = x.numel() / static_cast<std::size_t>(input_dim_);
    for (std::size_t row = 0; row < rows; ++row)
    {
        const std::size_t input_offset  = row * static_cast<std::size_t>(input_dim_);
        const std::size_t output_offset = row * static_cast<std::size_t>(output_dim_);

        for (std::int64_t out_dim = 0; out_dim < output_dim_; ++out_dim)
        {
            float acc = bias_.has_value() ? bias_->flat(static_cast<std::size_t>(out_dim)) : 0.0F;

            for (std::int64_t in_dim = 0; in_dim < input_dim_; ++in_dim)
            {
                acc += x.flat(input_offset + static_cast<std::size_t>(in_dim))
                       * weight_.at({out_dim, in_dim});
            }

            out.flat(output_offset + static_cast<std::size_t>(out_dim)) = acc;
        }
    }

    return out;
}

void Linear::LoadParameters(const StateDict&   state_dict,
                            const std::string& prefix)
{
    const std::string stem       = name_.empty() ? "" : name_ + ".";
    const std::string weight_key = detail::JoinKey(prefix, stem + "weight");
    weight_ = detail::RequireTensor(state_dict, weight_key, {output_dim_, input_dim_});

    if (bias_.has_value())
    {
        const std::string bias_key = detail::JoinKey(prefix, stem + "bias");
        bias_ = detail::RequireTensor(state_dict, bias_key, {output_dim_});
    }
}

std::vector<TensorSpec> Linear::ParameterSpecs(const std::string& prefix) const
{
    const std::string stem = name_.empty() ? "" : name_ + ".";

    std::vector<TensorSpec> specs = {
        {detail::JoinKey(prefix, stem + "weight"), {output_dim_, input_dim_}},
    };

    if (bias_.has_value())
    {
        specs.push_back({detail::JoinKey(prefix, stem + "bias"), {output_dim_}});
    }

    return specs;
}

const Tensor& Linear::weight() const noexcept
{
    return weight_;
}

const std::optional<Tensor>& Linear::bias() const noexcept
{
    return bias_;
}

LayerNorm::LayerNorm(std::int64_t embed_dim,
                     float        eps)
    : embed_dim_(embed_dim),
      eps_(eps),
      weight_({embed_dim}, 1.0F),
      bias_({embed_dim}, 0.0F)
{
    if (embed_dim <= 0)
    {
        throw std::invalid_argument("LayerNorm embed_dim must be positive.");
    }
}

Tensor LayerNorm::Forward(const Tensor& x) const
{
    ValidateLastDim(x, embed_dim_, "LayerNorm::Forward");

    Tensor out(x.shape(), 0.0F);
    const std::size_t rows = x.numel() / static_cast<std::size_t>(embed_dim_);

    for (std::size_t row = 0; row < rows; ++row)
    {
        const std::size_t offset = row * static_cast<std::size_t>(embed_dim_);

        float mean = 0.0F;
        for (std::int64_t dim = 0; dim < embed_dim_; ++dim)
        {
            mean += x.flat(offset + static_cast<std::size_t>(dim));
        }
        mean /= static_cast<float>(embed_dim_);

        float variance = 0.0F;
        for (std::int64_t dim = 0; dim < embed_dim_; ++dim)
        {
            const float centered = x.flat(offset + static_cast<std::size_t>(dim)) - mean;
            variance += centered * centered;
        }
        variance /= static_cast<float>(embed_dim_);

        const float inv_std = 1.0F / std::sqrt(variance + eps_);
        for (std::int64_t dim = 0; dim < embed_dim_; ++dim)
        {
            const float normalized = (x.flat(offset + static_cast<std::size_t>(dim)) - mean) * inv_std;
            out.flat(offset + static_cast<std::size_t>(dim)) =
                normalized * weight_.flat(static_cast<std::size_t>(dim))
                + bias_.flat(static_cast<std::size_t>(dim));
        }
    }

    return out;
}

void LayerNorm::LoadParameters(const StateDict&   state_dict,
                               const std::string& prefix)
{
    weight_ = detail::RequireTensor(state_dict,
                                    detail::JoinKey(prefix, "weight"),
                                    {embed_dim_});
    bias_   = detail::RequireTensor(state_dict,
                                    detail::JoinKey(prefix, "bias"),
                                    {embed_dim_});
}

std::vector<TensorSpec> LayerNorm::ParameterSpecs(const std::string& prefix) const
{
    return {
        {detail::JoinKey(prefix, "weight"), {embed_dim_}},
        {detail::JoinKey(prefix, "bias"),   {embed_dim_}},
    };
}

RotaryEmbedding::RotaryEmbedding(std::int64_t head_dim,
                                 std::int64_t base,
                                 std::int64_t max_seq_len)
    : head_dim_(head_dim),
      base_(base),
      max_seq_len_(max_seq_len)
{
    if (head_dim <= 0 || head_dim % 2 != 0)
    {
        throw std::invalid_argument("RoPE requires a positive even head_dim.");
    }
    if (max_seq_len <= 0)
    {
        throw std::invalid_argument("max_seq_len must be positive.");
    }

    const std::int64_t half_dim = head_dim_ / 2;
    inv_freq_.reserve(static_cast<std::size_t>(half_dim));
    for (std::int64_t index = 0; index < half_dim; ++index)
    {
        const float exponent = static_cast<float>(2 * index) / static_cast<float>(head_dim_);
        inv_freq_.push_back(1.0F / std::pow(static_cast<float>(base_), exponent));
    }

    BuildCache(max_seq_len_);
}

std::pair<Tensor, Tensor> RotaryEmbedding::Forward(const Tensor& q,
                                                   const Tensor& k,
                                                   std::int64_t  position_offset)
{
    if (q.rank() != 4 || k.rank() != 4)
    {
        throw std::invalid_argument("RoPE expects rank-4 q and k tensors.");
    }
    if (q.dim(2) != k.dim(2))
    {
        throw std::invalid_argument("RoPE expects q and k to have matching seq_len.");
    }
    if (q.dim(3) != head_dim_ || k.dim(3) != head_dim_)
    {
        throw std::invalid_argument("RoPE head_dim does not match q/k tensors.");
    }

    const auto [cos, sin] = BuildCosSin(q.dim(2), position_offset);
    const Tensor rotated_q = RotateHalf(q);
    const Tensor rotated_k = RotateHalf(k);

    Tensor out_q(q.shape(), 0.0F);
    Tensor out_k(k.shape(), 0.0F);

    for (std::int64_t batch = 0; batch < q.dim(0); ++batch)
    {
        for (std::int64_t head = 0; head < q.dim(1); ++head)
        {
            for (std::int64_t seq = 0; seq < q.dim(2); ++seq)
            {
                for (std::int64_t dim = 0; dim < q.dim(3); ++dim)
                {
                    const float cos_v = cos.at({0, 0, seq, dim});
                    const float sin_v = sin.at({0, 0, seq, dim});

                    out_q.at({batch, head, seq, dim}) =
                        (q.at({batch, head, seq, dim}) * cos_v)
                        + (rotated_q.at({batch, head, seq, dim}) * sin_v);

                    out_k.at({batch, head, seq, dim}) =
                        (k.at({batch, head, seq, dim}) * cos_v)
                        + (rotated_k.at({batch, head, seq, dim}) * sin_v);
                }
            }
        }
    }

    return {out_q, out_k};
}

std::int64_t RotaryEmbedding::max_seq_len() const noexcept
{
    return max_seq_len_;
}

const Tensor& RotaryEmbedding::cos_cached() const noexcept
{
    return cos_cached_;
}

const Tensor& RotaryEmbedding::sin_cached() const noexcept
{
    return sin_cached_;
}

Tensor RotaryEmbedding::RotateHalf(const Tensor& x) const
{
    if (x.rank() == 0 || x.dim(x.rank() - 1) != head_dim_)
    {
        throw std::invalid_argument("RotateHalf expects the last dimension to equal head_dim.");
    }

    const std::int64_t half_dim = head_dim_ / 2;
    Tensor out(x.shape(), 0.0F);
    const std::size_t rows = x.numel() / static_cast<std::size_t>(head_dim_);

    for (std::size_t row = 0; row < rows; ++row)
    {
        const std::size_t offset = row * static_cast<std::size_t>(head_dim_);
        for (std::int64_t dim = 0; dim < half_dim; ++dim)
        {
            out.flat(offset + static_cast<std::size_t>(dim)) =
                -x.flat(offset + static_cast<std::size_t>(half_dim + dim));
            out.flat(offset + static_cast<std::size_t>(half_dim + dim)) =
                x.flat(offset + static_cast<std::size_t>(dim));
        }
    }

    return out;
}

void RotaryEmbedding::BuildCache(std::int64_t max_seq_len)
{
    cos_cached_.Resize({1, 1, max_seq_len, head_dim_}, 0.0F);
    sin_cached_.Resize({1, 1, max_seq_len, head_dim_}, 0.0F);

    const std::int64_t half_dim = head_dim_ / 2;
    for (std::int64_t position = 0; position < max_seq_len; ++position)
    {
        for (std::int64_t dim = 0; dim < half_dim; ++dim)
        {
            const float angle = static_cast<float>(position) * inv_freq_.at(static_cast<std::size_t>(dim));
            const float cos_v = std::cos(angle);
            const float sin_v = std::sin(angle);

            cos_cached_.at({0, 0, position, dim})            = cos_v;
            cos_cached_.at({0, 0, position, half_dim + dim}) = cos_v;
            sin_cached_.at({0, 0, position, dim})            = sin_v;
            sin_cached_.at({0, 0, position, half_dim + dim}) = sin_v;
        }
    }
}

void RotaryEmbedding::EnsureCacheCapacity(std::int64_t required_len)
{
    if (required_len <= max_seq_len_)
    {
        return;
    }

    max_seq_len_ = std::max(required_len, max_seq_len_ * 2);
    BuildCache(max_seq_len_);
}

std::pair<Tensor, Tensor> RotaryEmbedding::BuildCosSin(std::int64_t seq_len,
                                                       std::int64_t position_offset) const
{
    if (seq_len < 0 || position_offset < 0)
    {
        throw std::invalid_argument("RoPE seq_len and position_offset must be non-negative.");
    }

    const_cast<RotaryEmbedding*>(this)->EnsureCacheCapacity(position_offset + seq_len);

    Tensor cos({1, 1, seq_len, head_dim_}, 0.0F);
    Tensor sin({1, 1, seq_len, head_dim_}, 0.0F);

    for (std::int64_t seq = 0; seq < seq_len; ++seq)
    {
        for (std::int64_t dim = 0; dim < head_dim_; ++dim)
        {
            cos.at({0, 0, seq, dim}) = cos_cached_.at({0, 0, position_offset + seq, dim});
            sin.at({0, 0, seq, dim}) = sin_cached_.at({0, 0, position_offset + seq, dim});
        }
    }

    return {cos, sin};
}

MultiHeadSelfAttention::MultiHeadSelfAttention(std::int64_t embed_dim,
                                               std::int64_t num_heads,
                                               float        dropout,
                                               bool         flash_attention,
                                               bool         qkv_bias,
                                               bool         use_rope,
                                               std::int64_t rope_base)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      dropout_(dropout),
      flash_attention_(flash_attention),
      capture_attention_(false),
      capture_qkv_(false),
      w_q_(embed_dim, embed_dim, qkv_bias, "w_q"),
      w_k_(embed_dim, embed_dim, qkv_bias, "w_k"),
      w_v_(embed_dim, embed_dim, qkv_bias, "w_v"),
      w_o_(embed_dim, embed_dim, true, "w_o")
{
    if (embed_dim <= 0 || num_heads <= 0 || embed_dim % num_heads != 0)
    {
        throw std::invalid_argument("embed_dim must be divisible by num_heads.");
    }

    if (use_rope)
    {
        rope_.emplace(head_dim_, rope_base);
    }
}

void MultiHeadSelfAttention::SetTrace(bool enabled,
                                      bool capture_qkv)
{
    capture_attention_ = enabled;
    capture_qkv_       = enabled && capture_qkv;

    if (!enabled)
    {
        ClearTrace();
    }
}

void MultiHeadSelfAttention::ClearTrace()
{
    last_attention_weights_.reset();
    last_q_.reset();
    last_k_.reset();
    last_v_.reset();
}

AttentionResult MultiHeadSelfAttention::Forward(const Tensor&                      x,
                                                const std::optional<Tensor>&       mask,
                                                const std::optional<KeyValueCache>& past_kv,
                                                bool                               use_cache,
                                                bool                               is_causal,
                                                bool                               need_weights)
{
    Tensor q = SplitHeads(w_q_.Forward(x));
    Tensor k = SplitHeads(w_k_.Forward(x));
    Tensor v = SplitHeads(w_v_.Forward(x));

    const std::int64_t position_offset = past_kv.has_value() ? past_kv->key.dim(2) : 0;
    if (rope_.has_value())
    {
        auto rotated = rope_->Forward(q, k, position_offset);
        q            = std::move(rotated.first);
        k            = std::move(rotated.second);
    }

    if (past_kv.has_value())
    {
        k = ConcatSequence(past_kv->key, k);
        v = ConcatSequence(past_kv->value, v);
    }

    std::optional<Tensor> attention_mask = std::nullopt;
    if (mask.has_value())
    {
        attention_mask = NormalizeAttentionMask(*mask,
                                                x.dim(0),
                                                q.dim(2),
                                                k.dim(2));
    }

    const bool capture_attention = need_weights || capture_attention_;
    AttentionResult attention_result = ScaledDotProduct(q,
                                                        k,
                                                        v,
                                                        attention_mask,
                                                        is_causal,
                                                        capture_attention);

    if (capture_attention)
    {
        last_attention_weights_ = attention_result.attention_weights;
    }
    else
    {
        last_attention_weights_.reset();
    }

    if (capture_qkv_)
    {
        last_q_ = q;
        last_k_ = k;
        last_v_ = v;
    }
    else
    {
        last_q_.reset();
        last_k_.reset();
        last_v_.reset();
    }

    attention_result.output = w_o_.Forward(CombineHeads(attention_result.output));

    if (use_cache)
    {
        attention_result.cache = KeyValueCache{k, v};
    }
    else
    {
        attention_result.cache.reset();
    }

    return attention_result;
}

void MultiHeadSelfAttention::LoadParameters(const StateDict&   state_dict,
                                            const std::string& prefix)
{
    w_q_.LoadParameters(state_dict, prefix);
    w_k_.LoadParameters(state_dict, prefix);
    w_v_.LoadParameters(state_dict, prefix);
    w_o_.LoadParameters(state_dict, prefix);
}

std::vector<TensorSpec> MultiHeadSelfAttention::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs, w_q_.ParameterSpecs(prefix));
    detail::AppendSpecs(specs, w_k_.ParameterSpecs(prefix));
    detail::AppendSpecs(specs, w_v_.ParameterSpecs(prefix));
    detail::AppendSpecs(specs, w_o_.ParameterSpecs(prefix));
    return specs;
}

const std::optional<Tensor>& MultiHeadSelfAttention::last_attention_weights() const noexcept
{
    return last_attention_weights_;
}

const std::optional<Tensor>& MultiHeadSelfAttention::last_q() const noexcept
{
    return last_q_;
}

const std::optional<Tensor>& MultiHeadSelfAttention::last_k() const noexcept
{
    return last_k_;
}

const std::optional<Tensor>& MultiHeadSelfAttention::last_v() const noexcept
{
    return last_v_;
}

Tensor MultiHeadSelfAttention::SplitHeads(const Tensor& x) const
{
    if (x.rank() != 3 || x.dim(2) != embed_dim_)
    {
        throw std::invalid_argument("SplitHeads expects a [batch, seq, embed_dim] tensor.");
    }

    const auto batch_size = x.dim(0);
    const auto seq_len    = x.dim(1);
    Tensor out({batch_size, num_heads_, seq_len, head_dim_}, 0.0F);

    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t seq = 0; seq < seq_len; ++seq)
        {
            for (std::int64_t head = 0; head < num_heads_; ++head)
            {
                for (std::int64_t dim = 0; dim < head_dim_; ++dim)
                {
                    out.at({batch, head, seq, dim}) =
                        x.at({batch, seq, head * head_dim_ + dim});
                }
            }
        }
    }

    return out;
}

Tensor MultiHeadSelfAttention::CombineHeads(const Tensor& x) const
{
    if (x.rank() != 4
        || x.dim(1) != num_heads_
        || x.dim(3) != head_dim_)
    {
        throw std::invalid_argument("CombineHeads expects a [batch, heads, seq, head_dim] tensor.");
    }

    const auto batch_size = x.dim(0);
    const auto seq_len    = x.dim(2);
    Tensor out({batch_size, seq_len, embed_dim_}, 0.0F);

    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t seq = 0; seq < seq_len; ++seq)
        {
            for (std::int64_t head = 0; head < num_heads_; ++head)
            {
                for (std::int64_t dim = 0; dim < head_dim_; ++dim)
                {
                    out.at({batch, seq, head * head_dim_ + dim}) =
                        x.at({batch, head, seq, dim});
                }
            }
        }
    }

    return out;
}

AttentionResult MultiHeadSelfAttention::ScaledDotProduct(const Tensor&                q,
                                                         const Tensor&                k,
                                                         const Tensor&                v,
                                                         const std::optional<Tensor>& mask,
                                                         bool                         is_causal,
                                                         bool                         need_weights) const
{
    if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4)
    {
        throw std::invalid_argument("ScaledDotProduct expects rank-4 q/k/v tensors.");
    }

    const auto batch_size    = q.dim(0);
    const auto num_heads     = q.dim(1);
    const auto q_len         = q.dim(2);
    const auto k_len         = k.dim(2);
    const auto causal_offset = std::max<std::int64_t>(0, k_len - q_len);

    if (k.dim(0) != batch_size
        || v.dim(0) != batch_size
        || k.dim(1) != num_heads
        || v.dim(1) != num_heads
        || k.dim(3) != head_dim_
        || v.dim(3) != head_dim_
        || v.dim(2) != k_len)
    {
        throw std::invalid_argument("ScaledDotProduct received incompatible q/k/v shapes.");
    }

    AttentionResult result;
    result.output.Resize({batch_size, num_heads, q_len, head_dim_}, 0.0F);

    if (need_weights)
    {
        result.attention_weights.Resize({batch_size, num_heads, q_len, k_len}, 0.0F);
        result.has_attention_weights = true;
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(head_dim_));

    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t head = 0; head < num_heads; ++head)
        {
            for (std::int64_t query = 0; query < q_len; ++query)
            {
                std::vector<float> scores(static_cast<std::size_t>(k_len),
                                          -std::numeric_limits<float>::infinity());

                float max_score = -std::numeric_limits<float>::infinity();
                for (std::int64_t key = 0; key < k_len; ++key)
                {
                    bool allowed = true;

                    if (is_causal && key > causal_offset + query)
                    {
                        allowed = false;
                    }

                    if (allowed && mask.has_value())
                    {
                        allowed = MaskAllows(*mask, batch, head, query, key);
                    }

                    if (!allowed)
                    {
                        continue;
                    }

                    float dot = 0.0F;
                    for (std::int64_t dim = 0; dim < head_dim_; ++dim)
                    {
                        dot += q.at({batch, head, query, dim})
                               * k.at({batch, head, key, dim});
                    }

                    scores[static_cast<std::size_t>(key)] = dot * scale;
                    max_score = std::max(max_score, scores[static_cast<std::size_t>(key)]);
                }

                if (!std::isfinite(max_score))
                {
                    continue;
                }

                float sum = 0.0F;
                for (std::int64_t key = 0; key < k_len; ++key)
                {
                    const std::size_t index = static_cast<std::size_t>(key);
                    if (!std::isfinite(scores[index]))
                    {
                        continue;
                    }

                    scores[index] = std::exp(scores[index] - max_score);
                    sum += scores[index];
                }

                if (sum <= 0.0F)
                {
                    continue;
                }

                for (std::int64_t key = 0; key < k_len; ++key)
                {
                    const std::size_t index = static_cast<std::size_t>(key);
                    if (!std::isfinite(scores[index]))
                    {
                        scores[index] = 0.0F;
                    }
                    else
                    {
                        scores[index] /= sum;
                    }

                    if (need_weights)
                    {
                        result.attention_weights.at({batch, head, query, key}) = scores[index];
                    }

                    for (std::int64_t dim = 0; dim < head_dim_; ++dim)
                    {
                        result.output.at({batch, head, query, dim}) +=
                            scores[index] * v.at({batch, head, key, dim});
                    }
                }
            }
        }
    }

    return result;
}

FeedForward::FeedForward(std::int64_t   embed_dim,
                         std::int64_t   hidden_dim,
                         std::int64_t   output_dim,
                         ActivationType activation,
                         float          dropout,
                         bool           bias)
    : embed_dim_(embed_dim),
      hidden_dim_(hidden_dim),
      output_dim_(output_dim),
      activation_(activation),
      dropout_(dropout),
      fc1_(embed_dim, hidden_dim, bias, "fc1"),
      fc2_(hidden_dim, output_dim, bias, "fc2")
{
    if (embed_dim != output_dim)
    {
        throw std::invalid_argument("FeedForward expects input_dim and output_dim to match.");
    }
}

Tensor FeedForward::Forward(const Tensor& x) const
{
    return fc2_.Forward(ApplyActivation(fc1_.Forward(x), activation_));
}

void FeedForward::LoadParameters(const StateDict&   state_dict,
                                 const std::string& prefix)
{
    fc1_.LoadParameters(state_dict, prefix);
    fc2_.LoadParameters(state_dict, prefix);
}

std::vector<TensorSpec> FeedForward::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs, fc1_.ParameterSpecs(prefix));
    detail::AppendSpecs(specs, fc2_.ParameterSpecs(prefix));
    return specs;
}

ResidualAttentionBlock::ResidualAttentionBlock(std::int64_t                           embed_dim,
                                               std::shared_ptr<MultiHeadSelfAttention> module,
                                               float                                  dropout,
                                               bool                                   norm_first,
                                               float                                  layer_norm_eps,
                                               float                                  drop_path)
    : norm_first_(norm_first),
      dropout_(dropout),
      drop_path_(drop_path),
      norm_(embed_dim, layer_norm_eps),
      module_(std::move(module))
{
    if (!module_)
    {
        throw std::invalid_argument("ResidualAttentionBlock requires a non-null module.");
    }
}

AttentionResult ResidualAttentionBlock::Forward(const Tensor&                      x,
                                                const std::optional<Tensor>&       mask,
                                                const std::optional<KeyValueCache>& past_kv,
                                                bool                               use_cache,
                                                bool                               is_causal)
{
    AttentionResult inner = module_->Forward(norm_first_ ? norm_.Forward(x) : x,
                                             mask,
                                             past_kv,
                                             use_cache,
                                             is_causal,
                                             false);

    Tensor output = Add(x, drop_path_.Forward(inner.output));
    if (!norm_first_)
    {
        output = norm_.Forward(output);
    }

    inner.output = std::move(output);
    return inner;
}

void ResidualAttentionBlock::LoadParameters(const StateDict&   state_dict,
                                            const std::string& prefix)
{
    module_->LoadParameters(state_dict, detail::JoinKey(prefix, "module."));
    norm_.LoadParameters(state_dict, detail::JoinKey(prefix, "norm."));
}

std::vector<TensorSpec> ResidualAttentionBlock::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs, module_->ParameterSpecs(detail::JoinKey(prefix, "module.")));
    detail::AppendSpecs(specs, norm_.ParameterSpecs(detail::JoinKey(prefix, "norm.")));
    return specs;
}

ResidualFeedForwardBlock::ResidualFeedForwardBlock(std::int64_t                 embed_dim,
                                                   std::shared_ptr<FeedForward> module,
                                                   float                        dropout,
                                                   bool                         norm_first,
                                                   float                        layer_norm_eps,
                                                   float                        drop_path)
    : norm_first_(norm_first),
      dropout_(dropout),
      drop_path_(drop_path),
      norm_(embed_dim, layer_norm_eps),
      module_(std::move(module))
{
    if (!module_)
    {
        throw std::invalid_argument("ResidualFeedForwardBlock requires a non-null module.");
    }
}

Tensor ResidualFeedForwardBlock::Forward(const Tensor& x) const
{
    Tensor out = module_->Forward(norm_first_ ? norm_.Forward(x) : x);
    out        = Add(x, drop_path_.Forward(out));

    if (!norm_first_)
    {
        out = norm_.Forward(out);
    }

    return out;
}

void ResidualFeedForwardBlock::LoadParameters(const StateDict&   state_dict,
                                              const std::string& prefix)
{
    module_->LoadParameters(state_dict, detail::JoinKey(prefix, "module."));
    norm_.LoadParameters(state_dict, detail::JoinKey(prefix, "norm."));
}

std::vector<TensorSpec> ResidualFeedForwardBlock::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs, module_->ParameterSpecs(detail::JoinKey(prefix, "module.")));
    detail::AppendSpecs(specs, norm_.ParameterSpecs(detail::JoinKey(prefix, "norm.")));
    return specs;
}

}  // namespace inference::transformer_core
