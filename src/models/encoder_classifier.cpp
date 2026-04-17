/// \file
/// \brief Encoder-classifier runtime model implementation.

#include "inference/models/encoder_classifier.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "inference/transformer_core/common.hpp"

namespace inference::models
{

namespace
{

using inference::transformer_core::ActivationType;
using inference::transformer_core::IndexTensor;
using inference::transformer_core::PositionalEncoding;
using inference::transformer_core::PositionalEncodingMethod;
using inference::transformer_core::StateDict;
using inference::transformer_core::Tensor;
using inference::transformer_core::TensorSpec;
using inference::transformer_core::TransformerEncoderLayer;

Tensor Activate(const Tensor& x,
                ActivationType activation)
{
    Tensor out(x.shape(), 0.0F);
    for (std::size_t i = 0; i < x.numel(); ++i)
    {
        const float value = x.flat(i);
        switch (activation)
        {
            case ActivationType::Identity:
                out.flat(i) = value;
                break;
            case ActivationType::Gelu:
                out.flat(i) = 0.5F * value * (1.0F + std::erf(value / std::sqrt(2.0F)));
                break;
            case ActivationType::Relu:
                out.flat(i) = std::max(0.0F, value);
                break;
            case ActivationType::Silu:
                out.flat(i) = value / (1.0F + std::exp(-value));
                break;
        }
    }

    return out;
}

void AppendSpecs(std::vector<TensorSpec>&       dst,
                 const std::vector<TensorSpec>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

const Tensor& RequireTensor(const StateDict&   state_dict,
                            const std::string& key)
{
    const auto it = state_dict.find(key);
    if (it == state_dict.end())
    {
        throw std::invalid_argument("Missing tensor '" + key + "' in state_dict.");
    }
    return it->second;
}

bool HasTensor(const StateDict&   state_dict,
               const std::string& key)
{
    return state_dict.find(key) != state_dict.end();
}

}  // namespace

EncoderClassifier::EncoderClassifier(EncoderClassifierConfig config)
    : config_(std::move(config)),
      token_embedding_(config_.vocab_size, config_.embed_dim),
      norm_(config_.embed_dim, config_.layer_norm_eps),
      output_head_(config_.cls_head_dim.value_or(config_.embed_dim),
                   config_.num_outputs,
                   true,
                   config_.cls_head_dim.has_value() ? "head.3" : "head"),
      cls_token_({1, 1, config_.embed_dim}, 0.0F)
{
    if (config_.vocab_size <= 0
        || config_.max_length <= 0
        || config_.embed_dim <= 0
        || config_.depth <= 0
        || config_.num_heads <= 0
        || config_.num_outputs <= 0)
    {
        throw std::invalid_argument("EncoderClassifier requires positive model dimensions.");
    }

    if (!config_.use_rope)
    {
        position_.emplace(config_.max_length,
                          config_.embed_dim,
                          config_.dropout,
                          PositionalEncodingMethod::Trainable);
    }

    encoder_.reserve(static_cast<std::size_t>(config_.depth));
    for (std::int64_t index = 0; index < config_.depth; ++index)
    {
        encoder_.emplace_back(config_.embed_dim,
                              config_.num_heads,
                              config_.mlp_ratio,
                              config_.activation,
                              config_.attention_dropout,
                              config_.dropout,
                              config_.pre_norm,
                              false,
                              config_.qkv_bias,
                              config_.use_rope,
                              config_.rope_base,
                              config_.mlp_hidden_dim,
                              config_.layer_norm_eps,
                              config_.drop_path);
    }

    if (config_.cls_head_dim.has_value())
    {
        head0_.emplace(config_.embed_dim,
                       *config_.cls_head_dim,
                       true,
                       "head.0");
    }
}

Tensor EncoderClassifier::ForwardFeatures(const IndexTensor&          inputs,
                                          const std::optional<Tensor>& attention_mask)
{
    Tensor x = token_embedding_.Forward(inputs);
    std::optional<Tensor> token_mask = BuildTokenMask(inputs, attention_mask);

    if (config_.use_cls_token)
    {
        x = PrependClsToken(x);
        token_mask = PrependClsMask(*token_mask);
    }

    if (position_.has_value())
    {
        x = position_->Forward(x);
    }

    for (auto& layer : encoder_)
    {
        x = layer.Forward(x, token_mask);
    }

    x = norm_.Forward(x);
    return PoolSequence(x, token_mask);
}

Tensor EncoderClassifier::Forward(const IndexTensor&          inputs,
                                  const std::optional<Tensor>& attention_mask)
{
    Tensor features = ForwardFeatures(inputs, attention_mask);
    if (head0_.has_value())
    {
        features = Activate(head0_->Forward(features), config_.activation);
    }
    return output_head_.Forward(features);
}

void EncoderClassifier::LoadParameters(const StateDict& state_dict)
{
    StateDict token_embedding_state;
    token_embedding_state.emplace("embedding.weight", RequireTensor(state_dict, "token_embedding.weight"));
    token_embedding_.LoadParameters(token_embedding_state);

    if (config_.use_cls_token)
    {
        cls_token_ = RequireTensor(state_dict, "cls_token");
    }

    if (position_.has_value())
    {
        position_->LoadParameters(state_dict, "position.");
    }

    for (std::size_t index = 0; index < encoder_.size(); ++index)
    {
        encoder_[index].LoadParameters(state_dict, "encoder." + std::to_string(index) + ".");
    }

    norm_.LoadParameters(state_dict, "norm.");

    if (head0_.has_value())
    {
        head0_->LoadParameters(state_dict);
    }

    output_head_.LoadParameters(state_dict);
}

std::vector<TensorSpec> EncoderClassifier::ParameterSpecs() const
{
    std::vector<TensorSpec> specs;

    if (config_.use_cls_token)
    {
        specs.push_back({"cls_token", {1, 1, config_.embed_dim}});
    }

    specs.push_back({"token_embedding.weight", {config_.vocab_size, config_.embed_dim}});

    if (position_.has_value())
    {
        AppendSpecs(specs, position_->ParameterSpecs("position."));
    }

    for (std::size_t index = 0; index < encoder_.size(); ++index)
    {
        AppendSpecs(specs, encoder_[index].ParameterSpecs("encoder." + std::to_string(index) + "."));
    }

    AppendSpecs(specs, norm_.ParameterSpecs("norm."));

    if (head0_.has_value())
    {
        AppendSpecs(specs, head0_->ParameterSpecs());
    }

    AppendSpecs(specs, output_head_.ParameterSpecs());
    return specs;
}

const EncoderClassifierConfig& EncoderClassifier::config() const noexcept
{
    return config_;
}

Tensor EncoderClassifier::BuildTokenMask(const IndexTensor&          inputs,
                                         const std::optional<Tensor>& attention_mask) const
{
    if (attention_mask.has_value())
    {
        return *attention_mask;
    }

    Tensor mask(inputs.shape(), 0.0F);
    for (std::size_t i = 0; i < inputs.numel(); ++i)
    {
        mask.flat(i) = inputs.data()[i] != 0 ? 1.0F : 0.0F;
    }

    return mask;
}

Tensor EncoderClassifier::PrependClsToken(const Tensor& x) const
{
    if (x.rank() != 3 || x.dim(2) != config_.embed_dim)
    {
        throw std::invalid_argument("EncoderClassifier expects [batch, seq, embed_dim] token embeddings.");
    }

    Tensor out({x.dim(0), x.dim(1) + 1, x.dim(2)}, 0.0F);
    for (std::int64_t batch = 0; batch < x.dim(0); ++batch)
    {
        for (std::int64_t dim = 0; dim < x.dim(2); ++dim)
        {
            out.at({batch, 0, dim}) = cls_token_.at({0, 0, dim});
        }

        for (std::int64_t seq = 0; seq < x.dim(1); ++seq)
        {
            for (std::int64_t dim = 0; dim < x.dim(2); ++dim)
            {
                out.at({batch, seq + 1, dim}) = x.at({batch, seq, dim});
            }
        }
    }

    return out;
}

Tensor EncoderClassifier::PrependClsMask(const Tensor& mask) const
{
    if (mask.rank() != 2)
    {
        throw std::invalid_argument("EncoderClassifier token masks must be rank-2.");
    }

    Tensor out({mask.dim(0), mask.dim(1) + 1}, 1.0F);
    for (std::int64_t batch = 0; batch < mask.dim(0); ++batch)
    {
        for (std::int64_t seq = 0; seq < mask.dim(1); ++seq)
        {
            out.at({batch, seq + 1}) = mask.at({batch, seq});
        }
    }
    return out;
}

Tensor EncoderClassifier::PoolSequence(const Tensor&                 x,
                                       const std::optional<Tensor>& token_mask) const
{
    if (x.rank() != 3 || x.dim(2) != config_.embed_dim)
    {
        throw std::invalid_argument("PoolSequence expects [batch, seq, embed_dim] inputs.");
    }

    if (config_.pooling == "cls")
    {
        Tensor out({x.dim(0), x.dim(2)}, 0.0F);
        for (std::int64_t batch = 0; batch < x.dim(0); ++batch)
        {
            for (std::int64_t dim = 0; dim < x.dim(2); ++dim)
            {
                out.at({batch, dim}) = x.at({batch, 0, dim});
            }
        }
        return out;
    }

    const std::int64_t token_offset = config_.use_cls_token ? 1 : 0;
    Tensor out({x.dim(0), x.dim(2)}, 0.0F);

    for (std::int64_t batch = 0; batch < x.dim(0); ++batch)
    {
        float length = 0.0F;

        for (std::int64_t seq = token_offset; seq < x.dim(1); ++seq)
        {
            const float weight = token_mask.has_value() ? token_mask->at({batch, seq}) : 1.0F;
            if (weight <= 0.0F)
            {
                continue;
            }

            length += weight;
            for (std::int64_t dim = 0; dim < x.dim(2); ++dim)
            {
                out.at({batch, dim}) += x.at({batch, seq, dim}) * weight;
            }
        }

        const float denom = std::max(length, 1.0F);
        for (std::int64_t dim = 0; dim < x.dim(2); ++dim)
        {
            out.at({batch, dim}) /= denom;
        }
    }

    return out;
}

}  // namespace inference::models
