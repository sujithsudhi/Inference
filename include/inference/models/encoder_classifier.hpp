#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "inference/transformer_core/text.hpp"

namespace inference::models
{

/// Configuration describing a token-embedding encoder classifier.
struct EncoderClassifierConfig
{
    std::int64_t               vocab_size        = 0;
    std::int64_t               max_length        = 0;
    std::int64_t               embed_dim         = 0;
    std::int64_t               depth             = 0;
    std::int64_t               num_heads         = 0;
    float                      mlp_ratio         = 4.0F;
    std::optional<std::int64_t> mlp_hidden_dim    = std::nullopt;
    transformer_core::ActivationType activation = transformer_core::ActivationType::Gelu;
    float                      dropout           = 0.0F;
    float                      attention_dropout = 0.0F;
    bool                       qkv_bias          = true;
    bool                       pre_norm          = true;
    float                      layer_norm_eps    = 1e-5F;
    float                      drop_path         = 0.0F;
    bool                       use_cls_token     = true;
    std::optional<std::int64_t> cls_head_dim      = std::nullopt;
    std::int64_t               num_outputs       = 1;
    std::string                pooling           = "cls";
    bool                       use_rope          = true;
    std::int64_t               rope_base         = 10000;
};

/// Transformer encoder classifier that mirrors the Python IMDB classifier layout.
class EncoderClassifier
{
public:
    /// Build the classifier from a resolved configuration.
    explicit EncoderClassifier(EncoderClassifierConfig config);

    /// Run the encoder stack and pooling path to produce one feature vector per sample.
    transformer_core::Tensor ForwardFeatures(const transformer_core::IndexTensor&         inputs,
                                             const std::optional<transformer_core::Tensor>& attention_mask = std::nullopt);

    /// Run the full classifier head and return logits shaped `[batch, num_outputs]`.
    transformer_core::Tensor Forward(const transformer_core::IndexTensor&         inputs,
                                     const std::optional<transformer_core::Tensor>& attention_mask = std::nullopt);

    /// Load all parameters from a PyTorch-style state dict.
    void LoadParameters(const transformer_core::StateDict& state_dict);

    /// Describe the exact PyTorch-style parameters expected by the model.
    std::vector<transformer_core::TensorSpec> ParameterSpecs() const;

    /// Access the resolved configuration used by the model.
    const EncoderClassifierConfig& config() const noexcept;

private:
    transformer_core::Tensor BuildTokenMask(const transformer_core::IndexTensor& inputs,
                                            const std::optional<transformer_core::Tensor>& attention_mask) const;

    transformer_core::Tensor PrependClsToken(const transformer_core::Tensor& x) const;

    transformer_core::Tensor PrependClsMask(const transformer_core::Tensor& mask) const;

    transformer_core::Tensor PoolSequence(const transformer_core::Tensor&                x,
                                          const std::optional<transformer_core::Tensor>& token_mask) const;

    EncoderClassifierConfig                          config_;
    transformer_core::TokenEmbedding                 token_embedding_;
    std::optional<transformer_core::PositionalEncoding> position_;
    std::vector<transformer_core::TransformerEncoderLayer> encoder_;
    transformer_core::LayerNorm                      norm_;
    std::optional<transformer_core::Linear>          head0_;
    transformer_core::Linear                         output_head_;
    transformer_core::Tensor                         cls_token_;
};

}  // namespace inference::models
