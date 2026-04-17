#pragma once

/// \file
/// \brief Encoder-classifier model and configuration types.

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
    /// Vocabulary size for the token embedding table.
    std::int64_t               vocab_size        = 0;
    /// Maximum supported token sequence length.
    std::int64_t               max_length        = 0;
    /// Embedding width shared across the encoder stack.
    std::int64_t               embed_dim         = 0;
    /// Number of encoder blocks.
    std::int64_t               depth             = 0;
    /// Number of attention heads per encoder block.
    std::int64_t               num_heads         = 0;
    /// Feed-forward expansion ratio when `mlp_hidden_dim` is not set explicitly.
    float                      mlp_ratio         = 4.0F;
    /// Optional explicit hidden width for the feed-forward sublayer.
    std::optional<std::int64_t> mlp_hidden_dim    = std::nullopt;
    /// Activation function used inside each feed-forward block.
    transformer_core::ActivationType activation = transformer_core::ActivationType::Gelu;
    /// Residual/dropout rate used by the classifier blocks.
    float                      dropout           = 0.0F;
    /// Dropout rate used inside self-attention.
    float                      attention_dropout = 0.0F;
    /// Whether the attention projections include bias terms.
    bool                       qkv_bias          = true;
    /// Whether the encoder uses pre-norm residual blocks.
    bool                       pre_norm          = true;
    /// Epsilon used by layer normalization.
    float                      layer_norm_eps    = 1e-5F;
    /// Stochastic-depth rate carried for parity with the Python model.
    float                      drop_path         = 0.0F;
    /// Whether a learned CLS token is prepended before encoding.
    bool                       use_cls_token     = true;
    /// Optional hidden width for an intermediate classifier head layer.
    std::optional<std::int64_t> cls_head_dim      = std::nullopt;
    /// Output-logit width.
    std::int64_t               num_outputs       = 1;
    /// Sequence pooling mode, currently `cls` or mean-style pooling.
    std::string                pooling           = "cls";
    /// Whether the encoder uses rotary position embeddings instead of a trainable table.
    bool                       use_rope          = true;
    /// Rotary-embedding base when `use_rope` is enabled.
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
