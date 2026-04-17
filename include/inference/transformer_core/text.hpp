#pragma once

/// \file
/// \brief Text-oriented transformer-core layers and decoder helpers.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "inference/transformer_core/common.hpp"

namespace inference::transformer_core
{

/// Positional-encoding strategies supported by the text modules.
enum class PositionalEncodingMethod
{
    /// Deterministic sinusoidal position encoding.
    Normal,
    /// Trainable positional table loaded from the checkpoint.
    Trainable,
};

/// Token-id embedding layer matching the Python `TokenEmbedding` module.
class TokenEmbedding
{
public:
    /// Construct one token embedding table.
    TokenEmbedding(std::int64_t                vocab_size  = 256,
                   std::int64_t                embed_dim   = 256,
                   std::optional<std::int64_t> padding_idx = std::nullopt);

    /// Embed integer token ids into dense vectors.
    Tensor Forward(const IndexTensor& tokens) const;

    /// Load parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    std::int64_t                vocab_size_;
    std::int64_t                embed_dim_;
    std::optional<std::int64_t> padding_idx_;
    Tensor                      embedding_weight_;
};

/// Sinusoidal or trainable positional-encoding table.
class PositionalEncoding
{
public:
    /// Construct one positional-encoding table.
    PositionalEncoding(std::int64_t              max_len,
                       std::int64_t              embed_dim,
                       float                     dropout,
                       PositionalEncodingMethod  method = PositionalEncodingMethod::Normal);

    /// Add positional encodings to the input sequence.
    Tensor Forward(const Tensor& x,
                   std::int64_t  offset = 0) const;

    /// Load trainable positional parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    std::int64_t             max_len_;
    std::int64_t             embed_dim_;
    float                    dropout_;
    PositionalEncodingMethod method_;
    Tensor                   positional_table_;
};

/// Transformer encoder block matching the Python `TransformerEncoderLayer`.
class TransformerEncoderLayer
{
public:
    /// Construct one transformer encoder layer.
    TransformerEncoderLayer(std::int64_t embed_dim,
                            std::int64_t num_heads,
                            float        mlp_ratio         = 4.0F,
                            ActivationType activation      = ActivationType::Gelu,
                            float        attention_dropout = 0.0F,
                            float        dropout           = 0.0F,
                            bool         norm_first        = true,
                            bool         flash_attention   = false,
                            bool         qkv_bias          = true,
                            bool         use_rope          = false,
                            std::int64_t rope_base         = 10000,
                            std::optional<std::int64_t> mlp_hidden_dim = std::nullopt,
                            float        layer_norm_eps    = 1e-5F,
                            float        drop_path         = 0.0F);

    /// Run the encoder block.
    Tensor Forward(const Tensor&                x,
                   const std::optional<Tensor>& mask = std::nullopt);

    /// Load parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    std::shared_ptr<MultiHeadSelfAttention> attention_;
    std::shared_ptr<FeedForward>            feed_forward_;
    ResidualAttentionBlock                  residual_attention_;
    ResidualFeedForwardBlock                residual_mlp_;
};

/// Output bundle returned by decoder blocks when cache output is enabled.
struct DecoderResult
{
    /// Decoder output tensor for the current forward pass.
    Tensor                      output;
    /// Optional grown KV cache when cache output is enabled.
    std::optional<KeyValueCache> cache;
};

/// Transformer decoder block matching the Python `TransformerDecoderLayer`.
class TransformerDecoderLayer
{
public:
    /// Construct one transformer decoder layer.
    TransformerDecoderLayer(std::int64_t embed_dim,
                            std::int64_t num_heads,
                            float        mlp_ratio         = 4.0F,
                            ActivationType activation      = ActivationType::Gelu,
                            float        attention_dropout = 0.0F,
                            float        dropout           = 0.0F,
                            bool         norm_first        = true,
                            bool         flash_attention   = false,
                            bool         qkv_bias          = true,
                            bool         use_rope          = false,
                            std::int64_t rope_base         = 10000,
                            std::optional<std::int64_t> mlp_hidden_dim = std::nullopt,
                            float        layer_norm_eps    = 1e-5F,
                            float        drop_path         = 0.0F);

    /// Run the decoder block with optional KV cache support.
    DecoderResult Forward(const Tensor&                     x,
                          const std::optional<Tensor>&      mask      = std::nullopt,
                          const std::optional<KeyValueCache>& past_kv   = std::nullopt,
                          bool                              use_cache = false);

    /// Load parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    Tensor BuildCausalMask(const Tensor& x,
                           const Tensor& mask,
                           std::int64_t  past_len) const;

    std::shared_ptr<MultiHeadSelfAttention> attention_;
    std::shared_ptr<FeedForward>            feed_forward_;
    ResidualAttentionBlock                  residual_attention_;
    ResidualFeedForwardBlock                residual_mlp_;
};

}  // namespace inference::transformer_core
