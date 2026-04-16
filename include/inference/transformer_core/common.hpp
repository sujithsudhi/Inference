#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "inference/transformer_core/tensor.hpp"

namespace inference::transformer_core
{

/// Activation functions supported by the lightweight transformer kernels.
enum class ActivationType
{
    Identity,
    Gelu,
    Relu,
    Silu,
};

/// Stochastic-depth placeholder used for API parity with the Python layers.
class DropPath
{
public:
    /// Construct a drop-path module with one drop probability.
    explicit DropPath(float drop_prob = 0.0F);

    /// Apply drop path to one residual branch tensor.
    Tensor Forward(const Tensor& x) const;

    /// Return the configured drop probability.
    float drop_prob() const noexcept;

private:
    float drop_prob_;
};

/// Fully connected projection with PyTorch-compatible parameter names and shapes.
class Linear
{
public:
    /// Construct one linear layer.
    Linear(std::int64_t  input_dim,
           std::int64_t  output_dim,
           bool          bias = true,
           std::string   name = "");

    /// Apply the linear projection to the last tensor dimension.
    Tensor Forward(const Tensor& x) const;

    /// Load parameters from a state dict, optionally under one prefix.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

    /// Return the projection weight tensor.
    const Tensor& weight() const noexcept;

    /// Return the optional bias tensor.
    const std::optional<Tensor>& bias() const noexcept;

private:
    std::int64_t           input_dim_;
    std::int64_t           output_dim_;
    std::optional<Tensor>  bias_;
    Tensor                 weight_;
    std::string            name_;
};

/// Layer normalization over the last tensor dimension.
class LayerNorm
{
public:
    /// Construct one layer-normalization module.
    LayerNorm(std::int64_t embed_dim,
              float        eps = 1e-5F);

    /// Normalize the last tensor dimension.
    Tensor Forward(const Tensor& x) const;

    /// Load the affine parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected affine parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    std::int64_t embed_dim_;
    float        eps_;
    Tensor       weight_;
    Tensor       bias_;
};

/// Rotary positional embedding cache used by attention query/key tensors.
class RotaryEmbedding
{
public:
    /// Construct one rotary-embedding helper.
    RotaryEmbedding(std::int64_t head_dim,
                    std::int64_t base        = 10000,
                    std::int64_t max_seq_len = 2048);

    /// Apply rotary embeddings to query and key tensors.
    std::pair<Tensor, Tensor> Forward(const Tensor& q,
                                      const Tensor& k,
                                      std::int64_t  position_offset = 0);

    /// Return the currently cached maximum sequence length.
    std::int64_t max_seq_len() const noexcept;

    /// Return the cached cosine table.
    const Tensor& cos_cached() const noexcept;

    /// Return the cached sine table.
    const Tensor& sin_cached() const noexcept;

private:
    Tensor RotateHalf(const Tensor& x) const;

    void BuildCache(std::int64_t max_seq_len);

    void EnsureCacheCapacity(std::int64_t required_len);

    std::pair<Tensor, Tensor> BuildCosSin(std::int64_t seq_len,
                                          std::int64_t position_offset) const;

    std::int64_t       head_dim_;
    std::int64_t       base_;
    std::int64_t       max_seq_len_;
    std::vector<float> inv_freq_;
    Tensor             cos_cached_;
    Tensor             sin_cached_;
};

/// Output bundle returned by self-attention forward passes.
struct AttentionResult
{
    Tensor                      output;
    std::optional<KeyValueCache> cache;
    Tensor                      attention_weights;
    bool                        has_attention_weights = false;
};

/// Multi-head self-attention with optional KV cache and trace capture.
class MultiHeadSelfAttention
{
public:
    /// Construct one self-attention module.
    MultiHeadSelfAttention(std::int64_t embed_dim,
                           std::int64_t num_heads,
                           float        dropout         = 0.0F,
                           bool         flash_attention = false,
                           bool         qkv_bias        = true,
                           bool         use_rope        = false,
                           std::int64_t rope_base       = 10000);

    /// Enable or disable capture of attention weights and optional Q/K/V tensors.
    void SetTrace(bool enabled = true,
                  bool capture_qkv = false);

    /// Clear any cached trace tensors from the previous forward pass.
    void ClearTrace();

    /// Run the self-attention layer.
    AttentionResult Forward(const Tensor&                     x,
                            const std::optional<Tensor>&      mask         = std::nullopt,
                            const std::optional<KeyValueCache>& past_kv      = std::nullopt,
                            bool                              use_cache    = false,
                            bool                              is_causal    = false,
                            bool                              need_weights = false);

    /// Load projection parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected projection parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

    /// Return the most recently captured attention weights, if tracing is enabled.
    const std::optional<Tensor>& last_attention_weights() const noexcept;
    /// Return the most recently captured query tensor, if tracing is enabled.
    const std::optional<Tensor>& last_q() const noexcept;
    /// Return the most recently captured key tensor, if tracing is enabled.
    const std::optional<Tensor>& last_k() const noexcept;
    /// Return the most recently captured value tensor, if tracing is enabled.
    const std::optional<Tensor>& last_v() const noexcept;

private:
    Tensor SplitHeads(const Tensor& x) const;
    Tensor CombineHeads(const Tensor& x) const;

    AttentionResult ScaledDotProduct(const Tensor&                q,
                                     const Tensor&                k,
                                     const Tensor&                v,
                                     const std::optional<Tensor>& mask,
                                     bool                         is_causal,
                                     bool                         need_weights) const;

    std::int64_t                 embed_dim_;
    std::int64_t                 num_heads_;
    std::int64_t                 head_dim_;
    float                        dropout_;
    bool                         flash_attention_;
    bool                         capture_attention_;
    bool                         capture_qkv_;
    Linear                       w_q_;
    Linear                       w_k_;
    Linear                       w_v_;
    Linear                       w_o_;
    std::optional<RotaryEmbedding> rope_;
    std::optional<Tensor>        last_attention_weights_;
    std::optional<Tensor>        last_q_;
    std::optional<Tensor>        last_k_;
    std::optional<Tensor>        last_v_;
};

/// Two-layer feed-forward network used inside transformer blocks.
class FeedForward
{
public:
    /// Construct one feed-forward block.
    FeedForward(std::int64_t embed_dim,
                std::int64_t hidden_dim,
                std::int64_t output_dim,
                ActivationType activation = ActivationType::Gelu,
                float          dropout    = 0.0F,
                bool           bias       = true);

    /// Run the feed-forward block on the input tensor.
    Tensor Forward(const Tensor& x) const;

    /// Load parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    std::int64_t   embed_dim_;
    std::int64_t   hidden_dim_;
    std::int64_t   output_dim_;
    ActivationType activation_;
    float          dropout_;
    Linear         fc1_;
    Linear         fc2_;
};

/// Residual wrapper around a self-attention module.
class ResidualAttentionBlock
{
public:
    /// Construct one residual attention block.
    ResidualAttentionBlock(std::int64_t                      embed_dim,
                           std::shared_ptr<MultiHeadSelfAttention> module,
                           float                             dropout        = 0.0F,
                           bool                              norm_first     = true,
                           float                             layer_norm_eps = 1e-5F,
                           float                             drop_path      = 0.0F);

    /// Run the residual attention block.
    AttentionResult Forward(const Tensor&                     x,
                            const std::optional<Tensor>&      mask         = std::nullopt,
                            const std::optional<KeyValueCache>& past_kv      = std::nullopt,
                            bool                              use_cache    = false,
                            bool                              is_causal    = false);

    /// Load parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    bool                              norm_first_;
    float                             dropout_;
    DropPath                          drop_path_;
    LayerNorm                         norm_;
    std::shared_ptr<MultiHeadSelfAttention> module_;
};

/// Residual wrapper around a feed-forward module.
class ResidualFeedForwardBlock
{
public:
    /// Construct one residual feed-forward block.
    ResidualFeedForwardBlock(std::int64_t               embed_dim,
                             std::shared_ptr<FeedForward> module,
                             float                      dropout        = 0.0F,
                             bool                       norm_first     = true,
                             float                      layer_norm_eps = 1e-5F,
                             float                      drop_path      = 0.0F);

    /// Run the residual feed-forward block.
    Tensor Forward(const Tensor& x) const;

    /// Load parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    bool                      norm_first_;
    float                     dropout_;
    DropPath                  drop_path_;
    LayerNorm                 norm_;
    std::shared_ptr<FeedForward> module_;
};

}  // namespace inference::transformer_core
