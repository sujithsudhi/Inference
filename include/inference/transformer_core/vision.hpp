#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "inference/transformer_core/text.hpp"

namespace inference::transformer_core
{

/// Base class for transformer models.
class Model
{
public:
    virtual ~Model() = default;

    /// Run forward pass.
    virtual Tensor Forward(const Tensor& input) = 0;

    /// Load parameters from state dict.
    virtual void LoadParameters(const StateDict& state_dict, const std::string& prefix = "") = 0;

    /// Return parameter specs.
    virtual std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const = 0;
};

/// Strided-convolution patch embedding matching the Python `PatchEmbedding` module.
class PatchEmbedding
{
public:
    /// Construct one patch embedding layer.
    PatchEmbedding(std::int64_t image_size  = 224,
                   std::int64_t patch_size  = 16,
                   std::int64_t in_channels = 3,
                   std::int64_t embed_dim   = 768,
                   bool         flatten     = true);

    /// Project an image batch into patch tokens.
    Tensor Forward(const Tensor& x) const;

    /// Load parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    std::int64_t image_size_;
    std::int64_t patch_size_;
    std::int64_t in_channels_;
    std::int64_t embed_dim_;
    std::int64_t grid_size_;
    std::int64_t num_patches_;
    bool         flatten_;
    Tensor       weight_;
    Tensor       bias_;
};

/// Vision-transformer encoder block matching the Python `ViTEncoderLayer`.
class ViTEncoderLayer
{
public:
    /// Construct one ViT encoder block.
    ViTEncoderLayer(std::int64_t embed_dim,
                    std::int64_t num_heads,
                    float        mlp_ratio         = 4.0F,
                    ActivationType activation      = ActivationType::Gelu,
                    float        attention_dropout = 0.0F,
                    float        dropout           = 0.0F,
                    bool         norm_first        = true,
                    bool         flash_attention   = false);

    /// Run the ViT encoder block.
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

/// Configuration for VisionTransformer.
struct VisionTransformerConfig
{
    std::int64_t image_size = 224;
    std::int64_t patch_size = 16;
    std::int64_t in_channels = 3;
    std::int64_t embed_dim = 1024;
    std::int64_t num_layers = 12;
    std::int64_t num_heads = 16;
    float        mlp_ratio = 4.0F;
};

/// Vision Transformer (ViT) model.
class VisionTransformer : public Model
{
public:
    /// Construct a Vision Transformer.
    explicit VisionTransformer(const VisionTransformerConfig& config);

    /// Run the Vision Transformer.
    Tensor Forward(const Tensor& x);

    /// Load parameters from a state dict.
    void LoadParameters(const StateDict&   state_dict,
                        const std::string& prefix = "");

    /// Return the expected parameter names and shapes.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const;

private:
    PatchEmbedding                    patch_embed_;
    Tensor                            pos_embed_;
    Tensor                            cls_token_;
    std::vector<ViTEncoderLayer>      layers_;
    std::int64_t                      num_patches_;
};

/// Configuration for TextTransformer.
struct TextTransformerConfig
{
    std::int64_t vocab_size = 50257;  // GPT-2 vocab
    std::int64_t max_length = 1024;
    std::int64_t embed_dim = 768;
    std::int64_t depth = 12;
    std::int64_t num_heads = 12;
    float        mlp_ratio = 4.0F;
    std::int64_t num_classes = 2;  // For IMDB binary classification
};

inline void from_json(const nlohmann::json& j, VisionTransformerConfig& c)
{
    if (j.contains("image_size")) j.at("image_size").get_to(c.image_size);
    if (j.contains("patch_size")) j.at("patch_size").get_to(c.patch_size);
    if (j.contains("in_channels")) j.at("in_channels").get_to(c.in_channels);
    if (j.contains("embed_dim")) j.at("embed_dim").get_to(c.embed_dim);
    if (j.contains("num_layers")) j.at("num_layers").get_to(c.num_layers);
    if (j.contains("num_heads")) j.at("num_heads").get_to(c.num_heads);
    if (j.contains("mlp_ratio")) j.at("mlp_ratio").get_to(c.mlp_ratio);
}

inline void from_json(const nlohmann::json& j, TextTransformerConfig& c)
{
    if (j.contains("vocab_size")) j.at("vocab_size").get_to(c.vocab_size);
    if (j.contains("max_length")) j.at("max_length").get_to(c.max_length);
    if (j.contains("embed_dim")) j.at("embed_dim").get_to(c.embed_dim);
    if (j.contains("depth")) j.at("depth").get_to(c.depth);
    if (j.contains("num_heads")) j.at("num_heads").get_to(c.num_heads);
    if (j.contains("mlp_ratio")) j.at("mlp_ratio").get_to(c.mlp_ratio);
    if (j.contains("num_classes")) j.at("num_classes").get_to(c.num_classes);
}

/// Text Transformer (e.g., for sentiment classification).
class TextTransformer : public Model
{
public:
    /// Construct a Text Transformer.
    explicit TextTransformer(const TextTransformerConfig& config);

    /// Run the Text Transformer on token sequences.
    Tensor Forward(const Tensor& input) override;

    /// Load parameters from state dict.
    void LoadParameters(const StateDict& state_dict, const std::string& prefix = "") override;

    /// Return parameter specs.
    std::vector<TensorSpec> ParameterSpecs(const std::string& prefix = "") const override;

private:
    TextTransformerConfig config_;
    TokenEmbedding        token_embed_;
    PositionalEncoding    pos_embed_;
    std::vector<TransformerEncoderLayer> layers_;
    Linear                 classifier_;
};

}  // namespace inference::transformer_core
