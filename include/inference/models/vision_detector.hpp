#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "inference/transformer_core/text.hpp"
#include "inference/transformer_core/vision.hpp"

namespace inference::models
{

/// Configuration describing the shared vision backbone used by the detector.
struct VisionBackboneConfig
{
    std::int64_t            image_size       = 0;
    std::int64_t            patch_size       = 0;
    std::int64_t            in_channels      = 0;
    std::int64_t            embed_dim        = 0;
    std::int64_t            num_layers       = 0;
    std::int64_t            num_heads        = 0;
    float                   mlp_ratio        = 4.0F;
    std::int64_t            mlp_hidden_dim   = 0;
    float                   attention_dropout = 0.0F;
    float                   dropout          = 0.0F;
    bool                    qkv_bias         = true;
    bool                    use_cls_token    = true;
    bool                    use_rope         = true;
    float                   layer_norm_eps   = 1e-6F;
    std::int64_t            local_window_size = 7;
    std::int64_t            local_rope_base  = 10000;
    std::int64_t            global_rope_base = 1000000;
    std::vector<std::string> block_pattern    = {"local", "local", "local", "global"};
};

/// Configuration describing the fixed-query detection head.
struct VisionDetectionHeadConfig
{
    std::int64_t num_queries      = 0;
    std::int64_t num_classes      = 0;
    std::int64_t num_heads        = 0;
    std::int64_t mlp_hidden_dim   = 0;
    float        dropout          = 0.0F;
    float        layer_norm_eps   = 1e-5F;
};

/// Configuration for the full detector model.
struct VisionDetectorConfig
{
    VisionBackboneConfig      backbone;
    VisionDetectionHeadConfig head;
};

/// Backbone outputs surfaced for debugging and downstream heads.
struct VisionBackboneOutput
{
    transformer_core::Tensor sequence_output;
    std::int64_t             grid_height = 0;
    std::int64_t             grid_width  = 0;
};

/// Raw detector outputs before any thresholding or NMS.
struct VisionDetectionOutput
{
    transformer_core::Tensor query_features;
    transformer_core::Tensor pred_boxes;
    transformer_core::Tensor pred_objectness_logits;
    transformer_core::Tensor pred_class_logits;
};

/// Vision detector that mirrors the VLM Python `VisionDetector` checkpoint layout.
class VisionDetector
{
public:
    /// Build the detector from one resolved configuration.
    explicit VisionDetector(VisionDetectorConfig config);

    /// Run only the shared vision backbone.
    VisionBackboneOutput ForwardBackbone(const transformer_core::Tensor& images);

    /// Run the full detector and return raw query-level predictions.
    VisionDetectionOutput Forward(const transformer_core::Tensor& images);

    /// Load all parameters from a PyTorch-style state dict.
    void LoadParameters(const transformer_core::StateDict& state_dict);

    /// Describe the exact PyTorch-style parameters expected by the detector.
    std::vector<transformer_core::TensorSpec> ParameterSpecs() const;

    /// Access the resolved detector configuration.
    const VisionDetectorConfig& config() const noexcept;

private:
    transformer_core::Tensor BuildLocalAttentionMask(std::int64_t grid_height,
                                                     std::int64_t grid_width) const;

    transformer_core::Tensor AddPositionEmbedding(const transformer_core::Tensor& tokens) const;

    transformer_core::Tensor PrependClsToken(const transformer_core::Tensor& patches) const;

    transformer_core::Tensor ExpandQueryEmbedding(std::int64_t batch_size) const;

    transformer_core::Tensor PackedProjection(const transformer_core::Tensor& x,
                                              std::int64_t                     projection_index) const;

    transformer_core::Tensor CrossAttention(const transformer_core::Tensor& queries,
                                            const transformer_core::Tensor& memory) const;

    VisionDetectorConfig                             config_;
    transformer_core::PatchEmbedding                 patch_embed_;
    transformer_core::Tensor                         pos_embed_;
    transformer_core::Tensor                         cls_token_;
    std::vector<transformer_core::TransformerEncoderLayer> blocks_;
    std::vector<bool>                                local_attention_;
    transformer_core::LayerNorm                      backbone_norm_;

    transformer_core::Tensor                         query_embed_;
    transformer_core::LayerNorm                      query_norm_;
    transformer_core::LayerNorm                      memory_norm_;
    transformer_core::Tensor                         cross_attention_in_proj_weight_;
    transformer_core::Tensor                         cross_attention_in_proj_bias_;
    transformer_core::Linear                         cross_attention_out_proj_;
    transformer_core::LayerNorm                      ffn_norm_;
    transformer_core::Linear                         ffn_fc1_;
    transformer_core::Linear                         ffn_fc2_;
    transformer_core::LayerNorm                      box_head_norm_;
    transformer_core::Linear                         box_head_fc1_;
    transformer_core::Linear                         box_head_fc2_;
    transformer_core::LayerNorm                      objectness_head_norm_;
    transformer_core::Linear                         objectness_head_fc_;
    transformer_core::LayerNorm                      class_head_norm_;
    transformer_core::Linear                         class_head_fc_;
};

}  // namespace inference::models
