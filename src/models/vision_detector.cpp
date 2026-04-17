#include "inference/models/vision_detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace inference::models
{

namespace
{

using inference::transformer_core::ActivationType;
using inference::transformer_core::StateDict;
using inference::transformer_core::Tensor;
using inference::transformer_core::TensorSpec;

void AppendSpecs(std::vector<TensorSpec>&       dst,
                 const std::vector<TensorSpec>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

bool ShapesEqual(const std::vector<std::int64_t>& lhs,
                 const std::vector<std::int64_t>& rhs)
{
    return lhs == rhs;
}

const Tensor& RequireTensor(const StateDict&                 state_dict,
                            const std::string&               key,
                            const std::vector<std::int64_t>& expected_shape = {})
{
    const auto it = state_dict.find(key);
    if (it == state_dict.end())
    {
        throw std::invalid_argument("Missing tensor '" + key + "' in state_dict.");
    }

    if (!expected_shape.empty() && !ShapesEqual(it->second.shape(), expected_shape))
    {
        throw std::invalid_argument("Tensor shape mismatch for '" + key + "'.");
    }

    return it->second;
}

Tensor Add(const Tensor& lhs,
           const Tensor& rhs)
{
    if (!ShapesEqual(lhs.shape(), rhs.shape()))
    {
        throw std::invalid_argument("Tensor add expects matching shapes.");
    }

    Tensor out(lhs.shape(), 0.0F);
    for (std::size_t index = 0; index < lhs.numel(); ++index)
    {
        out.flat(index) = lhs.flat(index) + rhs.flat(index);
    }
    return out;
}

Tensor ApplyActivation(const Tensor& x,
                       ActivationType activation)
{
    Tensor out(x.shape(), 0.0F);
    for (std::size_t index = 0; index < x.numel(); ++index)
    {
        const float value = x.flat(index);
        switch (activation)
        {
            case ActivationType::Identity:
                out.flat(index) = value;
                break;
            case ActivationType::Gelu:
                out.flat(index) = 0.5F * value * (1.0F + std::erf(value / std::sqrt(2.0F)));
                break;
            case ActivationType::Relu:
                out.flat(index) = std::max(0.0F, value);
                break;
            case ActivationType::Silu:
                out.flat(index) = value / (1.0F + std::exp(-value));
                break;
        }
    }

    return out;
}

Tensor ApplySigmoid(const Tensor& x)
{
    Tensor out(x.shape(), 0.0F);
    for (std::size_t index = 0; index < x.numel(); ++index)
    {
        out.flat(index) = 1.0F / (1.0F + std::exp(-x.flat(index)));
    }
    return out;
}

Tensor SplitHeads(const Tensor& x,
                  std::int64_t  num_heads)
{
    if (x.rank() != 3)
    {
        throw std::invalid_argument("SplitHeads expects a [batch, seq, embed_dim] tensor.");
    }

    const std::int64_t embed_dim = x.dim(2);
    if (num_heads <= 0 || embed_dim % num_heads != 0)
    {
        throw std::invalid_argument("embed_dim must be divisible by num_heads.");
    }

    const std::int64_t head_dim = embed_dim / num_heads;
    Tensor             out({x.dim(0), num_heads, x.dim(1), head_dim}, 0.0F);

    for (std::int64_t batch = 0; batch < x.dim(0); ++batch)
    {
        for (std::int64_t seq = 0; seq < x.dim(1); ++seq)
        {
            for (std::int64_t head = 0; head < num_heads; ++head)
            {
                for (std::int64_t dim = 0; dim < head_dim; ++dim)
                {
                    out.at({batch, head, seq, dim}) =
                        x.at({batch, seq, (head * head_dim) + dim});
                }
            }
        }
    }

    return out;
}

Tensor CombineHeads(const Tensor& x)
{
    if (x.rank() != 4)
    {
        throw std::invalid_argument("CombineHeads expects a [batch, heads, seq, head_dim] tensor.");
    }

    const std::int64_t embed_dim = x.dim(1) * x.dim(3);
    Tensor             out({x.dim(0), x.dim(2), embed_dim}, 0.0F);

    for (std::int64_t batch = 0; batch < x.dim(0); ++batch)
    {
        for (std::int64_t seq = 0; seq < x.dim(2); ++seq)
        {
            for (std::int64_t head = 0; head < x.dim(1); ++head)
            {
                for (std::int64_t dim = 0; dim < x.dim(3); ++dim)
                {
                    out.at({batch, seq, (head * x.dim(3)) + dim}) =
                        x.at({batch, head, seq, dim});
                }
            }
        }
    }

    return out;
}

Tensor ScaledDotProduct(const Tensor& q,
                        const Tensor& k,
                        const Tensor& v)
{
    if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4)
    {
        throw std::invalid_argument("ScaledDotProduct expects rank-4 q/k/v tensors.");
    }

    if (q.dim(0) != k.dim(0)
        || q.dim(0) != v.dim(0)
        || q.dim(1) != k.dim(1)
        || q.dim(1) != v.dim(1)
        || q.dim(3) != k.dim(3)
        || q.dim(3) != v.dim(3)
        || k.dim(2) != v.dim(2))
    {
        throw std::invalid_argument("ScaledDotProduct received incompatible q/k/v shapes.");
    }

    const std::int64_t batch_size = q.dim(0);
    const std::int64_t num_heads  = q.dim(1);
    const std::int64_t query_len  = q.dim(2);
    const std::int64_t key_len    = k.dim(2);
    const std::int64_t head_dim   = q.dim(3);
    const float        scale      = 1.0F / std::sqrt(static_cast<float>(head_dim));

    Tensor out({batch_size, num_heads, query_len, head_dim}, 0.0F);

    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t head = 0; head < num_heads; ++head)
        {
            for (std::int64_t query = 0; query < query_len; ++query)
            {
                std::vector<float> scores(static_cast<std::size_t>(key_len),
                                          -std::numeric_limits<float>::infinity());

                float max_score = -std::numeric_limits<float>::infinity();
                for (std::int64_t key = 0; key < key_len; ++key)
                {
                    float dot = 0.0F;
                    for (std::int64_t dim = 0; dim < head_dim; ++dim)
                    {
                        dot += q.at({batch, head, query, dim})
                               * k.at({batch, head, key, dim});
                    }

                    scores[static_cast<std::size_t>(key)] = dot * scale;
                    max_score = std::max(max_score, scores[static_cast<std::size_t>(key)]);
                }

                float sum = 0.0F;
                for (std::int64_t key = 0; key < key_len; ++key)
                {
                    const std::size_t index = static_cast<std::size_t>(key);
                    scores[index] = std::exp(scores[index] - max_score);
                    sum += scores[index];
                }

                if (sum <= 0.0F)
                {
                    continue;
                }

                for (std::int64_t key = 0; key < key_len; ++key)
                {
                    const float weight = scores[static_cast<std::size_t>(key)] / sum;
                    for (std::int64_t dim = 0; dim < head_dim; ++dim)
                    {
                        out.at({batch, head, query, dim}) +=
                            weight * v.at({batch, head, key, dim});
                    }
                }
            }
        }
    }

    return out;
}

Tensor SqueezeLastDim(const Tensor& x)
{
    if (x.rank() == 0 || x.dim(x.rank() - 1) != 1)
    {
        throw std::invalid_argument("SqueezeLastDim expects the last dimension to equal 1.");
    }

    std::vector<std::int64_t> shape = x.shape();
    shape.pop_back();

    Tensor out(shape, 0.0F);
    for (std::size_t index = 0; index < out.numel(); ++index)
    {
        out.flat(index) = x.flat(index);
    }

    return out;
}

Tensor DecodeBoxes(const Tensor& box_params)
{
    if (box_params.rank() != 3 || box_params.dim(2) != 4)
    {
        throw std::invalid_argument("DecodeBoxes expects a [batch, queries, 4] tensor.");
    }

    Tensor out(box_params.shape(), 0.0F);

    for (std::int64_t batch = 0; batch < box_params.dim(0); ++batch)
    {
        for (std::int64_t query = 0; query < box_params.dim(1); ++query)
        {
            const float center_x = box_params.at({batch, query, 0});
            const float center_y = box_params.at({batch, query, 1});
            const float width    = box_params.at({batch, query, 2});
            const float height   = box_params.at({batch, query, 3});
            const float half_w   = width * 0.5F;
            const float half_h   = height * 0.5F;

            out.at({batch, query, 0}) = std::clamp(center_x - half_w, 0.0F, 1.0F);
            out.at({batch, query, 1}) = std::clamp(center_y - half_h, 0.0F, 1.0F);
            out.at({batch, query, 2}) = std::clamp(center_x + half_w, 0.0F, 1.0F);
            out.at({batch, query, 3}) = std::clamp(center_y + half_h, 0.0F, 1.0F);
        }
    }

    return out;
}

}  // namespace

VisionDetector::VisionDetector(VisionDetectorConfig config)
    : config_(std::move(config)),
      patch_embed_(config_.backbone.image_size,
                   config_.backbone.patch_size,
                   config_.backbone.in_channels,
                   config_.backbone.embed_dim,
                   true),
      pos_embed_({1,
                  ((config_.backbone.image_size / config_.backbone.patch_size)
                   * (config_.backbone.image_size / config_.backbone.patch_size))
                  + (config_.backbone.use_cls_token ? 1 : 0),
                  config_.backbone.embed_dim},
                 0.0F),
      cls_token_(config_.backbone.use_cls_token
                     ? Tensor({1, 1, config_.backbone.embed_dim}, 0.0F)
                     : Tensor()),
      blocks_(),
      local_attention_(),
      backbone_norm_(config_.backbone.embed_dim,
                     config_.backbone.layer_norm_eps),
      query_embed_({1, config_.head.num_queries, config_.backbone.embed_dim}, 0.0F),
      query_norm_(config_.backbone.embed_dim, config_.head.layer_norm_eps),
      memory_norm_(config_.backbone.embed_dim, config_.head.layer_norm_eps),
      cross_attention_in_proj_weight_({config_.backbone.embed_dim * 3, config_.backbone.embed_dim}, 0.0F),
      cross_attention_in_proj_bias_({config_.backbone.embed_dim * 3}, 0.0F),
      cross_attention_out_proj_(config_.backbone.embed_dim,
                                config_.backbone.embed_dim,
                                true,
                                "out_proj"),
      ffn_norm_(config_.backbone.embed_dim, config_.head.layer_norm_eps),
      ffn_fc1_(config_.backbone.embed_dim,
               config_.head.mlp_hidden_dim,
               true,
               "0"),
      ffn_fc2_(config_.head.mlp_hidden_dim,
               config_.backbone.embed_dim,
               true,
               "3"),
      box_head_norm_(config_.backbone.embed_dim, config_.head.layer_norm_eps),
      box_head_fc1_(config_.backbone.embed_dim,
                    config_.head.mlp_hidden_dim,
                    true,
                    "1"),
      box_head_fc2_(config_.head.mlp_hidden_dim,
                    4,
                    true,
                    "3"),
      objectness_head_norm_(config_.backbone.embed_dim, config_.head.layer_norm_eps),
      objectness_head_fc_(config_.backbone.embed_dim,
                          1,
                          true,
                          "1"),
      class_head_norm_(config_.backbone.embed_dim, config_.head.layer_norm_eps),
      class_head_fc_(config_.backbone.embed_dim,
                     config_.head.num_classes,
                     true,
                     "1")
{
    if (config_.backbone.image_size <= 0
        || config_.backbone.patch_size <= 0
        || config_.backbone.in_channels <= 0
        || config_.backbone.embed_dim <= 0
        || config_.backbone.num_layers <= 0
        || config_.backbone.num_heads <= 0
        || config_.backbone.mlp_hidden_dim <= 0
        || config_.head.num_queries <= 0
        || config_.head.num_classes <= 0
        || config_.head.num_heads <= 0
        || config_.head.mlp_hidden_dim <= 0)
    {
        throw std::invalid_argument("VisionDetector requires positive model dimensions.");
    }

    if (config_.backbone.block_pattern.empty())
    {
        throw std::invalid_argument("VisionDetector requires at least one attention block type.");
    }

    blocks_.reserve(static_cast<std::size_t>(config_.backbone.num_layers));
    local_attention_.reserve(static_cast<std::size_t>(config_.backbone.num_layers));

    for (std::int64_t layer_idx = 0; layer_idx < config_.backbone.num_layers; ++layer_idx)
    {
        const std::string& block_type =
            config_.backbone.block_pattern[static_cast<std::size_t>(layer_idx)
                                           % config_.backbone.block_pattern.size()];

        const bool         is_local  = block_type == "local";
        const bool         is_global = block_type == "global";
        if (!is_local && !is_global)
        {
            throw std::invalid_argument("VisionDetector block_pattern must contain only 'local' or 'global'.");
        }

        const std::int64_t rope_base =
            is_local ? config_.backbone.local_rope_base : config_.backbone.global_rope_base;

        blocks_.emplace_back(config_.backbone.embed_dim,
                             config_.backbone.num_heads,
                             config_.backbone.mlp_ratio,
                             ActivationType::Gelu,
                             config_.backbone.attention_dropout,
                             config_.backbone.dropout,
                             true,
                             false,
                             config_.backbone.qkv_bias,
                             config_.backbone.use_rope,
                             rope_base,
                             config_.backbone.mlp_hidden_dim,
                             config_.backbone.layer_norm_eps,
                             0.0F);
        local_attention_.push_back(is_local);
    }
}

VisionBackboneOutput VisionDetector::ForwardBackbone(const Tensor& images)
{
    Tensor patches = patch_embed_.Forward(images);
    Tensor tokens  = config_.backbone.use_cls_token ? PrependClsToken(patches) : patches;
    tokens         = AddPositionEmbedding(tokens);

    const std::int64_t patch_tokens = patches.dim(1);
    const std::int64_t grid_side =
        static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(patch_tokens))));

    if (grid_side * grid_side != patch_tokens)
    {
        throw std::invalid_argument("Vision backbone expects a square patch grid.");
    }

    for (std::size_t layer_idx = 0; layer_idx < blocks_.size(); ++layer_idx)
    {
        if (local_attention_[layer_idx])
        {
            tokens = blocks_[layer_idx].Forward(tokens, BuildLocalAttentionMask(grid_side, grid_side));
        }
        else
        {
            tokens = blocks_[layer_idx].Forward(tokens);
        }
    }

    VisionBackboneOutput output;
    output.sequence_output = backbone_norm_.Forward(tokens);
    output.grid_height     = grid_side;
    output.grid_width      = grid_side;
    return output;
}

VisionDetectionOutput VisionDetector::Forward(const Tensor& images)
{
    VisionBackboneOutput backbone = ForwardBackbone(images);
    Tensor               memory   = memory_norm_.Forward(backbone.sequence_output);
    Tensor               queries  = ExpandQueryEmbedding(images.dim(0));
    Tensor               attended = CrossAttention(query_norm_.Forward(queries), memory);
    Tensor               query_features = Add(queries, attended);

    Tensor ffn_hidden = ffn_fc1_.Forward(ffn_norm_.Forward(query_features));
    Tensor ffn_output = ffn_fc2_.Forward(ApplyActivation(ffn_hidden, ActivationType::Gelu));
    query_features    = Add(query_features, ffn_output);

    Tensor box_hidden = box_head_fc1_.Forward(box_head_norm_.Forward(query_features));
    Tensor box_params = ApplySigmoid(box_head_fc2_.Forward(ApplyActivation(box_hidden, ActivationType::Gelu)));

    VisionDetectionOutput output;
    output.query_features         = query_features;
    output.pred_boxes             = DecodeBoxes(box_params);
    output.pred_objectness_logits =
        SqueezeLastDim(objectness_head_fc_.Forward(objectness_head_norm_.Forward(query_features)));
    output.pred_class_logits      = class_head_fc_.Forward(class_head_norm_.Forward(query_features));
    return output;
}

void VisionDetector::LoadParameters(const StateDict& state_dict)
{
    patch_embed_.LoadParameters(state_dict, "backbone.patch_embed.");
    pos_embed_ = RequireTensor(state_dict,
                               "backbone.pos_embed",
                               {1,
                                ((config_.backbone.image_size / config_.backbone.patch_size)
                                 * (config_.backbone.image_size / config_.backbone.patch_size))
                                + (config_.backbone.use_cls_token ? 1 : 0),
                                config_.backbone.embed_dim});

    if (config_.backbone.use_cls_token)
    {
        cls_token_ = RequireTensor(state_dict,
                                   "backbone.cls_token",
                                   {1, 1, config_.backbone.embed_dim});
    }

    for (std::size_t layer_idx = 0; layer_idx < blocks_.size(); ++layer_idx)
    {
        blocks_[layer_idx].LoadParameters(state_dict,
                                          "backbone.blocks." + std::to_string(layer_idx) + ".");
    }

    backbone_norm_.LoadParameters(state_dict, "backbone.norm.");

    query_embed_ = RequireTensor(state_dict,
                                 "detection_head.query_embed",
                                 {1, config_.head.num_queries, config_.backbone.embed_dim});
    query_norm_.LoadParameters(state_dict, "detection_head.query_norm.");
    memory_norm_.LoadParameters(state_dict, "detection_head.memory_norm.");

    cross_attention_in_proj_weight_ =
        RequireTensor(state_dict,
                      "detection_head.cross_attention.in_proj_weight",
                      {config_.backbone.embed_dim * 3, config_.backbone.embed_dim});
    cross_attention_in_proj_bias_ =
        RequireTensor(state_dict,
                      "detection_head.cross_attention.in_proj_bias",
                      {config_.backbone.embed_dim * 3});
    cross_attention_out_proj_.LoadParameters(state_dict, "detection_head.cross_attention.");

    ffn_norm_.LoadParameters(state_dict, "detection_head.ffn_norm.");
    ffn_fc1_.LoadParameters(state_dict, "detection_head.ffn.");
    ffn_fc2_.LoadParameters(state_dict, "detection_head.ffn.");

    box_head_norm_.LoadParameters(state_dict, "detection_head.box_head.0.");
    box_head_fc1_.LoadParameters(state_dict, "detection_head.box_head.");
    box_head_fc2_.LoadParameters(state_dict, "detection_head.box_head.");

    objectness_head_norm_.LoadParameters(state_dict, "detection_head.objectness_head.0.");
    objectness_head_fc_.LoadParameters(state_dict, "detection_head.objectness_head.");

    class_head_norm_.LoadParameters(state_dict, "detection_head.class_head.0.");
    class_head_fc_.LoadParameters(state_dict, "detection_head.class_head.");
}

std::vector<TensorSpec> VisionDetector::ParameterSpecs() const
{
    std::vector<TensorSpec> specs;

    AppendSpecs(specs, patch_embed_.ParameterSpecs("backbone.patch_embed."));
    specs.push_back({"backbone.pos_embed", pos_embed_.shape()});

    if (config_.backbone.use_cls_token)
    {
        specs.push_back({"backbone.cls_token", {1, 1, config_.backbone.embed_dim}});
    }

    for (std::size_t layer_idx = 0; layer_idx < blocks_.size(); ++layer_idx)
    {
        AppendSpecs(specs,
                    blocks_[layer_idx].ParameterSpecs("backbone.blocks."
                                                      + std::to_string(layer_idx)
                                                      + "."));
    }

    AppendSpecs(specs, backbone_norm_.ParameterSpecs("backbone.norm."));

    specs.push_back({"detection_head.query_embed", {1, config_.head.num_queries, config_.backbone.embed_dim}});
    AppendSpecs(specs, query_norm_.ParameterSpecs("detection_head.query_norm."));
    AppendSpecs(specs, memory_norm_.ParameterSpecs("detection_head.memory_norm."));
    specs.push_back({"detection_head.cross_attention.in_proj_weight",
                     {config_.backbone.embed_dim * 3, config_.backbone.embed_dim}});
    specs.push_back({"detection_head.cross_attention.in_proj_bias",
                     {config_.backbone.embed_dim * 3}});
    AppendSpecs(specs, cross_attention_out_proj_.ParameterSpecs("detection_head.cross_attention."));
    AppendSpecs(specs, ffn_norm_.ParameterSpecs("detection_head.ffn_norm."));
    AppendSpecs(specs, ffn_fc1_.ParameterSpecs("detection_head.ffn."));
    AppendSpecs(specs, ffn_fc2_.ParameterSpecs("detection_head.ffn."));
    AppendSpecs(specs, box_head_norm_.ParameterSpecs("detection_head.box_head.0."));
    AppendSpecs(specs, box_head_fc1_.ParameterSpecs("detection_head.box_head."));
    AppendSpecs(specs, box_head_fc2_.ParameterSpecs("detection_head.box_head."));
    AppendSpecs(specs, objectness_head_norm_.ParameterSpecs("detection_head.objectness_head.0."));
    AppendSpecs(specs, objectness_head_fc_.ParameterSpecs("detection_head.objectness_head."));
    AppendSpecs(specs, class_head_norm_.ParameterSpecs("detection_head.class_head.0."));
    AppendSpecs(specs, class_head_fc_.ParameterSpecs("detection_head.class_head."));
    return specs;
}

const VisionDetectorConfig& VisionDetector::config() const noexcept
{
    return config_;
}

Tensor VisionDetector::BuildLocalAttentionMask(std::int64_t grid_height,
                                               std::int64_t grid_width) const
{
    const std::int64_t patch_count   = grid_height * grid_width;
    const std::int64_t window_radius = config_.backbone.local_window_size / 2;
    const std::int64_t total_tokens  = patch_count + (config_.backbone.use_cls_token ? 1 : 0);

    Tensor mask({1, 1, total_tokens, total_tokens}, 0.0F);

    if (config_.backbone.use_cls_token)
    {
        for (std::int64_t index = 0; index < total_tokens; ++index)
        {
            mask.at({0, 0, 0, index}) = 1.0F;
            mask.at({0, 0, index, 0}) = 1.0F;
        }
    }

    for (std::int64_t lhs = 0; lhs < patch_count; ++lhs)
    {
        const std::int64_t lhs_row = lhs / grid_width;
        const std::int64_t lhs_col = lhs % grid_width;

        for (std::int64_t rhs = 0; rhs < patch_count; ++rhs)
        {
            const std::int64_t rhs_row = rhs / grid_width;
            const std::int64_t rhs_col = rhs % grid_width;
            const bool         allowed =
                std::abs(lhs_row - rhs_row) <= window_radius
                && std::abs(lhs_col - rhs_col) <= window_radius;

            if (!allowed)
            {
                continue;
            }

            const std::int64_t row = lhs + (config_.backbone.use_cls_token ? 1 : 0);
            const std::int64_t col = rhs + (config_.backbone.use_cls_token ? 1 : 0);
            mask.at({0, 0, row, col}) = 1.0F;
        }
    }

    return mask;
}

Tensor VisionDetector::AddPositionEmbedding(const Tensor& tokens) const
{
    if (tokens.rank() != 3 || tokens.dim(2) != config_.backbone.embed_dim)
    {
        throw std::invalid_argument("AddPositionEmbedding expects [batch, seq, embed_dim] inputs.");
    }

    if (tokens.dim(1) > pos_embed_.dim(1))
    {
        throw std::invalid_argument("Sequence length exceeds configured position embeddings.");
    }

    Tensor out(tokens.shape(), 0.0F);
    for (std::int64_t batch = 0; batch < tokens.dim(0); ++batch)
    {
        for (std::int64_t seq = 0; seq < tokens.dim(1); ++seq)
        {
            for (std::int64_t dim = 0; dim < tokens.dim(2); ++dim)
            {
                out.at({batch, seq, dim}) =
                    tokens.at({batch, seq, dim}) + pos_embed_.at({0, seq, dim});
            }
        }
    }

    return out;
}

Tensor VisionDetector::PrependClsToken(const Tensor& patches) const
{
    if (patches.rank() != 3 || patches.dim(2) != config_.backbone.embed_dim)
    {
        throw std::invalid_argument("PrependClsToken expects [batch, patches, embed_dim] inputs.");
    }

    Tensor out({patches.dim(0), patches.dim(1) + 1, patches.dim(2)}, 0.0F);

    for (std::int64_t batch = 0; batch < patches.dim(0); ++batch)
    {
        for (std::int64_t dim = 0; dim < patches.dim(2); ++dim)
        {
            out.at({batch, 0, dim}) = cls_token_.at({0, 0, dim});
        }

        for (std::int64_t patch = 0; patch < patches.dim(1); ++patch)
        {
            for (std::int64_t dim = 0; dim < patches.dim(2); ++dim)
            {
                out.at({batch, patch + 1, dim}) = patches.at({batch, patch, dim});
            }
        }
    }

    return out;
}

Tensor VisionDetector::ExpandQueryEmbedding(std::int64_t batch_size) const
{
    Tensor out({batch_size, config_.head.num_queries, config_.backbone.embed_dim}, 0.0F);

    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t query = 0; query < config_.head.num_queries; ++query)
        {
            for (std::int64_t dim = 0; dim < config_.backbone.embed_dim; ++dim)
            {
                out.at({batch, query, dim}) = query_embed_.at({0, query, dim});
            }
        }
    }

    return out;
}

Tensor VisionDetector::PackedProjection(const Tensor& x,
                                        std::int64_t  projection_index) const
{
    if (projection_index < 0 || projection_index > 2)
    {
        throw std::invalid_argument("projection_index must be 0, 1, or 2.");
    }

    if (x.rank() != 3 || x.dim(2) != config_.backbone.embed_dim)
    {
        throw std::invalid_argument("PackedProjection expects [batch, seq, embed_dim] inputs.");
    }

    const std::int64_t embed_dim = config_.backbone.embed_dim;
    Tensor             out(x.shape(), 0.0F);

    for (std::int64_t batch = 0; batch < x.dim(0); ++batch)
    {
        for (std::int64_t seq = 0; seq < x.dim(1); ++seq)
        {
            for (std::int64_t out_dim = 0; out_dim < embed_dim; ++out_dim)
            {
                const std::int64_t weight_row = (projection_index * embed_dim) + out_dim;
                float              acc = cross_attention_in_proj_bias_.at({weight_row});

                for (std::int64_t in_dim = 0; in_dim < embed_dim; ++in_dim)
                {
                    acc += x.at({batch, seq, in_dim})
                           * cross_attention_in_proj_weight_.at({weight_row, in_dim});
                }

                out.at({batch, seq, out_dim}) = acc;
            }
        }
    }

    return out;
}

Tensor VisionDetector::CrossAttention(const Tensor& queries,
                                      const Tensor& memory) const
{
    Tensor q = SplitHeads(PackedProjection(queries, 0), config_.head.num_heads);
    Tensor k = SplitHeads(PackedProjection(memory, 1), config_.head.num_heads);
    Tensor v = SplitHeads(PackedProjection(memory, 2), config_.head.num_heads);

    return cross_attention_out_proj_.Forward(CombineHeads(ScaledDotProduct(q, k, v)));
}

}  // namespace inference::models
