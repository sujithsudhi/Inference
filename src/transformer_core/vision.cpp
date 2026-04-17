/// \file
/// \brief Vision-oriented transformer-core layer and model implementation.

#include "inference/transformer_core/vision.hpp"

#include <stdexcept>

#include "detail.hpp"

namespace inference::transformer_core
{

namespace
{

bool HasAnyParameter(const StateDict&               state_dict,
                     const std::vector<TensorSpec>& specs)
{
    for (const auto& spec : specs)
    {
        if (detail::HasTensor(state_dict, spec.name))
        {
            return true;
        }
    }
    return false;
}

}  // namespace

PatchEmbedding::PatchEmbedding(std::int64_t image_size,
                               std::int64_t patch_size,
                               std::int64_t in_channels,
                               std::int64_t embed_dim,
                               bool         flatten)
    : image_size_(image_size),
      patch_size_(patch_size),
      in_channels_(in_channels),
      embed_dim_(embed_dim),
      grid_size_(image_size / patch_size),
      num_patches_(grid_size_ * grid_size_),
      flatten_(flatten),
      weight_({embed_dim, in_channels, patch_size, patch_size}, 0.0F),
      bias_({embed_dim}, 0.0F)
{
    if (image_size <= 0 || patch_size <= 0 || in_channels <= 0 || embed_dim <= 0)
    {
        throw std::invalid_argument("PatchEmbedding dimensions must be positive.");
    }
    if (image_size % patch_size != 0)
    {
        throw std::invalid_argument("image_size must be divisible by patch_size.");
    }
}

Tensor PatchEmbedding::Forward(const Tensor& x) const
{
    if (x.rank() != 4)
    {
        throw std::invalid_argument("PatchEmbedding expects [batch, channels, height, width] inputs.");
    }
    if (x.dim(1) != in_channels_ || x.dim(2) != image_size_ || x.dim(3) != image_size_)
    {
        throw std::invalid_argument("PatchEmbedding input shape does not match module configuration.");
    }

    const auto batch_size = x.dim(0);
    Tensor out = flatten_
        ? Tensor({batch_size, num_patches_, embed_dim_}, 0.0F)
        : Tensor({batch_size, embed_dim_, grid_size_, grid_size_}, 0.0F);

    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t out_channel = 0; out_channel < embed_dim_; ++out_channel)
        {
            for (std::int64_t grid_y = 0; grid_y < grid_size_; ++grid_y)
            {
                for (std::int64_t grid_x = 0; grid_x < grid_size_; ++grid_x)
                {
                    float acc = bias_.at({out_channel});

                    for (std::int64_t in_channel = 0; in_channel < in_channels_; ++in_channel)
                    {
                        for (std::int64_t kernel_y = 0; kernel_y < patch_size_; ++kernel_y)
                        {
                            for (std::int64_t kernel_x = 0; kernel_x < patch_size_; ++kernel_x)
                            {
                                const auto image_y = (grid_y * patch_size_) + kernel_y;
                                const auto image_x = (grid_x * patch_size_) + kernel_x;

                                acc += weight_.at({out_channel, in_channel, kernel_y, kernel_x})
                                       * x.at({batch, in_channel, image_y, image_x});
                            }
                        }
                    }

                    if (flatten_)
                    {
                        const auto patch_index = (grid_y * grid_size_) + grid_x;
                        out.at({batch, patch_index, out_channel}) = acc;
                    }
                    else
                    {
                        out.at({batch, out_channel, grid_y, grid_x}) = acc;
                    }
                }
            }
        }
    }

    return out;
}

void PatchEmbedding::LoadParameters(const StateDict&   state_dict,
                                    const std::string& prefix)
{
    weight_ = detail::RequireTensor(state_dict,
                                    detail::JoinKey(prefix, "proj.weight"),
                                    {embed_dim_, in_channels_, patch_size_, patch_size_});
    bias_   = detail::RequireTensor(state_dict,
                                    detail::JoinKey(prefix, "proj.bias"),
                                    {embed_dim_});
}

std::vector<TensorSpec> PatchEmbedding::ParameterSpecs(const std::string& prefix) const
{
    return {
        {detail::JoinKey(prefix, "proj.weight"), {embed_dim_, in_channels_, patch_size_, patch_size_}},
        {detail::JoinKey(prefix, "proj.bias"),   {embed_dim_}},
    };
}

ViTEncoderLayer::ViTEncoderLayer(std::int64_t   embed_dim,
                                 std::int64_t   num_heads,
                                 float          mlp_ratio,
                                 ActivationType activation,
                                 float          attention_dropout,
                                 float          dropout,
                                 bool           norm_first,
                                 bool           flash_attention)
    : attention_(std::make_shared<MultiHeadSelfAttention>(embed_dim,
                                                          num_heads,
                                                          attention_dropout,
                                                          flash_attention)),
      feed_forward_(std::make_shared<FeedForward>(embed_dim,
                                                 static_cast<std::int64_t>(embed_dim * mlp_ratio),
                                                 embed_dim,
                                                 activation,
                                                 dropout)),
      residual_attention_(embed_dim,
                          attention_,
                          dropout,
                          norm_first),
      residual_mlp_(embed_dim,
                    feed_forward_,
                    dropout,
                    norm_first)
{
}

Tensor ViTEncoderLayer::Forward(const Tensor&                x,
                                const std::optional<Tensor>& mask)
{
    AttentionResult attention_out = residual_attention_.Forward(x,
                                                                mask,
                                                                std::nullopt,
                                                                false,
                                                                false);
    return residual_mlp_.Forward(attention_out.output);
}

void ViTEncoderLayer::LoadParameters(const StateDict&   state_dict,
                                     const std::string& prefix)
{
    const auto attention_specs = attention_->ParameterSpecs(detail::JoinKey(prefix, "attention."));
    if (HasAnyParameter(state_dict, attention_specs))
    {
        attention_->LoadParameters(state_dict, detail::JoinKey(prefix, "attention."));
    }

    const auto mlp_specs = feed_forward_->ParameterSpecs(detail::JoinKey(prefix, "feed_forward."));
    if (HasAnyParameter(state_dict, mlp_specs))
    {
        feed_forward_->LoadParameters(state_dict, detail::JoinKey(prefix, "feed_forward."));
    }

    residual_attention_.LoadParameters(state_dict, detail::JoinKey(prefix, "residual_attention."));
    residual_mlp_.LoadParameters(state_dict, detail::JoinKey(prefix, "residual_mlp."));
}

std::vector<TensorSpec> ViTEncoderLayer::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs, attention_->ParameterSpecs(detail::JoinKey(prefix, "attention.")));
    detail::AppendSpecs(specs,
                        residual_attention_.ParameterSpecs(detail::JoinKey(prefix, "residual_attention.")));
    detail::AppendSpecs(specs, feed_forward_->ParameterSpecs(detail::JoinKey(prefix, "feed_forward.")));
    detail::AppendSpecs(specs,
                        residual_mlp_.ParameterSpecs(detail::JoinKey(prefix, "residual_mlp.")));
    return specs;
}

VisionTransformer::VisionTransformer(const VisionTransformerConfig& config)
    : patch_embed_(config.image_size, config.patch_size, config.in_channels, config.embed_dim, true),
      pos_embed_({1, (config.image_size / config.patch_size) * (config.image_size / config.patch_size) + 1, config.embed_dim}, 0.0F),
      cls_token_({1, 1, config.embed_dim}, 0.0F),
      layers_(),
      num_patches_((config.image_size / config.patch_size) * (config.image_size / config.patch_size))
{
    for (std::int64_t i = 0; i < config.num_layers; ++i)
    {
        layers_.emplace_back(config.embed_dim, config.num_heads, config.mlp_ratio);
    }
}

Tensor VisionTransformer::Forward(const Tensor& x)
{
    if (x.rank() != 4)
    {
        throw std::invalid_argument("VisionTransformer expects [batch, channels, height, width] inputs.");
    }

    Tensor patches = patch_embed_.Forward(x);
    const auto batch_size = patches.dim(0);
    Tensor tokens({batch_size, patches.dim(1) + 1, patches.dim(2)}, 0.0F);

    for (std::int64_t batch = 0; batch < batch_size; ++batch)
    {
        for (std::int64_t dim = 0; dim < patches.dim(2); ++dim)
        {
            tokens.at({batch, 0, dim}) =
                cls_token_.at({0, 0, dim}) + pos_embed_.at({0, 0, dim});
        }

        for (std::int64_t patch = 0; patch < patches.dim(1); ++patch)
        {
            for (std::int64_t dim = 0; dim < patches.dim(2); ++dim)
            {
                tokens.at({batch, patch + 1, dim}) =
                    patches.at({batch, patch, dim}) + pos_embed_.at({0, patch + 1, dim});
            }
        }
    }

    for (auto& layer : layers_)
    {
        tokens = layer.Forward(tokens);
    }

    const auto embed_dim = tokens.dim(2);
    Tensor cls_output({batch_size, embed_dim}, 0.0F);
    for (std::int64_t b = 0; b < batch_size; ++b)
    {
        for (std::int64_t d = 0; d < embed_dim; ++d)
        {
            cls_output.at({b, d}) = tokens.at({b, 0, d});
        }
    }

    return cls_output;
}

void VisionTransformer::LoadParameters(const StateDict&   state_dict,
                                       const std::string& prefix)
{
    patch_embed_.LoadParameters(state_dict, detail::JoinKey(prefix, "patch_embed."));
    const std::vector<std::int64_t> expected_pos_shape = pos_embed_.shape();
    const std::vector<std::int64_t> expected_cls_shape = cls_token_.shape();
    pos_embed_ = detail::RequireTensor(state_dict,
                                       detail::JoinKey(prefix, "pos_embed"),
                                       expected_pos_shape);
    cls_token_ = detail::RequireTensor(state_dict,
                                       detail::JoinKey(prefix, "cls_token"),
                                       expected_cls_shape);

    for (std::size_t i = 0; i < layers_.size(); ++i)
    {
        layers_[i].LoadParameters(state_dict, detail::JoinKey(detail::JoinKey(prefix, "blocks."), detail::JoinKey(std::to_string(i), ".")));
    }
}

std::vector<TensorSpec> VisionTransformer::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs, patch_embed_.ParameterSpecs(detail::JoinKey(prefix, "patch_embed.")));
    specs.push_back({detail::JoinKey(prefix, "pos_embed"), pos_embed_.shape()});
    specs.push_back({detail::JoinKey(prefix, "cls_token"), cls_token_.shape()});

    for (std::size_t i = 0; i < layers_.size(); ++i)
    {
        detail::AppendSpecs(specs, layers_[i].ParameterSpecs(detail::JoinKey(detail::JoinKey(prefix, "blocks."), detail::JoinKey(std::to_string(i), "."))));
    }
    return specs;
}

TextTransformer::TextTransformer(const TextTransformerConfig& config)
    : config_(config),
      token_embed_(config.vocab_size, config.embed_dim),
      pos_embed_(config.max_length, config.embed_dim, 0.0F, PositionalEncodingMethod::Trainable),
      layers_(),
      classifier_(config.embed_dim, config.num_classes)
{
    for (std::int64_t i = 0; i < config.depth; ++i)
    {
        layers_.emplace_back(config.embed_dim, config.num_heads, config.mlp_ratio);
    }
}

Tensor TextTransformer::Forward(const Tensor& input)
{
    if (input.rank() != 2)
    {
        throw std::invalid_argument("TextTransformer expects [batch, seq] floating-point token ids.");
    }

    const auto batch_size = input.dim(0);
    const auto seq_len = input.dim(1);
    IndexTensor token_ids({batch_size, seq_len}, 0);
    for (std::int64_t b = 0; b < batch_size; ++b)
    {
        for (std::int64_t s = 0; s < seq_len; ++s)
        {
            token_ids.at({b, s}) = static_cast<std::int64_t>(input.flat(b * seq_len + s));
        }
    }

    Tensor embeddings = token_embed_.Forward(token_ids);
    embeddings = pos_embed_.Forward(embeddings);
    for (auto& layer : layers_)
    {
        embeddings = layer.Forward(embeddings);
    }

    Tensor cls_token({batch_size, config_.embed_dim}, 0.0F);
    for (std::int64_t b = 0; b < batch_size; ++b)
    {
        for (std::int64_t d = 0; d < config_.embed_dim; ++d)
        {
            cls_token.flat(b * config_.embed_dim + d) = embeddings.flat(b * seq_len * config_.embed_dim + 0 * config_.embed_dim + d);
        }
    }

    return classifier_.Forward(cls_token);
}

void TextTransformer::LoadParameters(const StateDict&   state_dict,
                                     const std::string& prefix)
{
    token_embed_.LoadParameters(state_dict, detail::JoinKey(prefix, "token_embed."));
    pos_embed_.LoadParameters(state_dict, detail::JoinKey(prefix, "pos_embed."));
    classifier_.LoadParameters(state_dict, detail::JoinKey(prefix, "classifier."));

    for (std::size_t i = 0; i < layers_.size(); ++i)
    {
        layers_[i].LoadParameters(state_dict, detail::JoinKey(detail::JoinKey(prefix, "blocks."), detail::JoinKey(std::to_string(i), ".")));
    }
}

std::vector<TensorSpec> TextTransformer::ParameterSpecs(const std::string& prefix) const
{
    std::vector<TensorSpec> specs;
    detail::AppendSpecs(specs, token_embed_.ParameterSpecs(detail::JoinKey(prefix, "token_embed.")));
    detail::AppendSpecs(specs, pos_embed_.ParameterSpecs(detail::JoinKey(prefix, "pos_embed.")));
    detail::AppendSpecs(specs, classifier_.ParameterSpecs(detail::JoinKey(prefix, "classifier.")));

    for (std::size_t i = 0; i < layers_.size(); ++i)
    {
        detail::AppendSpecs(specs, layers_[i].ParameterSpecs(detail::JoinKey(detail::JoinKey(prefix, "blocks."), detail::JoinKey(std::to_string(i), "."))));
    }
    return specs;
}

}  // namespace inference::transformer_core
