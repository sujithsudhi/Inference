#include "inference/model_builder/model_builder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <stdexcept>

namespace inference::model_builder
{

namespace
{

using inference::transformer_core::ActivationType;
using inference::transformer_core::StateDict;
using inference::transformer_core::Tensor;

const Json& ResolveModelMetadata(const Json& metadata)
{
    const Json* resolved = nullptr;

    if (metadata.contains("config")
        && metadata["config"].is_object()
        && metadata["config"].contains("model")
        && metadata["config"]["model"].is_object())
    {
        resolved = &metadata["config"]["model"];
    }
    else if (metadata.contains("model") && metadata["model"].is_object())
    {
        resolved = &metadata["model"];
    }

    if (resolved == nullptr)
    {
        throw std::invalid_argument("Artifact metadata does not contain a model config.");
    }

    return *resolved;
}

const Json* FindModelMetadata(const Json& metadata)
{
    if (metadata.contains("config")
        && metadata["config"].is_object()
        && metadata["config"].contains("model")
        && metadata["config"]["model"].is_object())
    {
        return &metadata["config"]["model"];
    }

    if (metadata.contains("model") && metadata["model"].is_object())
    {
        return &metadata["model"];
    }

    return nullptr;
}

const Json* FindNestedObject(const Json* parent,
                             const char* key)
{
    if (parent == nullptr
        || !parent->contains(key)
        || !(*parent)[key].is_object())
    {
        return nullptr;
    }

    return &(*parent)[key];
}

const Tensor* FindTensor(const StateDict&   state_dict,
                         const std::string& key)
{
    const auto it = state_dict.find(key);
    if (it == state_dict.end())
    {
        return nullptr;
    }
    return &it->second;
}

const Tensor& RequireTensor(const StateDict&   state_dict,
                            const std::string& key)
{
    const Tensor* tensor = FindTensor(state_dict, key);
    if (tensor == nullptr)
    {
        throw std::invalid_argument("Missing tensor '" + key + "' in state_dict.");
    }
    return *tensor;
}

std::int64_t CountIndexedPrefix(const StateDict&   state_dict,
                                const std::string& prefix)
{
    std::set<std::int64_t> indices;

    for (const auto& kv : state_dict)
    {
        if (kv.first.rfind(prefix, 0) != 0)
        {
            continue;
        }

        const auto suffix = kv.first.substr(prefix.size());
        const auto dot    = suffix.find('.');
        if (dot == std::string::npos)
        {
            continue;
        }

        const auto index_str = suffix.substr(0, dot);
        if (index_str.empty()
            || !std::all_of(index_str.begin(),
                            index_str.end(),
                            [](unsigned char ch) { return std::isdigit(ch) != 0; }))
        {
            continue;
        }

        indices.insert(std::stoll(index_str));
    }

    return static_cast<std::int64_t>(indices.size());
}

ActivationType ResolveActivation(const Json& model_cfg)
{
    const std::string activation = model_cfg.value("activation", std::string("gelu"));

    if (activation == "identity")
    {
        return ActivationType::Identity;
    }
    if (activation == "relu")
    {
        return ActivationType::Relu;
    }
    if (activation == "silu")
    {
        return ActivationType::Silu;
    }
    return ActivationType::Gelu;
}

std::vector<std::string> ResolveBlockPattern(const Json*     backbone_cfg,
                                             std::int64_t    num_layers)
{
    if (backbone_cfg != nullptr
        && backbone_cfg->contains("block_pattern")
        && (*backbone_cfg)["block_pattern"].is_array())
    {
        std::vector<std::string> block_pattern;
        for (const auto& item : (*backbone_cfg)["block_pattern"])
        {
            if (!item.is_string())
            {
                throw std::invalid_argument("block_pattern entries must be strings.");
            }
            block_pattern.push_back(item.get<std::string>());
        }

        if (!block_pattern.empty())
        {
            return block_pattern;
        }
    }

    std::vector<std::string> fallback;
    fallback.reserve(static_cast<std::size_t>(num_layers));
    for (std::int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx)
    {
        fallback.push_back(layer_idx + 1 == num_layers ? "global" : "local");
    }
    return fallback;
}

}  // namespace

bool BuiltModel::HasEncoderClassifier() const noexcept
{
    return static_cast<bool>(encoder_classifier);
}

bool BuiltModel::HasVisionDetector() const noexcept
{
    return static_cast<bool>(vision_detector);
}

void ModelBuilderRegistry::Register(std::string model_type,
                                    BuilderFn   builder)
{
    builders_[std::move(model_type)] = std::move(builder);
}

BuiltModel ModelBuilderRegistry::Build(const Json&     metadata,
                                       const StateDict& state_dict) const
{
    const std::string model_type = InferModelType(metadata, state_dict);
    const auto        it         = builders_.find(model_type);
    if (it == builders_.end())
    {
        throw std::invalid_argument("No builder registered for model type '" + model_type + "'.");
    }

    return it->second(metadata, state_dict);
}

BuiltModel ModelBuilderRegistry::Build(const artifacts::npz::LoadedStateDictArtifact& artifact) const
{
    return Build(artifact.metadata, artifact.state_dict);
}

ModelBuilderRegistry ModelBuilderRegistry::CreateDefault()
{
    ModelBuilderRegistry registry;
    registry.Register("transformers.encoder_classifier", BuildEncoderClassifier);
    registry.Register("vlm.vision_detector", BuildVisionDetector);
    return registry;
}

std::string InferModelType(const Json&     metadata,
                           const StateDict& state_dict)
{
    if (metadata.contains("builder")
        && metadata["builder"].is_object()
        && metadata["builder"].contains("model_type")
        && metadata["builder"]["model_type"].is_string())
    {
        return metadata["builder"]["model_type"].get<std::string>();
    }

    if (FindTensor(state_dict, "token_embedding.weight") != nullptr
        && CountIndexedPrefix(state_dict, "encoder.") > 0
        && (FindTensor(state_dict, "head.3.weight") != nullptr
            || FindTensor(state_dict, "head.weight") != nullptr))
    {
        return "transformers.encoder_classifier";
    }

    if (FindTensor(state_dict, "backbone.patch_embed.proj.weight") != nullptr
        && CountIndexedPrefix(state_dict, "backbone.blocks.") > 0
        && FindTensor(state_dict, "detection_head.query_embed") != nullptr
        && FindTensor(state_dict, "detection_head.class_head.1.weight") != nullptr)
    {
        return "vlm.vision_detector";
    }

    throw std::invalid_argument("Unable to infer a model type from metadata/state_dict.");
}

models::EncoderClassifierConfig ResolveEncoderClassifierConfig(const Json&     metadata,
                                                              const StateDict& state_dict)
{
    const Json& model_cfg = ResolveModelMetadata(metadata);

    const Tensor& token_embedding = RequireTensor(state_dict, "token_embedding.weight");
    const Tensor* position        = FindTensor(state_dict, "position.positional_table");
    const Tensor* cls_token       = FindTensor(state_dict, "cls_token");
    const Tensor* fc1_weight      = FindTensor(state_dict, "encoder.0.residual_mlp.module.fc1.weight");
    const Tensor* head0_weight    = FindTensor(state_dict, "head.0.weight");
    const Tensor* head3_weight    = FindTensor(state_dict, "head.3.weight");
    const Tensor* head_weight     = FindTensor(state_dict, "head.weight");

    models::EncoderClassifierConfig config;
    config.vocab_size        = model_cfg.value("vocab_size", token_embedding.dim(0));
    config.max_length        = model_cfg.value("max_length",
                                               position != nullptr ? position->dim(1) : 0);
    config.embed_dim         = model_cfg.value("embed_dim", token_embedding.dim(1));
    config.depth             = model_cfg.value("depth", CountIndexedPrefix(state_dict, "encoder."));
    config.num_heads         = model_cfg.value("num_heads", 0);
    config.activation        = ResolveActivation(model_cfg);
    config.dropout           = model_cfg.value("dropout", 0.0F);
    config.attention_dropout = model_cfg.value("attention_dropout", config.dropout);
    config.qkv_bias          = model_cfg.value("qkv_bias", true);
    config.pre_norm          = model_cfg.value("pre_norm", true);
    config.layer_norm_eps    = model_cfg.value("layer_norm_eps", 1e-5F);
    config.drop_path         = model_cfg.value("drop_path", 0.0F);
    config.use_cls_token     = model_cfg.value("use_cls_token", cls_token != nullptr);
    config.pooling           = model_cfg.value("pooling", std::string("cls"));
    config.use_rope          = model_cfg.value("use_rope", position == nullptr);
    config.rope_base         = model_cfg.value("rope_base", 10000);

    if (model_cfg.contains("mlp_hidden_dim") && model_cfg["mlp_hidden_dim"].is_number_integer())
    {
        config.mlp_hidden_dim = model_cfg["mlp_hidden_dim"].get<std::int64_t>();
    }
    else if (fc1_weight != nullptr)
    {
        config.mlp_hidden_dim = fc1_weight->dim(0);
    }

    if (model_cfg.contains("mlp_ratio") && model_cfg["mlp_ratio"].is_number())
    {
        config.mlp_ratio = model_cfg["mlp_ratio"].get<float>();
    }
    else if (config.mlp_hidden_dim.has_value() && config.embed_dim > 0)
    {
        config.mlp_ratio = static_cast<float>(*config.mlp_hidden_dim) / static_cast<float>(config.embed_dim);
    }

    if (model_cfg.contains("cls_head_dim") && model_cfg["cls_head_dim"].is_number_integer())
    {
        config.cls_head_dim = model_cfg["cls_head_dim"].get<std::int64_t>();
    }
    else if (head0_weight != nullptr)
    {
        config.cls_head_dim = head0_weight->dim(0);
    }

    if (head3_weight != nullptr)
    {
        config.num_outputs = head3_weight->dim(0);
    }
    else if (head_weight != nullptr)
    {
        config.num_outputs = head_weight->dim(0);
    }
    else
    {
        config.num_outputs = model_cfg.value("num_outputs", 1);
    }

    if (config.max_length <= 0
        || config.num_heads <= 0)
    {
        throw std::invalid_argument("Resolved encoder-classifier config is incomplete. "
                                    "Expected metadata with max_length and num_heads.");
    }

    return config;
}

BuiltModel BuildEncoderClassifier(const Json&     metadata,
                                  const StateDict& state_dict)
{
    auto model = std::make_unique<models::EncoderClassifier>(ResolveEncoderClassifierConfig(metadata, state_dict));
    model->LoadParameters(state_dict);

    BuiltModel built;
    built.model_type         = "transformers.encoder_classifier";
    built.encoder_classifier = std::move(model);
    return built;
}

models::VisionDetectorConfig ResolveVisionDetectorConfig(const Json&     metadata,
                                                         const StateDict& state_dict)
{
    const Json* model_cfg    = FindModelMetadata(metadata);
    const Json* backbone_cfg = FindNestedObject(model_cfg, "backbone");
    const Json* head_cfg     = FindNestedObject(model_cfg, "head");

    if (backbone_cfg == nullptr)
    {
        backbone_cfg = model_cfg;
    }
    if (head_cfg == nullptr)
    {
        head_cfg = model_cfg;
    }

    const Tensor& patch_weight = RequireTensor(state_dict, "backbone.patch_embed.proj.weight");
    const Tensor& pos_embed    = RequireTensor(state_dict, "backbone.pos_embed");
    const Tensor* cls_token    = FindTensor(state_dict, "backbone.cls_token");
    const Tensor* q_bias       = FindTensor(state_dict, "backbone.blocks.0.residual_attention.module.w_q.bias");
    const Tensor* mlp_weight   = FindTensor(state_dict, "backbone.blocks.0.residual_mlp.module.fc1.weight");
    const Tensor& query_embed  = RequireTensor(state_dict, "detection_head.query_embed");
    const Tensor* box_hidden   = FindTensor(state_dict, "detection_head.box_head.1.weight");
    const Tensor* class_weight = FindTensor(state_dict, "detection_head.class_head.1.weight");

    const bool         use_cls_token = backbone_cfg != nullptr
                                           ? backbone_cfg->value("use_cls_token", cls_token != nullptr)
                                           : cls_token != nullptr;
    const std::int64_t patch_tokens  = pos_embed.dim(1) - (use_cls_token ? 1 : 0);
    const std::int64_t grid_size =
        static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(patch_tokens))));

    if (grid_size * grid_size != patch_tokens)
    {
        throw std::invalid_argument("Unable to infer a square patch grid from backbone.pos_embed.");
    }

    models::VisionDetectorConfig config;
    config.backbone.patch_size        = backbone_cfg != nullptr
                                            ? backbone_cfg->value("patch_size", patch_weight.dim(2))
                                            : patch_weight.dim(2);
    config.backbone.in_channels       = backbone_cfg != nullptr
                                            ? backbone_cfg->value("in_channels", patch_weight.dim(1))
                                            : patch_weight.dim(1);
    config.backbone.embed_dim         = backbone_cfg != nullptr
                                            ? backbone_cfg->value("embed_dim", patch_weight.dim(0))
                                            : patch_weight.dim(0);
    config.backbone.image_size        = backbone_cfg != nullptr
                                            ? backbone_cfg->value("image_size", config.backbone.patch_size * grid_size)
                                            : config.backbone.patch_size * grid_size;
    config.backbone.num_layers        = backbone_cfg != nullptr
                                            ? backbone_cfg->value("num_layers", CountIndexedPrefix(state_dict, "backbone.blocks."))
                                            : CountIndexedPrefix(state_dict, "backbone.blocks.");
    config.backbone.num_heads         = backbone_cfg != nullptr
                                            ? backbone_cfg->value("num_heads", 8)
                                            : 8;
    config.backbone.attention_dropout = backbone_cfg != nullptr
                                            ? backbone_cfg->value("attention_dropout", 0.0F)
                                            : 0.0F;
    config.backbone.dropout           = backbone_cfg != nullptr
                                            ? backbone_cfg->value("dropout", 0.0F)
                                            : 0.0F;
    config.backbone.qkv_bias          = backbone_cfg != nullptr
                                            ? backbone_cfg->value("qkv_bias", q_bias != nullptr)
                                            : q_bias != nullptr;
    config.backbone.use_cls_token     = use_cls_token;
    config.backbone.use_rope          = backbone_cfg != nullptr
                                            ? backbone_cfg->value("use_rope", true)
                                            : true;
    config.backbone.layer_norm_eps    = backbone_cfg != nullptr
                                            ? backbone_cfg->value("layer_norm_eps", 1e-6F)
                                            : 1e-6F;
    config.backbone.local_window_size = backbone_cfg != nullptr
                                            ? backbone_cfg->value("local_window_size", 7)
                                            : 7;
    config.backbone.local_rope_base   = backbone_cfg != nullptr
                                            ? backbone_cfg->value("local_rope_base", 10000)
                                            : 10000;
    config.backbone.global_rope_base  = backbone_cfg != nullptr
                                            ? backbone_cfg->value("global_rope_base", 1000000)
                                            : 1000000;
    config.backbone.block_pattern     = ResolveBlockPattern(backbone_cfg, config.backbone.num_layers);

    if (backbone_cfg != nullptr
        && backbone_cfg->contains("mlp_hidden_dim")
        && (*backbone_cfg)["mlp_hidden_dim"].is_number_integer())
    {
        config.backbone.mlp_hidden_dim = (*backbone_cfg)["mlp_hidden_dim"].get<std::int64_t>();
    }
    else if (mlp_weight != nullptr)
    {
        config.backbone.mlp_hidden_dim = mlp_weight->dim(0);
    }

    if (backbone_cfg != nullptr
        && backbone_cfg->contains("mlp_ratio")
        && (*backbone_cfg)["mlp_ratio"].is_number())
    {
        config.backbone.mlp_ratio = (*backbone_cfg)["mlp_ratio"].get<float>();
    }
    else if (config.backbone.embed_dim > 0 && config.backbone.mlp_hidden_dim > 0)
    {
        config.backbone.mlp_ratio =
            static_cast<float>(config.backbone.mlp_hidden_dim)
            / static_cast<float>(config.backbone.embed_dim);
    }

    config.head.num_queries    = head_cfg != nullptr
                                     ? head_cfg->value("num_queries", query_embed.dim(1))
                                     : query_embed.dim(1);
    config.head.num_classes    = head_cfg != nullptr
                                     ? head_cfg->value("num_classes", class_weight != nullptr ? class_weight->dim(0) : 0)
                                     : (class_weight != nullptr ? class_weight->dim(0) : 0);
    config.head.num_heads      = head_cfg != nullptr
                                     ? head_cfg->value("num_heads", config.backbone.num_heads)
                                     : config.backbone.num_heads;
    config.head.dropout        = head_cfg != nullptr
                                     ? head_cfg->value("dropout", 0.0F)
                                     : 0.0F;
    config.head.layer_norm_eps = head_cfg != nullptr
                                     ? head_cfg->value("layer_norm_eps", 1e-5F)
                                     : 1e-5F;

    if (head_cfg != nullptr
        && head_cfg->contains("mlp_hidden_dim")
        && (*head_cfg)["mlp_hidden_dim"].is_number_integer())
    {
        config.head.mlp_hidden_dim = (*head_cfg)["mlp_hidden_dim"].get<std::int64_t>();
    }
    else if (box_hidden != nullptr)
    {
        config.head.mlp_hidden_dim = box_hidden->dim(0);
    }

    if (config.backbone.patch_size <= 0
        || config.backbone.num_heads <= 0
        || config.backbone.mlp_hidden_dim <= 0
        || config.head.num_classes <= 0
        || config.head.num_queries <= 0
        || config.head.num_heads <= 0
        || config.head.mlp_hidden_dim <= 0)
    {
        throw std::invalid_argument("Resolved vision-detector config is incomplete.");
    }

    return config;
}

BuiltModel BuildVisionDetector(const Json&     metadata,
                               const StateDict& state_dict)
{
    auto model = std::make_unique<models::VisionDetector>(ResolveVisionDetectorConfig(metadata, state_dict));
    model->LoadParameters(state_dict);

    BuiltModel built;
    built.model_type      = "vlm.vision_detector";
    built.vision_detector = std::move(model);
    return built;
}

}  // namespace inference::model_builder
