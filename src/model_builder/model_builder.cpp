#include "inference/model_builder/model_builder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

namespace inference::model_builder
{

namespace
{

using inference::transformer_core::ActivationType;
using inference::transformer_core::StateDict;
using inference::transformer_core::Tensor;

struct GraphNodeSpec
{
    std::string              name;
    std::string              op;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string              param_prefix;
    Json                     attrs = Json::object();
};

struct GraphSpec
{
    std::string              version = "inference.graph/1";
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<GraphNodeSpec> nodes;
};

const Json* FindModelMetadata(const Json& metadata);

const Json* FindGraphMetadata(const Json& metadata)
{
    if (metadata.contains("builder")
        && metadata["builder"].is_object()
        && metadata["builder"].contains("graph")
        && metadata["builder"]["graph"].is_object())
    {
        return &metadata["builder"]["graph"];
    }

    if (metadata.contains("graph") && metadata["graph"].is_object())
    {
        return &metadata["graph"];
    }

    return nullptr;
}

bool HasGraphMetadata(const Json& metadata)
{
    return FindGraphMetadata(metadata) != nullptr;
}

std::vector<std::string> ParseStringArray(const Json&        value,
                                          const std::string& field_name)
{
    if (!value.is_array())
    {
        throw std::invalid_argument(field_name + " must be an array of strings.");
    }

    std::vector<std::string> items;
    items.reserve(value.size());

    for (const auto& item : value)
    {
        if (!item.is_string())
        {
            throw std::invalid_argument(field_name + " must contain only strings.");
        }
        items.push_back(item.get<std::string>());
    }

    return items;
}

GraphSpec ParseGraphSpec(const Json& metadata)
{
    const Json* graph_json = FindGraphMetadata(metadata);
    if (graph_json == nullptr)
    {
        throw std::invalid_argument("Artifact metadata does not contain builder.graph.");
    }

    GraphSpec graph;
    graph.version = graph_json->value("version", graph.version);
    if (graph.version != "inference.graph/1")
    {
        throw std::invalid_argument("Unsupported graph version '" + graph.version + "'.");
    }

    if (graph_json->contains("inputs"))
    {
        graph.inputs = ParseStringArray((*graph_json)["inputs"], "graph.inputs");
    }

    if (graph_json->contains("outputs"))
    {
        graph.outputs = ParseStringArray((*graph_json)["outputs"], "graph.outputs");
    }

    if (!graph_json->contains("nodes") || !(*graph_json)["nodes"].is_array())
    {
        throw std::invalid_argument("builder.graph.nodes must be an array.");
    }

    graph.nodes.reserve((*graph_json)["nodes"].size());

    for (const auto& node_json : (*graph_json)["nodes"])
    {
        if (!node_json.is_object())
        {
            throw std::invalid_argument("builder.graph.nodes entries must be objects.");
        }

        if (!node_json.contains("name") || !node_json["name"].is_string())
        {
            throw std::invalid_argument("builder.graph.nodes entries require a string name.");
        }

        if (!node_json.contains("op") || !node_json["op"].is_string())
        {
            throw std::invalid_argument("builder.graph.nodes entries require a string op.");
        }

        GraphNodeSpec node;
        node.name = node_json["name"].get<std::string>();
        node.op   = node_json["op"].get<std::string>();

        if (node_json.contains("inputs"))
        {
            node.inputs = ParseStringArray(node_json["inputs"],
                                           "graph.nodes[" + node.name + "].inputs");
        }

        if (node_json.contains("outputs"))
        {
            node.outputs = ParseStringArray(node_json["outputs"],
                                            "graph.nodes[" + node.name + "].outputs");
        }

        if (node.outputs.empty())
        {
            node.outputs.push_back(node.name);
        }

        if (node_json.contains("param_prefix"))
        {
            if (!node_json["param_prefix"].is_string())
            {
                throw std::invalid_argument("graph.nodes[" + node.name + "].param_prefix must be a string.");
            }
            node.param_prefix = node_json["param_prefix"].get<std::string>();
        }

        if (node_json.contains("attrs"))
        {
            if (!node_json["attrs"].is_object())
            {
                throw std::invalid_argument("graph.nodes[" + node.name + "].attrs must be an object.");
            }
            node.attrs = node_json["attrs"];
        }

        graph.nodes.push_back(std::move(node));
    }

    if (graph.outputs.empty() && !graph.nodes.empty())
    {
        graph.outputs = graph.nodes.back().outputs;
    }

    return graph;
}

void ValidateGraphSpec(const GraphSpec& graph)
{
    if (graph.nodes.empty())
    {
        throw std::invalid_argument("builder.graph.nodes must not be empty.");
    }

    std::unordered_set<std::string> graph_inputs(graph.inputs.begin(), graph.inputs.end());
    std::unordered_set<std::string> produced_outputs;
    std::unordered_set<std::string> node_names;

    for (const auto& node : graph.nodes)
    {
        if (node.name.empty())
        {
            throw std::invalid_argument("Graph nodes must have non-empty names.");
        }

        if (!node_names.insert(node.name).second)
        {
            throw std::invalid_argument("Duplicate graph node name '" + node.name + "'.");
        }

        for (const auto& input_name : node.inputs)
        {
            if (graph_inputs.find(input_name) == graph_inputs.end()
                && produced_outputs.find(input_name) == produced_outputs.end())
            {
                throw std::invalid_argument("Graph node '"
                                            + node.name
                                            + "' references unknown input '"
                                            + input_name
                                            + "'.");
            }
        }

        std::unordered_set<std::string> local_outputs;
        for (const auto& output_name : node.outputs)
        {
            if (output_name.empty())
            {
                throw std::invalid_argument("Graph node '" + node.name + "' declares an empty output name.");
            }

            if (!local_outputs.insert(output_name).second)
            {
                throw std::invalid_argument("Graph node '"
                                            + node.name
                                            + "' declares duplicate output '"
                                            + output_name
                                            + "'.");
            }

            if (graph_inputs.find(output_name) != graph_inputs.end()
                || !produced_outputs.insert(output_name).second)
            {
                throw std::invalid_argument("Graph output '" + output_name + "' is defined more than once.");
            }
        }
    }

    for (const auto& output_name : graph.outputs)
    {
        if (graph_inputs.find(output_name) == graph_inputs.end()
            && produced_outputs.find(output_name) == produced_outputs.end())
        {
            throw std::invalid_argument("builder.graph.outputs references unknown tensor '" + output_name + "'.");
        }
    }
}

const GraphNodeSpec* FindUniqueGraphNode(const GraphSpec&       graph,
                                         const std::string_view op)
{
    const GraphNodeSpec* match = nullptr;

    for (const auto& node : graph.nodes)
    {
        if (node.op != op)
        {
            continue;
        }

        if (match != nullptr)
        {
            throw std::invalid_argument("builder.graph contains more than one '" + std::string(op) + "' node.");
        }

        match = &node;
    }

    return match;
}

bool GraphHasOp(const GraphSpec&       graph,
                const std::string_view op)
{
    return FindUniqueGraphNode(graph, op) != nullptr;
}

void MergeObject(Json&        target,
                 const Json* source)
{
    if (source == nullptr || !source->is_object())
    {
        return;
    }

    for (const auto& [key, value] : source->items())
    {
        target[key] = value;
    }
}

Json ResolveGraphModelConfig(const Json& metadata)
{
    const Json* model_cfg = FindModelMetadata(metadata);
    if (model_cfg == nullptr)
    {
        return Json::object();
    }

    return *model_cfg;
}

std::string InferGraphModelType(const GraphSpec& graph)
{
    const bool is_encoder_classifier =
        GraphHasOp(graph, "token_embedding")
        && GraphHasOp(graph, "transformer_encoder")
        && GraphHasOp(graph, "layer_norm")
        && GraphHasOp(graph, "classifier_head");

    const bool is_vision_detector =
        GraphHasOp(graph, "patch_embedding")
        && GraphHasOp(graph, "vision_backbone")
        && GraphHasOp(graph, "detection_head");

    if (is_encoder_classifier == is_vision_detector)
    {
        throw std::invalid_argument("Unable to resolve a supported runtime model from builder.graph.");
    }

    return is_encoder_classifier
               ? "transformers.encoder_classifier"
               : "vlm.vision_detector";
}

std::string ValidateGraphModelTypeHint(const Json&        metadata,
                                       const std::string& graph_model_type)
{
    if (metadata.contains("builder")
        && metadata["builder"].is_object()
        && metadata["builder"].contains("model_type")
        && metadata["builder"]["model_type"].is_string())
    {
        const std::string hinted = metadata["builder"]["model_type"].get<std::string>();
        if (hinted != "graph" && hinted != graph_model_type)
        {
            throw std::invalid_argument("builder.model_type '"
                                        + hinted
                                        + "' does not match graph-derived type '"
                                        + graph_model_type
                                        + "'.");
        }
    }

    return graph_model_type;
}

void ValidateGraphPrefixes(const GraphSpec& graph,
                           const StateDict& state_dict)
{
    for (const auto& node : graph.nodes)
    {
        if (node.param_prefix.empty())
        {
            continue;
        }

        const bool found = std::any_of(state_dict.begin(),
                                       state_dict.end(),
                                       [&](const auto& kv)
                                       {
                                           return kv.first.rfind(node.param_prefix, 0) == 0;
                                       });

        if (!found)
        {
            throw std::invalid_argument("Graph node '"
                                        + node.name
                                        + "' did not match any state_dict tensor with prefix '"
                                        + node.param_prefix
                                        + "'.");
        }
    }
}

Json BuildGraphEncoderMetadata(const Json&      metadata,
                               const GraphSpec& graph)
{
    Json model_cfg = ResolveGraphModelConfig(metadata);

    MergeObject(model_cfg, FindUniqueGraphNode(graph, "token_embedding") != nullptr
                               ? &FindUniqueGraphNode(graph, "token_embedding")->attrs
                               : nullptr);
    MergeObject(model_cfg, FindUniqueGraphNode(graph, "cls_token") != nullptr
                               ? &FindUniqueGraphNode(graph, "cls_token")->attrs
                               : nullptr);
    MergeObject(model_cfg, FindUniqueGraphNode(graph, "positional_encoding") != nullptr
                               ? &FindUniqueGraphNode(graph, "positional_encoding")->attrs
                               : nullptr);
    MergeObject(model_cfg, FindUniqueGraphNode(graph, "transformer_encoder") != nullptr
                               ? &FindUniqueGraphNode(graph, "transformer_encoder")->attrs
                               : nullptr);
    MergeObject(model_cfg, FindUniqueGraphNode(graph, "layer_norm") != nullptr
                               ? &FindUniqueGraphNode(graph, "layer_norm")->attrs
                               : nullptr);
    MergeObject(model_cfg, FindUniqueGraphNode(graph, "classifier_head") != nullptr
                               ? &FindUniqueGraphNode(graph, "classifier_head")->attrs
                               : nullptr);

    if (!model_cfg.contains("num_outputs")
        && model_cfg.contains("num_classes")
        && model_cfg["num_classes"].is_number_integer())
    {
        model_cfg["num_outputs"] = model_cfg["num_classes"];
    }

    return Json{{"model", model_cfg}};
}

Json BuildGraphVisionMetadata(const Json&      metadata,
                              const GraphSpec& graph)
{
    const Json  model_cfg     = ResolveGraphModelConfig(metadata);
    const Json* backbone_base = model_cfg.contains("backbone") && model_cfg["backbone"].is_object()
                                    ? &model_cfg["backbone"]
                                    : &model_cfg;
    const Json* head_base     = model_cfg.contains("head") && model_cfg["head"].is_object()
                                    ? &model_cfg["head"]
                                    : &model_cfg;

    Json backbone_cfg = Json::object();
    Json head_cfg     = Json::object();

    MergeObject(backbone_cfg, backbone_base);
    MergeObject(head_cfg, head_base);

    MergeObject(backbone_cfg, FindUniqueGraphNode(graph, "patch_embedding") != nullptr
                                  ? &FindUniqueGraphNode(graph, "patch_embedding")->attrs
                                  : nullptr);
    MergeObject(backbone_cfg, FindUniqueGraphNode(graph, "cls_token") != nullptr
                                  ? &FindUniqueGraphNode(graph, "cls_token")->attrs
                                  : nullptr);
    MergeObject(backbone_cfg, FindUniqueGraphNode(graph, "vision_backbone") != nullptr
                                  ? &FindUniqueGraphNode(graph, "vision_backbone")->attrs
                                  : nullptr);
    MergeObject(head_cfg, FindUniqueGraphNode(graph, "detection_head") != nullptr
                              ? &FindUniqueGraphNode(graph, "detection_head")->attrs
                              : nullptr);

    return Json{{"model", {{"backbone", backbone_cfg},
                           {"head",     head_cfg}}}};
}

BuiltModel BuildGraphModel(const Json&     metadata,
                           const StateDict& state_dict)
{
    const GraphSpec   graph            = ParseGraphSpec(metadata);
    const std::string graph_model_type =
        ValidateGraphModelTypeHint(metadata, InferGraphModelType(graph));

    ValidateGraphSpec(graph);
    ValidateGraphPrefixes(graph, state_dict);

    if (graph_model_type == "transformers.encoder_classifier")
    {
        return BuildEncoderClassifier(BuildGraphEncoderMetadata(metadata, graph), state_dict);
    }

    if (graph_model_type == "vlm.vision_detector")
    {
        return BuildVisionDetector(BuildGraphVisionMetadata(metadata, graph), state_dict);
    }

    throw std::invalid_argument("No runtime builder available for graph-derived model type '"
                                + graph_model_type
                                + "'.");
}

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
    if (HasGraphMetadata(metadata))
    {
        return BuildGraphModel(metadata, state_dict);
    }

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
    if (HasGraphMetadata(metadata))
    {
        const GraphSpec graph = ParseGraphSpec(metadata);
        ValidateGraphSpec(graph);
        return ValidateGraphModelTypeHint(metadata, InferGraphModelType(graph));
    }

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
    else if (model_cfg.contains("num_classes") && model_cfg["num_classes"].is_number_integer())
    {
        config.num_outputs = model_cfg["num_classes"].get<std::int64_t>();
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
