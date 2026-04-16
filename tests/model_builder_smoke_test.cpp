#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "inference/model_builder.hpp"

namespace
{

using inference::model_builder::BuildEncoderClassifier;
using inference::model_builder::InferModelType;
using inference::model_builder::ModelBuilderRegistry;
using inference::models::EncoderClassifier;
using inference::models::EncoderClassifierConfig;
using inference::models::VisionDetector;
using inference::models::VisionDetectorConfig;
using inference::transformer_core::IndexTensor;
using inference::transformer_core::StateDict;
using inference::transformer_core::Tensor;
using inference::transformer_core::TensorSpec;

void Expect(bool condition,
            const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

StateDict BuildStateDict(const std::vector<TensorSpec>& specs)
{
    StateDict state_dict;
    float     next_value = 0.01F;

    for (const auto& spec : specs)
    {
        Tensor tensor(spec.shape, 0.0F);
        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            tensor.flat(i) = next_value;
            next_value += 0.01F;
        }
        state_dict.emplace(spec.name, std::move(tensor));
    }

    return state_dict;
}

void RunEncoderRegistryTest()
{
    EncoderClassifierConfig config;
    config.vocab_size        = 16;
    config.max_length        = 5;
    config.embed_dim         = 8;
    config.depth             = 2;
    config.num_heads         = 2;
    config.mlp_ratio         = 2.0F;
    config.mlp_hidden_dim    = 16;
    config.dropout           = 0.0F;
    config.attention_dropout = 0.0F;
    config.use_rope          = false;
    config.cls_head_dim      = 8;
    config.num_outputs       = 1;

    EncoderClassifier reference_model(config);
    StateDict         state_dict = BuildStateDict(reference_model.ParameterSpecs());

    nlohmann::json metadata = {
        {"builder", {{"model_type", "transformers.encoder_classifier"}}},
        {"config", {{"model", {{"vocab_size", 16},
                               {"max_length", 5},
                               {"embed_dim", 8},
                               {"depth", 2},
                               {"num_heads", 2},
                               {"mlp_ratio", 2.0},
                               {"mlp_hidden_dim", 16},
                               {"dropout", 0.0},
                               {"attention_dropout", 0.0},
                               {"qkv_bias", true},
                               {"pre_norm", true},
                               {"layer_norm_eps", 1e-5},
                               {"drop_path", 0.0},
                               {"use_cls_token", true},
                               {"cls_head_dim", 8},
                               {"num_outputs", 1},
                               {"pooling", "cls"},
                               {"use_rope", false},
                               {"rope_base", 10000}}}}}
    };

    Expect(InferModelType(metadata, state_dict) == "transformers.encoder_classifier",
           "InferModelType should resolve the encoder-classifier builder.");

    ModelBuilderRegistry registry = ModelBuilderRegistry::CreateDefault();
    auto built = registry.Build(metadata, state_dict);
    Expect(built.HasEncoderClassifier(),
           "Registry should build an encoder-classifier instance.");
    Expect(built.model_type == "transformers.encoder_classifier",
           "BuiltModel should preserve the builder key.");

    IndexTensor inputs({1, 4}, 0);
    inputs.at({0, 0}) = 1;
    inputs.at({0, 1}) = 2;
    inputs.at({0, 2}) = 3;

    Tensor attention_mask({1, 4}, 0.0F);
    attention_mask.at({0, 0}) = 1.0F;
    attention_mask.at({0, 1}) = 1.0F;
    attention_mask.at({0, 2}) = 1.0F;

    Tensor logits = built.encoder_classifier->Forward(inputs, attention_mask);
    Expect(logits.shape() == std::vector<std::int64_t>({1, 1}),
           "Encoder-classifier forward shape mismatch.");
}

void RunEncoderGraphBuilderTest()
{
    EncoderClassifierConfig config;
    config.vocab_size        = 16;
    config.max_length        = 5;
    config.embed_dim         = 8;
    config.depth             = 2;
    config.num_heads         = 2;
    config.mlp_ratio         = 2.0F;
    config.mlp_hidden_dim    = 16;
    config.dropout           = 0.0F;
    config.attention_dropout = 0.0F;
    config.use_rope          = false;
    config.cls_head_dim      = 8;
    config.num_outputs       = 1;

    EncoderClassifier reference_model(config);
    StateDict         state_dict = BuildStateDict(reference_model.ParameterSpecs());

    nlohmann::json metadata = nlohmann::json::parse(R"JSON(
{
  "builder": {
    "model_type": "graph",
    "graph": {
      "version": "inference.graph/1",
      "inputs": ["input_ids", "attention_mask"],
      "outputs": ["logits"],
      "nodes": [
        {
          "name": "token_embedding",
          "op": "token_embedding",
          "inputs": ["input_ids"],
          "outputs": ["embedded_tokens"],
          "param_prefix": "token_embedding.",
          "attrs": {
            "vocab_size": 16,
            "embed_dim": 8
          }
        },
        {
          "name": "cls_token",
          "op": "cls_token",
          "outputs": ["cls_token_tensor"],
          "param_prefix": "cls_token",
          "attrs": {
            "use_cls_token": true
          }
        },
        {
          "name": "position",
          "op": "positional_encoding",
          "inputs": ["embedded_tokens"],
          "outputs": ["positioned_tokens"],
          "param_prefix": "position.",
          "attrs": {
            "max_length": 5,
            "use_rope": false
          }
        },
        {
          "name": "encoder",
          "op": "transformer_encoder",
          "inputs": ["positioned_tokens", "attention_mask"],
          "outputs": ["encoded_tokens"],
          "param_prefix": "encoder.",
          "attrs": {
            "depth": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "mlp_hidden_dim": 16,
            "dropout": 0.0,
            "attention_dropout": 0.0,
            "qkv_bias": true,
            "pre_norm": true,
            "layer_norm_eps": 1e-5,
            "drop_path": 0.0,
            "pooling": "cls",
            "rope_base": 10000
          }
        },
        {
          "name": "norm",
          "op": "layer_norm",
          "inputs": ["encoded_tokens"],
          "outputs": ["normalized_tokens"],
          "param_prefix": "norm."
        },
        {
          "name": "classifier",
          "op": "classifier_head",
          "inputs": ["normalized_tokens"],
          "outputs": ["logits"],
          "param_prefix": "head.",
          "attrs": {
            "cls_head_dim": 8,
            "num_outputs": 1
          }
        }
      ]
    }
  }
}
)JSON");

    Expect(InferModelType(metadata, state_dict) == "transformers.encoder_classifier",
           "Graph metadata should resolve to the encoder-classifier runtime.");

    ModelBuilderRegistry registry = ModelBuilderRegistry::CreateDefault();
    auto built = registry.Build(metadata, state_dict);
    Expect(built.HasEncoderClassifier(),
           "Graph builder should build an encoder-classifier instance.");

    IndexTensor inputs({1, 4}, 0);
    inputs.at({0, 0}) = 1;
    inputs.at({0, 1}) = 2;
    inputs.at({0, 2}) = 3;

    Tensor attention_mask({1, 4}, 0.0F);
    attention_mask.at({0, 0}) = 1.0F;
    attention_mask.at({0, 1}) = 1.0F;
    attention_mask.at({0, 2}) = 1.0F;

    Tensor logits = built.encoder_classifier->Forward(inputs, attention_mask);
    Expect(logits.shape() == std::vector<std::int64_t>({1, 1}),
           "Graph-built encoder-classifier forward shape mismatch.");
}

void RunVisionGraphBuilderTest()
{
    VisionDetectorConfig config;
    config.backbone.image_size        = 8;
    config.backbone.patch_size        = 4;
    config.backbone.in_channels       = 3;
    config.backbone.embed_dim         = 8;
    config.backbone.num_layers        = 2;
    config.backbone.num_heads         = 2;
    config.backbone.mlp_ratio         = 2.0F;
    config.backbone.mlp_hidden_dim    = 16;
    config.backbone.attention_dropout = 0.0F;
    config.backbone.dropout           = 0.0F;
    config.backbone.qkv_bias          = true;
    config.backbone.use_cls_token     = true;
    config.backbone.use_rope          = true;
    config.backbone.layer_norm_eps    = 1e-6F;
    config.backbone.local_window_size = 3;
    config.backbone.local_rope_base   = 10000;
    config.backbone.global_rope_base  = 1000000;
    config.backbone.block_pattern     = {"local", "global"};
    config.head.num_queries           = 4;
    config.head.num_classes           = 3;
    config.head.num_heads             = 2;
    config.head.mlp_hidden_dim        = 12;
    config.head.dropout               = 0.0F;
    config.head.layer_norm_eps        = 1e-5F;

    VisionDetector reference_model(config);
    StateDict      state_dict = BuildStateDict(reference_model.ParameterSpecs());

    nlohmann::json metadata = nlohmann::json::parse(R"JSON(
{
  "builder": {
    "graph": {
      "version": "inference.graph/1",
      "inputs": ["image"],
      "outputs": ["pred_boxes", "pred_objectness_logits", "pred_class_logits"],
      "nodes": [
        {
          "name": "patch_embed",
          "op": "patch_embedding",
          "inputs": ["image"],
          "outputs": ["patch_tokens"],
          "param_prefix": "backbone.patch_embed.",
          "attrs": {
            "image_size": 8,
            "patch_size": 4,
            "in_channels": 3,
            "embed_dim": 8
          }
        },
        {
          "name": "backbone",
          "op": "vision_backbone",
          "inputs": ["patch_tokens"],
          "outputs": ["vision_features"],
          "param_prefix": "backbone.",
          "attrs": {
            "num_layers": 2,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "mlp_hidden_dim": 16,
            "attention_dropout": 0.0,
            "dropout": 0.0,
            "qkv_bias": true,
            "use_cls_token": true,
            "use_rope": true,
            "layer_norm_eps": 1e-6,
            "local_window_size": 3,
            "local_rope_base": 10000,
            "global_rope_base": 1000000,
            "block_pattern": ["local", "global"]
          }
        },
        {
          "name": "head",
          "op": "detection_head",
          "inputs": ["vision_features"],
          "outputs": ["pred_boxes", "pred_objectness_logits", "pred_class_logits"],
          "param_prefix": "detection_head.",
          "attrs": {
            "num_queries": 4,
            "num_classes": 3,
            "num_heads": 2,
            "mlp_hidden_dim": 12,
            "dropout": 0.0,
            "layer_norm_eps": 1e-5
          }
        }
      ]
    }
  }
}
)JSON");

    Expect(InferModelType(metadata, state_dict) == "vlm.vision_detector",
           "Graph metadata should resolve to the vision-detector runtime.");

    ModelBuilderRegistry registry = ModelBuilderRegistry::CreateDefault();
    auto built = registry.Build(metadata, state_dict);
    Expect(built.HasVisionDetector(),
           "Graph builder should build a vision-detector instance.");

    Tensor images({1, 3, 8, 8}, 0.0F);
    for (std::size_t index = 0; index < images.numel(); ++index)
    {
        images.flat(index) = static_cast<float>(index) / 255.0F;
    }

    const auto output = built.vision_detector->Forward(images);
    Expect(output.pred_boxes.shape() == std::vector<std::int64_t>({1, 4, 4}),
           "Graph-built vision detector predicted box shape mismatch.");
    Expect(output.pred_class_logits.shape() == std::vector<std::int64_t>({1, 4, 3}),
           "Graph-built vision detector class shape mismatch.");
}

}  // namespace

int main()
{
    try
    {
        RunEncoderRegistryTest();
        RunEncoderGraphBuilderTest();
        RunVisionGraphBuilderTest();
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
