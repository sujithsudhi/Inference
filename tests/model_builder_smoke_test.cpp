/// \file
/// \brief Smoke tests for the artifact model-builder registry.

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include "inference/core/artifact.hpp"
#include "inference/model_builder.hpp"
#include "inference/runtime/model_runner.hpp"
#include "inference/runtime/null_model_adapter.hpp"
#include "inference/runtime/request.hpp"
#include "inference/runtime/session.hpp"
#include "inference/tokenization/whitespace_tokenizer.hpp"

namespace
{

using inference::model_builder::BuildEncoderClassifier;
using inference::model_builder::InferModelType;
using inference::model_builder::ModelBuilderRegistry;
using inference::models::EncoderClassifier;
using inference::models::EncoderClassifierConfig;
using inference::models::VisionDetector;
using inference::models::VisionDetectorConfig;
using inference::runtime::ModelRunner;
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

std::vector<std::size_t> ToNpyShape(const std::vector<std::int64_t>& shape)
{
    std::vector<std::size_t> out;
    out.reserve(shape.size());

    for (const auto dim : shape)
    {
        out.push_back(static_cast<std::size_t>(dim));
    }

    return out;
}

void WriteLittleEndianUint64(std::ofstream& handle,
                             const std::uint64_t value)
{
    for (std::size_t index = 0; index < 8; ++index)
    {
        const auto byte = static_cast<unsigned char>((value >> (index * 8U)) & 0xFFU);
        handle.put(static_cast<char>(byte));
    }
}

void WriteSafeTensorWeights(const std::filesystem::path& path,
                            const StateDict&             state_dict)
{
    nlohmann::json header_json = nlohmann::json::object();
    std::uint64_t  offset      = 0;

    for (const auto& [key, tensor] : state_dict)
    {
        const std::uint64_t byte_count = static_cast<std::uint64_t>(tensor.numel()) * sizeof(float);
        header_json[key] = {{"dtype",        "F32"},
                            {"shape",        tensor.shape()},
                            {"data_offsets", {offset, offset + byte_count}}};
        offset += byte_count;
    }

    const std::string header_payload = header_json.dump();

    std::ofstream handle(path, std::ios::binary);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open safetensors artifact for writing.");
    }

    WriteLittleEndianUint64(handle, static_cast<std::uint64_t>(header_payload.size()));
    handle.write(header_payload.data(),
                 static_cast<std::streamsize>(header_payload.size()));

    for (const auto& [key, tensor] : state_dict)
    {
        (void)key;
        const auto byte_count = static_cast<std::streamsize>(tensor.numel() * sizeof(float));
        handle.write(reinterpret_cast<const char*>(tensor.data()), byte_count);
    }
}

void WriteArtifactBundle(const std::filesystem::path& root,
                         const nlohmann::json&        model_json,
                         const StateDict&             state_dict,
                         const std::string&           weight_format = "npz")
{
    std::filesystem::create_directories(root);

    const std::string weights_file = weight_format == "safetensors"
                                         ? "weights.safetensors"
                                         : "weights.npz";

    const nlohmann::json manifest_json = {
        {"schema_version", "inference.artifact/1"},
        {"artifact_name",  root.filename().string()},
        {"model_family",   "test"},
        {"task",           "test"},
        {"weight_format",  weight_format},
        {"files",          {{"metadata", "model.json"},
                            {"weights",  weights_file}}}
    };

    std::ofstream(root / "artifact.json") << manifest_json.dump(2) << "\n";
    std::ofstream(root / "model.json") << model_json.dump(2) << "\n";

    if (weight_format == "npz")
    {
        const auto npz_path = root / "weights.npz";
        std::filesystem::remove(npz_path);

        bool first = true;
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
        for (const auto& [key, tensor] : state_dict)
        {
            const auto shape = ToNpyShape(tensor.shape());
            cnpy::npz_save(npz_path.string(),
                           key,
                           tensor.data(),
                           shape,
                           first ? "w" : "a");
            first = false;
        }
#if defined(_MSC_VER)
#pragma warning(pop)
#endif
        return;
    }

    if (weight_format == "safetensors")
    {
        WriteSafeTensorWeights(root / "weights.safetensors",
                               state_dict);
        return;
    }

    throw std::runtime_error("Unsupported weight format for artifact smoke test: " + weight_format);
}

void RunSessionSmokeTest()
{
    const auto root = std::filesystem::temp_directory_path() / "inference-session-smoke";

    std::filesystem::create_directories(root);
    std::ofstream(root / "artifact.json") << "{}";
    std::ofstream(root / "tokenizer.json") << "{}";
    std::ofstream(root / "weights.npz") << "";

    inference::runtime::Session session(std::make_unique<inference::runtime::NullModelAdapter>(),
                                        std::make_shared<inference::tokenization::WhitespaceTokenizer>());

    const auto load_status = session.Load(inference::core::ArtifactBundle(root));
    Expect(load_status.ok(),
           "Expected session smoke artifact load to succeed.");

    inference::runtime::Request request;
    request.return_token_ids = true;
    request.segments.push_back({inference::runtime::InputKind::Text, "hello generic runtime", {}});

    const auto response = session.Run(request);
    Expect(!response.prompt_token_ids.empty(),
           "Expected prompt token ids to be populated.");
    Expect(response.status.code() == inference::core::StatusCode::NotImplemented,
           "Expected placeholder adapter to return NotImplemented.");

    std::filesystem::remove_all(root);
}

void RunArtifactBundleSmokeTest()
{
    const auto root = std::filesystem::temp_directory_path() / "inference-artifact-bundle-smoke";

    std::filesystem::remove_all(root);

    const auto manifest_root = root / "manifest_bundle";
    std::filesystem::create_directories(manifest_root / "tokenizer");

    std::ofstream(manifest_root / "artifact.json") << R"({
  "schema_version": "inference.artifact/1",
  "artifact_name": "tinystories-small",
  "model_family": "decoder",
  "task": "text",
  "weight_format": "npz",
  "files": {
    "metadata": "model.json",
    "weights": "weights.npz",
    "tokenizer": "tokenizer/tokenizer.json"
  }
})";
    std::ofstream(manifest_root / "model.json") << R"({
  "config": {
    "name": "tinystories"
  },
  "state_dict_keys": [
    "decoder.0.attention.Wq.weight"
  ]
})";
    std::ofstream(manifest_root / "weights.npz") << "";
    std::ofstream(manifest_root / "tokenizer" / "tokenizer.json") << "{}";

    const inference::core::ArtifactBundle manifest_bundle(manifest_root);
    const auto                            manifest_spec = manifest_bundle.Inspect();

    Expect(manifest_bundle.Exists(),
           "Expected manifest bundle to exist.");
    Expect(manifest_spec.layout == "manifest-dir",
           "Expected manifest-dir layout.");
    Expect(manifest_spec.weights_path.filename() == "weights.npz",
           "Expected manifest weights path to resolve.");
    Expect(manifest_spec.tokenizer_path.filename() == "tokenizer.json",
           "Expected manifest tokenizer path to resolve.");
    Expect(manifest_spec.model_family == "decoder",
           "Expected manifest model family to stay decoder.");

    const auto legacy_root = root / "legacy_prefix";
    std::filesystem::create_directories(legacy_root / "bert_tokenizer");

    std::ofstream(legacy_root / "imdb_checkpoint.json") << R"({
  "format": "npz",
  "config": {
    "name": "imdb"
  },
  "state_dict_keys": [
    "encoder.0.attention.Wq.weight"
  ]
})";
    std::ofstream(legacy_root / "imdb_checkpoint.npz") << "";
    std::ofstream(legacy_root / "bert_tokenizer" / "tokenizer.json") << "{}";

    const inference::core::ArtifactBundle legacy_bundle(legacy_root / "imdb_checkpoint");
    const auto                            legacy_spec = legacy_bundle.Inspect();

    Expect(legacy_bundle.Exists(),
           "Expected legacy bundle prefix to exist.");
    Expect(legacy_spec.layout == "legacy-prefix",
           "Expected legacy-prefix layout.");
    Expect(legacy_spec.weight_format == "npz",
           "Expected legacy weight format to resolve.");
    Expect(legacy_spec.metadata_path.filename() == "imdb_checkpoint.json",
           "Expected legacy metadata path to resolve.");
    Expect(legacy_spec.tokenizer_path.filename() == "tokenizer.json",
           "Expected legacy tokenizer path to resolve.");
    Expect(legacy_spec.model_family == "encoder",
           "Expected legacy model family hint to resolve.");

    std::filesystem::remove_all(root);
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

void RunModelRunnerMissingArtifactTest()
{
    ModelRunner runner;
    const auto  status = runner.Load(inference::core::ArtifactBundle(
        std::filesystem::temp_directory_path() / "inference-model-runner-missing"));

    Expect(status.code() == inference::core::StatusCode::NotFound,
           "Missing artifact load should return NotFound.");
    Expect(!runner.loaded(),
           "Runner should remain unloaded after a missing artifact load.");
}

void RunModelRunnerEncoderGraphTest()
{
    const auto temp_root = std::filesystem::temp_directory_path() / "inference-model-runner-encoder";
    std::filesystem::remove_all(temp_root);

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
    const StateDict   state_dict = BuildStateDict(reference_model.ParameterSpecs());

    const nlohmann::json metadata = nlohmann::json::parse(R"JSON(
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
            "use_rope": false,
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

    WriteArtifactBundle(temp_root, metadata, state_dict);

    ModelRunner runner;
    const auto  status = runner.Load(inference::core::ArtifactBundle(temp_root));
    Expect(status.ok(),
           "Encoder graph artifact should load through ModelRunner.");
    Expect(runner.loaded(),
           "Runner should report loaded after a successful load.");
    Expect(runner.HasEncoderClassifier(),
           "Runner should surface the encoder-classifier runtime.");
    Expect(runner.model_type() == "transformers.encoder_classifier",
           "Runner should preserve the resolved model type.");

    IndexTensor inputs({1, 4}, 0);
    inputs.at({0, 0}) = 1;
    inputs.at({0, 1}) = 2;
    inputs.at({0, 2}) = 3;

    Tensor attention_mask({1, 4}, 0.0F);
    attention_mask.at({0, 0}) = 1.0F;
    attention_mask.at({0, 1}) = 1.0F;
    attention_mask.at({0, 2}) = 1.0F;

    const Tensor logits = runner.RunEncoderClassifier(inputs, attention_mask);
    Expect(logits.shape() == std::vector<std::int64_t>({1, 1}),
           "ModelRunner encoder logits shape mismatch.");

    bool wrong_model_call_threw = false;
    try
    {
        runner.RunVisionDetector(Tensor({1, 3, 8, 8}, 0.0F));
    }
    catch (const std::logic_error&)
    {
        wrong_model_call_threw = true;
    }

    Expect(wrong_model_call_threw,
           "ModelRunner should reject vision execution for an encoder-only model.");

    std::filesystem::remove_all(temp_root);
}

void RunModelRunnerVisionGraphTest()
{
    const auto temp_root = std::filesystem::temp_directory_path() / "inference-model-runner-vision";
    std::filesystem::remove_all(temp_root);

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
    const StateDict state_dict = BuildStateDict(reference_model.ParameterSpecs());

    const nlohmann::json metadata = nlohmann::json::parse(R"JSON(
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

    WriteArtifactBundle(temp_root, metadata, state_dict);

    ModelRunner runner;
    const auto  status = runner.Load(inference::core::ArtifactBundle(temp_root));
    Expect(status.ok(),
           "Vision graph artifact should load through ModelRunner.");
    Expect(runner.HasVisionDetector(),
           "Runner should surface the vision-detector runtime.");
    Expect(runner.model_type() == "vlm.vision_detector",
           "Runner should preserve the resolved vision-detector model type.");

    Tensor image({1, 3, 8, 8}, 0.0F);
    for (std::size_t index = 0; index < image.numel(); ++index)
    {
        image.flat(index) = static_cast<float>(index) / 255.0F;
    }

    const auto backbone = runner.RunVisionBackbone(image);
    const auto output   = runner.RunVisionDetector(image);

    Expect(backbone.sequence_output.shape() == std::vector<std::int64_t>({1, 5, 8}),
           "ModelRunner vision-backbone output shape mismatch.");
    Expect(output.pred_boxes.shape() == std::vector<std::int64_t>({1, 4, 4}),
           "ModelRunner vision-detector box shape mismatch.");
    Expect(output.pred_class_logits.shape() == std::vector<std::int64_t>({1, 4, 3}),
           "ModelRunner vision-detector class-logit shape mismatch.");

    std::filesystem::remove_all(temp_root);
}

void RunModelRunnerVisionSafeTensorGraphTest()
{
    const auto temp_root = std::filesystem::temp_directory_path() / "inference-model-runner-vision-safetensors";
    std::filesystem::remove_all(temp_root);

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
    const StateDict state_dict = BuildStateDict(reference_model.ParameterSpecs());

    const nlohmann::json metadata = nlohmann::json::parse(R"JSON(
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
  },
  "format": "safetensors"
}
)JSON");

    WriteArtifactBundle(temp_root,
                        metadata,
                        state_dict,
                        "safetensors");

    ModelRunner runner;
    const auto  status = runner.Load(inference::core::ArtifactBundle(temp_root));
    Expect(status.ok(),
           "Vision safetensors artifact should load through ModelRunner.");
    Expect(runner.HasVisionDetector(),
           "Runner should surface the vision-detector runtime for safetensors artifacts.");

    Tensor image({1, 3, 8, 8}, 0.0F);
    for (std::size_t index = 0; index < image.numel(); ++index)
    {
        image.flat(index) = static_cast<float>(index) / 255.0F;
    }

    const auto output = runner.RunVisionDetector(image);
    Expect(output.pred_boxes.shape() == std::vector<std::int64_t>({1, 4, 4}),
           "Safetensors ModelRunner vision-detector box shape mismatch.");
    Expect(output.pred_class_logits.shape() == std::vector<std::int64_t>({1, 4, 3}),
           "Safetensors ModelRunner vision-detector class-logit shape mismatch.");

    std::filesystem::remove_all(temp_root);
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 2 && std::string(argv[1]) == "session_smoke")
        {
            RunSessionSmokeTest();
            return 0;
        }

        if (argc == 2 && std::string(argv[1]) == "artifact_bundle")
        {
            RunArtifactBundleSmokeTest();
            return 0;
        }

        if (argc == 2 && std::string(argv[1]) == "model_runner")
        {
            RunModelRunnerMissingArtifactTest();
            RunModelRunnerEncoderGraphTest();
            RunModelRunnerVisionGraphTest();
            RunModelRunnerVisionSafeTensorGraphTest();
            return 0;
        }

        if (argc != 1)
        {
            throw std::runtime_error(
                "Usage: model_builder_smoke_test [session_smoke|artifact_bundle|model_runner]");
        }

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
