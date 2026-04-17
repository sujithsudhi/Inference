/// \file
/// \brief Smoke test for artifact bundle inspection.

#include <filesystem>
#include <fstream>
#include <iostream>

#include "inference/core/artifact.hpp"

int main()
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
    const auto manifest_spec = manifest_bundle.Inspect();

    if (!manifest_bundle.Exists())
    {
        std::cerr << "Expected manifest bundle to exist." << std::endl;
        return 1;
    }
    if (manifest_spec.layout != "manifest-dir")
    {
        std::cerr << "Expected manifest-dir layout." << std::endl;
        return 1;
    }
    if (manifest_spec.weights_path.filename() != "weights.npz")
    {
        std::cerr << "Expected manifest weights path to resolve." << std::endl;
        return 1;
    }
    if (manifest_spec.tokenizer_path.filename() != "tokenizer.json")
    {
        std::cerr << "Expected manifest tokenizer path to resolve." << std::endl;
        return 1;
    }
    if (manifest_spec.model_family != "decoder")
    {
        std::cerr << "Expected manifest model family to stay decoder." << std::endl;
        return 1;
    }

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
    const auto legacy_spec = legacy_bundle.Inspect();

    if (!legacy_bundle.Exists())
    {
        std::cerr << "Expected legacy bundle prefix to exist." << std::endl;
        return 1;
    }
    if (legacy_spec.layout != "legacy-prefix")
    {
        std::cerr << "Expected legacy-prefix layout." << std::endl;
        return 1;
    }
    if (legacy_spec.weight_format != "npz")
    {
        std::cerr << "Expected legacy weight format to resolve." << std::endl;
        return 1;
    }
    if (legacy_spec.metadata_path.filename() != "imdb_checkpoint.json")
    {
        std::cerr << "Expected legacy metadata path to resolve." << std::endl;
        return 1;
    }
    if (legacy_spec.tokenizer_path.filename() != "tokenizer.json")
    {
        std::cerr << "Expected legacy tokenizer path to resolve." << std::endl;
        return 1;
    }
    if (legacy_spec.model_family != "encoder")
    {
        std::cerr << "Expected legacy model family hint to resolve." << std::endl;
        return 1;
    }

    std::filesystem::remove_all(root);
    return 0;
}
