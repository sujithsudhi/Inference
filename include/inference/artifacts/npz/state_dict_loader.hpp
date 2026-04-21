#pragma once

/// \file
/// \brief Generic state-dict artifact loader interfaces for NPZ and safetensors bundles.

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <unordered_map>

#include "inference/core/artifact.hpp"
#include "inference/transformer_core/tensor.hpp"

namespace inference::artifacts::npz
{

/// \brief JSON alias used by state-dict artifact loaders.
using Json = nlohmann::json;

/// Loaded artifact bundle containing JSON metadata and a generic PyTorch-style state dict.
struct LoadedStateDictArtifact
{
    /// Resolved artifact layout plus file-role metadata.
    core::ArtifactSpec                     artifact;
    /// Parsed `model.json` payload.
    Json                                   metadata;
    /// Fully materialized PyTorch-style state dict.
    transformer_core::StateDict            state_dict;
    /// Optional tokenizer vocabulary extracted from the tokenizer artifact when available.
    std::unordered_map<std::string, int>   vocab;

    LoadedStateDictArtifact() = default;

    LoadedStateDictArtifact(const LoadedStateDictArtifact&) = delete;
    LoadedStateDictArtifact& operator = (const LoadedStateDictArtifact&) = delete;

    LoadedStateDictArtifact(LoadedStateDictArtifact&&) noexcept = default;
    LoadedStateDictArtifact& operator = (LoadedStateDictArtifact&&) noexcept = default;

    ~LoadedStateDictArtifact() = default;
};

/// \brief Convert one NPZ array into an in-memory tensor.
transformer_core::Tensor ToTensor(const cnpy::NpyArray& array);

/// \brief Convert all arrays in one NPZ archive into a PyTorch-style state dict.
transformer_core::StateDict LoadStateDict(const cnpy::npz_t& weights);

/// \brief Load one generic state-dict artifact from an inspected bundle root.
LoadedStateDictArtifact LoadStateDictArtifact(const core::ArtifactBundle& artifact);

/// \brief Load one generic state-dict artifact from an already-resolved artifact spec.
LoadedStateDictArtifact LoadStateDictArtifact(const core::ArtifactSpec& artifact);

}  // namespace inference::artifacts::npz
