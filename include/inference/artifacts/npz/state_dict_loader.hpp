#pragma once

#include <unordered_map>

#include "inference/artifacts/npz/transformer_loader.hpp"
#include "inference/transformer_core/tensor.hpp"

namespace inference::artifacts::npz
{

/// Loaded artifact bundle containing JSON metadata and a generic PyTorch-style state dict.
struct LoadedStateDictArtifact
{
    core::ArtifactSpec                     artifact;
    Json                                   metadata;
    transformer_core::StateDict            state_dict;
    std::unordered_map<std::string, int>   vocab;

    LoadedStateDictArtifact() = default;

    LoadedStateDictArtifact(const LoadedStateDictArtifact&) = delete;
    LoadedStateDictArtifact& operator = (const LoadedStateDictArtifact&) = delete;

    LoadedStateDictArtifact(LoadedStateDictArtifact&&) noexcept = default;
    LoadedStateDictArtifact& operator = (LoadedStateDictArtifact&&) noexcept = default;

    ~LoadedStateDictArtifact() = default;
};

/// Convert one NPZ array into an in-memory tensor.
transformer_core::Tensor ToTensor(const cnpy::NpyArray& array);

/// Convert all arrays in an NPZ archive into a PyTorch-style state dict.
transformer_core::StateDict LoadStateDict(const cnpy::npz_t& weights);

/// Load a generic state dict artifact from an inspected bundle.
LoadedStateDictArtifact LoadStateDictArtifact(const core::ArtifactBundle& artifact);

/// Load a generic state dict artifact from an artifact spec.
LoadedStateDictArtifact LoadStateDictArtifact(const core::ArtifactSpec& artifact);

}  // namespace inference::artifacts::npz
