#pragma once

/// \file
/// \brief Legacy transformer NPZ loader and inspection helpers.

#include <string>
#include <unordered_map>
#include <vector>

#include "inference/artifacts/npz/transformer_weights.hpp"
#include "inference/core/artifact.hpp"

namespace inference::artifacts::npz
{

/// \brief Load one legacy transformer artifact from a bundle root or legacy prefix.
LoadedTransformerArtifact LoadTransformerArtifact(const core::ArtifactBundle& artifact);

/// \brief Load one legacy transformer artifact from an already-resolved artifact spec.
LoadedTransformerArtifact LoadTransformerArtifact(const core::ArtifactSpec& artifact);

/// \brief Read one `.npz` archive from disk.
cnpy::npz_t LoadNpz(const std::string& path);

/// \brief Read one JSON document from disk.
Json LoadJson(const std::string& path);

/// \brief Convert legacy transformer NPZ weights into structured Eigen-backed views.
TransformerModelWeights LoadTransformerWeights(const cnpy::npz_t& weights,
                                               const Json&        metadata);

/// \brief Load a tokenizer vocabulary JSON file into a token-to-id map.
std::unordered_map<std::string, int> LoadVocab(const std::string& path);

/// \brief Return all NPZ keys that begin with one prefix, sorted lexicographically.
std::vector<std::string> FindKeysWithPrefix(const TransformerModelWeights& weights,
                                            const std::string&             prefix);

/// \brief Return one raw NPZ array by key when present.
const cnpy::NpyArray* FindArray(const TransformerModelWeights& weights,
                                const std::string&             key);

}  // namespace inference::artifacts::npz
