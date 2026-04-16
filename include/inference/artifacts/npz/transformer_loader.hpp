#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "inference/artifacts/npz/transformer_weights.hpp"
#include "inference/core/artifact.hpp"

namespace inference::artifacts::npz
{

LoadedTransformerArtifact LoadTransformerArtifact(const core::ArtifactBundle& artifact);

LoadedTransformerArtifact LoadTransformerArtifact(const core::ArtifactSpec& artifact);

cnpy::npz_t LoadNpz(const std::string& path);

Json LoadJson(const std::string& path);

TransformerModelWeights LoadTransformerWeights(const cnpy::npz_t& weights,
                                               const Json&        metadata);

std::unordered_map<std::string, int> LoadVocab(const std::string& path);

std::vector<std::string> FindKeysWithPrefix(const TransformerModelWeights& weights,
                                            const std::string&             prefix);

const cnpy::NpyArray* FindArray(const TransformerModelWeights& weights,
                                const std::string&             key);

}  // namespace inference::artifacts::npz
