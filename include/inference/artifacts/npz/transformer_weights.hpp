#pragma once

/// \file
/// \brief Legacy transformer weight view types used by the NPZ migration loader.

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>
#include <cnpy.h>
#include <nlohmann/json.hpp>

#include "inference/core/artifact.hpp"

namespace inference::artifacts::npz
{

using Json     = nlohmann::json;
using MatrixRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector   = Eigen::VectorXf;

/// \brief Eigen-backed attention parameter views for one legacy transformer layer.
struct TransformerAttentionWeights
{
    Eigen::Map<const MatrixRM> Wq_weight;
    Eigen::Map<const Vector>   Wq_bias;
    Eigen::Map<const MatrixRM> Wk_weight;
    Eigen::Map<const Vector>   Wk_bias;
    Eigen::Map<const MatrixRM> Wv_weight;
    Eigen::Map<const Vector>   Wv_bias;
    Eigen::Map<const MatrixRM> Wo_weight;
    Eigen::Map<const Vector>   Wo_bias;
};

/// \brief Eigen-backed feed-forward parameter views for one legacy transformer layer.
struct TransformerFeedForwardWeights
{
    Eigen::Map<const MatrixRM> fc1_weight;
    Eigen::Map<const Vector>   fc1_bias;
    Eigen::Map<const MatrixRM> fc2_weight;
    Eigen::Map<const Vector>   fc2_bias;
};

/// \brief Residual-attention wrapper weights including the nested module plus layer norm.
struct TransformerResidualAttention
{
    TransformerAttentionWeights module;
    Eigen::Map<const Vector>    norm_weight;
    Eigen::Map<const Vector>    norm_bias;
};

/// \brief Residual-MLP wrapper weights including the nested module plus layer norm.
struct TransformerResidualFeedForward
{
    TransformerFeedForwardWeights module;
    Eigen::Map<const Vector>      norm_weight;
    Eigen::Map<const Vector>      norm_bias;
};

/// \brief Legacy encoder-layer weight bundle assembled from the NPZ loader.
struct TransformerEncoderLayerWeights
{
    TransformerAttentionWeights    attention;
    TransformerFeedForwardWeights  ff;
    TransformerResidualAttention   residue1;
    TransformerResidualFeedForward residue2;
};

/// \brief Structured legacy transformer checkpoint views used during migration and inspection.
struct TransformerModelWeights
{
    /// Raw named arrays preserved from the source NPZ archive.
    std::map<std::string, cnpy::NpyArray> named;
    /// Encoder layer bundles materialized from the named arrays.
    std::vector<TransformerEncoderLayerWeights> encoder;

    /// Optional CLS token view.
    std::unique_ptr<Eigen::Map<const MatrixRM>> cls_token;
    /// Optional token-embedding table view.
    std::unique_ptr<Eigen::Map<const MatrixRM>> token_embedding_weight;
    /// Optional trainable position table view.
    std::unique_ptr<Eigen::Map<const MatrixRM>> position_positional_table;
    /// Optional final normalization scale.
    std::unique_ptr<Eigen::Map<const Vector>>   norm_weight;
    /// Optional final normalization bias.
    std::unique_ptr<Eigen::Map<const Vector>>   norm_bias;

    /// Optional first classifier-layer weight.
    std::unique_ptr<Eigen::Map<const MatrixRM>> head0_weight;
    /// Optional first classifier-layer bias.
    std::unique_ptr<Eigen::Map<const Vector>>   head0_bias;
    /// Optional output classifier-layer weight.
    std::unique_ptr<Eigen::Map<const MatrixRM>> head3_weight;
    /// Optional output classifier-layer bias.
    std::unique_ptr<Eigen::Map<const Vector>>   head3_bias;

    /// Additional head weights preserved by their original parameter names.
    std::unordered_map<std::string, std::unique_ptr<Eigen::Map<const MatrixRM>>> head_weights;
    /// Additional head biases preserved by their original parameter names.
    std::unordered_map<std::string, std::unique_ptr<Eigen::Map<const Vector>>>   head_biases;
};

/// \brief Loaded legacy transformer artifact containing metadata, weights, and optional vocab.
struct LoadedTransformerArtifact
{
    /// Resolved artifact layout plus file-role metadata.
    core::ArtifactSpec               artifact;
    /// Parsed metadata JSON.
    Json                             metadata;
    /// Structured weight views over the loaded NPZ archive.
    TransformerModelWeights          model_weights;
    /// Optional tokenizer vocabulary extracted from the tokenizer artifact.
    std::unordered_map<std::string, int> vocab;

    LoadedTransformerArtifact() = default;

    LoadedTransformerArtifact(const LoadedTransformerArtifact&) = delete;
    LoadedTransformerArtifact& operator = (const LoadedTransformerArtifact&) = delete;

    LoadedTransformerArtifact(LoadedTransformerArtifact&&) noexcept = default;
    LoadedTransformerArtifact& operator = (LoadedTransformerArtifact&&) noexcept = default;

    ~LoadedTransformerArtifact() = default;
};

}  // namespace inference::artifacts::npz
