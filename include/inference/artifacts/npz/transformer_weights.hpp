#pragma once

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

struct TransformerFeedForwardWeights
{
    Eigen::Map<const MatrixRM> fc1_weight;
    Eigen::Map<const Vector>   fc1_bias;
    Eigen::Map<const MatrixRM> fc2_weight;
    Eigen::Map<const Vector>   fc2_bias;
};

struct TransformerResidualAttention
{
    TransformerAttentionWeights module;
    Eigen::Map<const Vector>    norm_weight;
    Eigen::Map<const Vector>    norm_bias;
};

struct TransformerResidualFeedForward
{
    TransformerFeedForwardWeights module;
    Eigen::Map<const Vector>      norm_weight;
    Eigen::Map<const Vector>      norm_bias;
};

struct TransformerEncoderLayerWeights
{
    TransformerAttentionWeights    attention;
    TransformerFeedForwardWeights  ff;
    TransformerResidualAttention   residue1;
    TransformerResidualFeedForward residue2;
};

struct TransformerModelWeights
{
    std::map<std::string, cnpy::NpyArray> named;
    std::vector<TransformerEncoderLayerWeights> encoder;

    std::unique_ptr<Eigen::Map<const MatrixRM>> cls_token;
    std::unique_ptr<Eigen::Map<const MatrixRM>> token_embedding_weight;
    std::unique_ptr<Eigen::Map<const MatrixRM>> position_positional_table;
    std::unique_ptr<Eigen::Map<const Vector>>   norm_weight;
    std::unique_ptr<Eigen::Map<const Vector>>   norm_bias;

    std::unique_ptr<Eigen::Map<const MatrixRM>> head0_weight;
    std::unique_ptr<Eigen::Map<const Vector>>   head0_bias;
    std::unique_ptr<Eigen::Map<const MatrixRM>> head3_weight;
    std::unique_ptr<Eigen::Map<const Vector>>   head3_bias;

    std::unordered_map<std::string, std::unique_ptr<Eigen::Map<const MatrixRM>>> head_weights;
    std::unordered_map<std::string, std::unique_ptr<Eigen::Map<const Vector>>>   head_biases;
};

struct LoadedTransformerArtifact
{
    core::ArtifactSpec               artifact;
    Json                             metadata;
    TransformerModelWeights          model_weights;
    std::unordered_map<std::string, int> vocab;

    LoadedTransformerArtifact() = default;

    LoadedTransformerArtifact(const LoadedTransformerArtifact&) = delete;
    LoadedTransformerArtifact& operator = (const LoadedTransformerArtifact&) = delete;

    LoadedTransformerArtifact(LoadedTransformerArtifact&&) noexcept = default;
    LoadedTransformerArtifact& operator = (LoadedTransformerArtifact&&) noexcept = default;

    ~LoadedTransformerArtifact() = default;
};

}  // namespace inference::artifacts::npz
