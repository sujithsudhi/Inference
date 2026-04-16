#include "inference/artifacts/npz/transformer_loader.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace inference::artifacts::npz
{
namespace
{

bool has_prefix(const std::string& value,
                const std::string& prefix)
{
    return value.rfind(prefix, 0) == 0;
}

bool is_digits(const std::string& value)
{
    return !value.empty() &&
           std::all_of(value.begin(),
                       value.end(),
                       [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::vector<size_t> collect_encoder_indices(const cnpy::npz_t& weights)
{
    std::set<size_t> indices;

    for (const auto& kv : weights)
    {
        const std::string& key = kv.first;
        if (!has_prefix(key, "encoder."))
        {
            continue;
        }

        const size_t start = std::string("encoder.").size();
        const size_t dot   = key.find('.', start);

        if (dot == std::string::npos)
        {
            continue;
        }

        const std::string idx_str = key.substr(start, dot - start);
        if (!is_digits(idx_str))
        {
            continue;
        }

        indices.insert(static_cast<size_t>(std::stoull(idx_str)));
    }

    return std::vector<size_t>(indices.begin(), indices.end());
}

bool check_required_keys(const cnpy::npz_t&              weights,
                         const std::vector<std::string>& keys,
                         const std::string&              context)
{
    bool ok = true;

    for (const auto& key : keys)
    {
        if (weights.find(key) == weights.end())
        {
            std::cerr << "Warning: missing key for " << context << ": " << key << "\n";
            ok = false;
        }
    }

    return ok;
}

const cnpy::NpyArray& require_array(const cnpy::npz_t& weights,
                                    const std::string& name)
{
    const auto it = weights.find(name);
    if (it == weights.end())
    {
        throw std::runtime_error("Missing array in NPZ: " + name);
    }
    return it->second;
}

const Json& require_model_config(const Json& metadata)
{
    if (metadata.contains("config") &&
        metadata["config"].is_object() &&
        metadata["config"].contains("model"))
    {
        return metadata["config"]["model"];
    }

    if (metadata.contains("model") && metadata["model"].is_object())
    {
        return metadata["model"];
    }

    throw std::runtime_error("Artifact metadata does not contain a model config.");
}

Eigen::Map<const MatrixRM> map_matrix(const cnpy::NpyArray& arr,
                                      int                   expected_rows,
                                      int                   expected_cols,
                                      const std::string&    name)
{
    if (arr.shape.size() == 2)
    {
        if (expected_rows > 0 && expected_cols > 0)
        {
            if (arr.shape[0] != static_cast<size_t>(expected_rows) ||
                arr.shape[1] != static_cast<size_t>(expected_cols))
            {
                throw std::runtime_error("Unexpected shape for " + name + ": expected [" +
                                         std::to_string(expected_rows) + "," +
                                         std::to_string(expected_cols) + "], got [" +
                                         std::to_string(arr.shape[0]) + "," +
                                         std::to_string(arr.shape[1]) + "]");
            }
        }

        return Eigen::Map<const MatrixRM>(arr.data<float>(),
                                          static_cast<Eigen::Index>(arr.shape[0]),
                                          static_cast<Eigen::Index>(arr.shape[1]));
    }

    if (arr.shape.size() > 2 && arr.shape[0] == 1)
    {
        size_t cols = 1;

        for (size_t i = 2; i < arr.shape.size(); ++i)
        {
            cols *= arr.shape[i];
        }

        if (expected_rows > 0 && expected_cols > 0)
        {
            if (arr.shape[1] != static_cast<size_t>(expected_rows) ||
                cols != static_cast<size_t>(expected_cols))
            {
                throw std::runtime_error("Unexpected shape for " + name + ": expected [" +
                                         std::to_string(expected_rows) + "," +
                                         std::to_string(expected_cols) + "], got [" +
                                         std::to_string(arr.shape[1]) + "," +
                                         std::to_string(cols) + "]");
            }
        }

        return Eigen::Map<const MatrixRM>(arr.data<float>(),
                                          static_cast<Eigen::Index>(arr.shape[1]),
                                          static_cast<Eigen::Index>(cols));
    }

    throw std::runtime_error("Expected 2D array for matrix mapping for " + name +
                             ", got size " + std::to_string(arr.shape.size()));
}

Eigen::Map<const Vector> map_vector(const cnpy::NpyArray& arr,
                                    int                   expected_size,
                                    const std::string&    name)
{
    if (arr.shape.size() != 1)
    {
        throw std::runtime_error("Expected 1D array for vector mapping for " + name +
                                 ", got size " + std::to_string(arr.shape.size()));
    }

    if (expected_size > 0 && arr.shape[0] != static_cast<size_t>(expected_size))
    {
        throw std::runtime_error("Unexpected shape for " + name + ": expected [" +
                                 std::to_string(expected_size) + "], got [" +
                                 std::to_string(arr.shape[0]) + "]");
    }

    return Eigen::Map<const Vector>(arr.data<float>(),
                                    static_cast<Eigen::Index>(arr.num_vals));
}

TransformerAttentionWeights load_attention(const cnpy::npz_t& weights,
                                           const std::string& prefix,
                                           int                embed_dim)
{
    return TransformerAttentionWeights{
        map_matrix(require_array(weights, prefix + ".Wq.weight"),
                   embed_dim,
                   embed_dim,
                   prefix + ".Wq.weight"),
        map_vector(require_array(weights, prefix + ".Wq.bias"),
                   embed_dim,
                   prefix + ".Wq.bias"),
        map_matrix(require_array(weights, prefix + ".Wk.weight"),
                   embed_dim,
                   embed_dim,
                   prefix + ".Wk.weight"),
        map_vector(require_array(weights, prefix + ".Wk.bias"),
                   embed_dim,
                   prefix + ".Wk.bias"),
        map_matrix(require_array(weights, prefix + ".Wv.weight"),
                   embed_dim,
                   embed_dim,
                   prefix + ".Wv.weight"),
        map_vector(require_array(weights, prefix + ".Wv.bias"),
                   embed_dim,
                   prefix + ".Wv.bias"),
        map_matrix(require_array(weights, prefix + ".Wo.weight"),
                   embed_dim,
                   embed_dim,
                   prefix + ".Wo.weight"),
        map_vector(require_array(weights, prefix + ".Wo.bias"),
                   embed_dim,
                   prefix + ".Wo.bias"),
    };
}

TransformerFeedForwardWeights load_ff(const cnpy::npz_t& weights,
                                      const std::string& prefix,
                                      int                embed_dim,
                                      int                ff_dim)
{
    return TransformerFeedForwardWeights{
        map_matrix(require_array(weights, prefix + ".fullyConnected1.weight"),
                   ff_dim,
                   embed_dim,
                   prefix + ".fullyConnected1.weight"),
        map_vector(require_array(weights, prefix + ".fullyConnected1.bias"),
                   ff_dim,
                   prefix + ".fullyConnected1.bias"),
        map_matrix(require_array(weights, prefix + ".fullyConnected2.weight"),
                   embed_dim,
                   ff_dim,
                   prefix + ".fullyConnected2.weight"),
        map_vector(require_array(weights, prefix + ".fullyConnected2.bias"),
                   embed_dim,
                   prefix + ".fullyConnected2.bias"),
    };
}

TransformerResidualAttention load_residual_attention(const TransformerAttentionWeights& module,
                                                     const cnpy::npz_t&                weights,
                                                     const std::string&                prefix,
                                                     int                               embed_dim)
{
    return TransformerResidualAttention{
        module,
        map_vector(require_array(weights, prefix + ".norm.weight"),
                   embed_dim,
                   prefix + ".norm.weight"),
        map_vector(require_array(weights, prefix + ".norm.bias"),
                   embed_dim,
                   prefix + ".norm.bias"),
    };
}

TransformerResidualFeedForward load_residual_ff(const TransformerFeedForwardWeights& module,
                                                const cnpy::npz_t&                  weights,
                                                const std::string&                  prefix,
                                                int                                 embed_dim)
{
    return TransformerResidualFeedForward{
        module,
        map_vector(require_array(weights, prefix + ".norm.weight"),
                   embed_dim,
                   prefix + ".norm.weight"),
        map_vector(require_array(weights, prefix + ".norm.bias"),
                   embed_dim,
                   prefix + ".norm.bias"),
    };
}

void require_artifact_paths(const core::ArtifactSpec& artifact)
{
    if (artifact.metadata_path.empty())
    {
        throw std::runtime_error("Artifact is missing a metadata path.");
    }
    if (artifact.weights_path.empty())
    {
        throw std::runtime_error("Artifact is missing a weights path.");
    }
}

}  // namespace

Json LoadJson(const std::string& path)
{
    std::ifstream handle(path);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }

    Json payload;
    handle >> payload;
    return payload;
}

cnpy::npz_t LoadNpz(const std::string& path)
{
    return cnpy::npz_load(path);
}

std::unordered_map<std::string, int> LoadVocab(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open vocab file: " + path);
    }

    Json vocab_json;
    file >> vocab_json;

    std::unordered_map<std::string, int> vocab;

    if (vocab_json.contains("model") && vocab_json["model"].contains("vocab"))
    {
        const auto& vocab_obj = vocab_json["model"]["vocab"];
        for (const auto& [token, id] : vocab_obj.items())
        {
            vocab[token] = id.get<int>();
        }
    }
    else if (vocab_json.is_object())
    {
        for (const auto& [token, id] : vocab_json.items())
        {
            if (id.is_number_integer())
            {
                vocab[token] = id.get<int>();
            }
        }
    }
    else
    {
        throw std::runtime_error("Unsupported vocab format in: " + path);
    }

    return vocab;
}

TransformerModelWeights LoadTransformerWeights(const cnpy::npz_t& weights,
                                               const Json&        metadata)
{
    TransformerModelWeights model;
    model.named = weights;

    model.encoder.clear();
    model.head_weights.clear();
    model.head_biases.clear();

    const auto& model_cfg  = require_model_config(metadata);
    const int embed_dim    = model_cfg.value("embed_dim", 0);
    const double mlp_ratio = model_cfg.value("mlp_ratio", 0.0);
    const int max_length   = model_cfg.value("max_length", 0);
    const int vocab_size   = model_cfg.value("vocab_size", 0);
    const int ff_dim       = (embed_dim > 0 && mlp_ratio > 0.0)
        ? static_cast<int>(std::lround(embed_dim * mlp_ratio))
        : 0;
    const int depth        = model_cfg.value("depth", 0);

    auto set_optional_matrix = [&](std::unique_ptr<Eigen::Map<const MatrixRM>>& target,
                                   const std::string&                           key,
                                   int                                          expected_rows,
                                   int                                          expected_cols)
    {
        const auto it = weights.find(key);
        if (it != weights.end())
        {
            target = std::make_unique<Eigen::Map<const MatrixRM>>(map_matrix(it->second,
                                                                             expected_rows,
                                                                             expected_cols,
                                                                             key));
        }
        else
        {
            std::cerr << "Warning: missing key in NPZ: " << key << "\n";
        }
    };

    auto set_optional_vector = [&](std::unique_ptr<Eigen::Map<const Vector>>& target,
                                   const std::string&                         key,
                                   int                                        expected_size)
    {
        const auto it = weights.find(key);
        if (it != weights.end())
        {
            target = std::make_unique<Eigen::Map<const Vector>>(map_vector(it->second,
                                                                           expected_size,
                                                                           key));
        }
        else
        {
            std::cerr << "Warning: missing key in NPZ: " << key << "\n";
        }
    };

    const int expected_vocab     = vocab_size > 0 ? vocab_size : 0;
    const int expected_positions = max_length > 0 ? max_length + 1 : 0;

    set_optional_matrix(model.cls_token,
                        "cls_token",
                        embed_dim > 0 ? 1 : 0,
                        embed_dim);

    set_optional_matrix(model.token_embedding_weight,
                        "token_embedding.weight",
                        expected_vocab,
                        embed_dim);

    set_optional_matrix(model.position_positional_table,
                        "position.positional_table",
                        expected_positions,
                        embed_dim);

    set_optional_vector(model.norm_weight, "norm.weight", embed_dim);
    set_optional_vector(model.norm_bias,   "norm.bias",   embed_dim);

    const std::vector<size_t> found_layers = collect_encoder_indices(weights);

    if (depth > 0 && found_layers.size() != static_cast<size_t>(depth))
    {
        std::cerr << "Warning: depth=" << depth
                  << " but found " << found_layers.size()
                  << " encoder blocks in NPZ\n";
    }

    const auto load_layer = [&](size_t index)
    {
        const std::string layer_prefix = "encoder." + std::to_string(index);
        const std::vector<std::string> required = {
            layer_prefix + ".attention.Wq.weight",
            layer_prefix + ".attention.Wq.bias",
            layer_prefix + ".attention.Wk.weight",
            layer_prefix + ".attention.Wk.bias",
            layer_prefix + ".attention.Wv.weight",
            layer_prefix + ".attention.Wv.bias",
            layer_prefix + ".attention.Wo.weight",
            layer_prefix + ".attention.Wo.bias",
            layer_prefix + ".ff.fullyConnected1.weight",
            layer_prefix + ".ff.fullyConnected1.bias",
            layer_prefix + ".ff.fullyConnected2.weight",
            layer_prefix + ".ff.fullyConnected2.bias",
            layer_prefix + ".residue1.norm.weight",
            layer_prefix + ".residue1.norm.bias",
            layer_prefix + ".residue2.norm.weight",
            layer_prefix + ".residue2.norm.bias",
        };

        if (!check_required_keys(weights, required, layer_prefix))
        {
            std::cerr << "Warning: skipping encoder layer " << index
                      << " due to missing keys\n";
            return;
        }

        const auto attention = load_attention(weights,
                                              layer_prefix + ".attention",
                                              embed_dim);

        const auto ff = load_ff(weights,
                                layer_prefix + ".ff",
                                embed_dim,
                                ff_dim);

        model.encoder.push_back(TransformerEncoderLayerWeights{
            attention,
            ff,
            load_residual_attention(attention,
                                    weights,
                                    layer_prefix + ".residue1",
                                    embed_dim),
            load_residual_ff(ff,
                             weights,
                             layer_prefix + ".residue2",
                             embed_dim),
        });
    };

    if (depth > 0)
    {
        for (size_t i = 0; i < static_cast<size_t>(depth); ++i)
        {
            load_layer(i);
        }
    }
    else
    {
        for (const size_t index : found_layers)
        {
            load_layer(index);
        }
    }

    const std::unordered_set<std::string> non_head_keys = {
        "cls_token",
        "token_embedding.weight",
        "position.positional_table",
        "norm.weight",
        "norm.bias",
    };

    for (const auto& kv : weights)
    {
        const std::string& key = kv.first;

        if (has_prefix(key, "encoder.") || non_head_keys.count(key) != 0)
        {
            continue;
        }

        if (key == "head.0.weight")
        {
            model.head0_weight = std::make_unique<Eigen::Map<const MatrixRM>>(map_matrix(kv.second,
                                                                                          embed_dim,
                                                                                          embed_dim,
                                                                                          key));
        }
        else if (key == "head.0.bias")
        {
            model.head0_bias = std::make_unique<Eigen::Map<const Vector>>(map_vector(kv.second,
                                                                                      embed_dim,
                                                                                      key));
        }
        else if (key == "head.3.weight")
        {
            model.head3_weight = std::make_unique<Eigen::Map<const MatrixRM>>(map_matrix(kv.second,
                                                                                          1,
                                                                                          embed_dim,
                                                                                          key));
        }
        else if (key == "head.3.bias")
        {
            model.head3_bias = std::make_unique<Eigen::Map<const Vector>>(map_vector(kv.second,
                                                                                      1,
                                                                                      key));
        }

        if (key.size() >= 7 && key.rfind(".weight") == key.size() - 7)
        {
            if (kv.second.shape.size() == 2 ||
                (kv.second.shape.size() > 2 && kv.second.shape[0] == 1))
            {
                model.head_weights[key] = std::make_unique<Eigen::Map<const MatrixRM>>(map_matrix(kv.second,
                                                                                                    0,
                                                                                                    0,
                                                                                                    key));
            }
            else
            {
                std::cerr << "Warning: expected 2D weight for " << key << "\n";
            }
        }
        else if (key.size() >= 5 && key.rfind(".bias") == key.size() - 5)
        {
            if (kv.second.shape.size() == 1)
            {
                model.head_biases[key] = std::make_unique<Eigen::Map<const Vector>>(map_vector(kv.second,
                                                                                                0,
                                                                                                key));
            }
            else
            {
                std::cerr << "Warning: expected 1D bias for " << key << "\n";
            }
        }
    }

    if (!model.head0_weight)
    {
        std::cerr << "Warning: missing key in NPZ: head.0.weight\n";
    }
    if (!model.head0_bias)
    {
        std::cerr << "Warning: missing key in NPZ: head.0.bias\n";
    }
    if (!model.head3_weight)
    {
        std::cerr << "Warning: missing key in NPZ: head.3.weight\n";
    }
    if (!model.head3_bias)
    {
        std::cerr << "Warning: missing key in NPZ: head.3.bias\n";
    }

    return model;
}

std::vector<std::string> FindKeysWithPrefix(const TransformerModelWeights& weights,
                                            const std::string&             prefix)
{
    std::vector<std::string> matches;

    for (const auto& kv : weights.named)
    {
        if (kv.first.rfind(prefix, 0) == 0)
        {
            matches.push_back(kv.first);
        }
    }

    std::sort(matches.begin(), matches.end());
    return matches;
}

const cnpy::NpyArray* FindArray(const TransformerModelWeights& weights,
                                const std::string&             key)
{
    const auto it = weights.named.find(key);
    if (it == weights.named.end())
    {
        return nullptr;
    }

    return &it->second;
}

LoadedTransformerArtifact LoadTransformerArtifact(const core::ArtifactBundle& artifact)
{
    return LoadTransformerArtifact(artifact.Inspect());
}

LoadedTransformerArtifact LoadTransformerArtifact(const core::ArtifactSpec& artifact)
{
    require_artifact_paths(artifact);

    LoadedTransformerArtifact loaded;
    loaded.artifact      = artifact;
    loaded.metadata      = LoadJson(artifact.metadata_path.string());
    const cnpy::npz_t weights = LoadNpz(artifact.weights_path.string());
    loaded.model_weights = LoadTransformerWeights(weights, loaded.metadata);

    if (!artifact.tokenizer_path.empty())
    {
        loaded.vocab = LoadVocab(artifact.tokenizer_path.string());
    }

    return loaded;
}

}  // namespace inference::artifacts::npz
