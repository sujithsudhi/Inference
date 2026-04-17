/// \file
/// \brief Generic NPZ state-dict artifact loader implementation.

#include "inference/artifacts/npz/state_dict_loader.hpp"

#include <fstream>
#include <stdexcept>

namespace inference::artifacts::npz
{

namespace
{

std::vector<std::int64_t> ToShape(const cnpy::NpyArray& array)
{
    std::vector<std::int64_t> shape;
    shape.reserve(array.shape.size());

    for (const auto dim : array.shape)
    {
        shape.push_back(static_cast<std::int64_t>(dim));
    }

    return shape;
}

template <typename T>
void CopyTensorData(transformer_core::Tensor& out,
                    const cnpy::NpyArray&     array)
{
    const T* src = array.data<T>();
    for (std::size_t i = 0; i < out.numel(); ++i)
    {
        out.flat(i) = static_cast<float>(src[i]);
    }
}

void RequireArtifactPaths(const core::ArtifactSpec& artifact)
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

Json LoadJsonDocument(const std::string& path)
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

cnpy::npz_t LoadNpzArchive(const std::string& path)
{
    return cnpy::npz_load(path);
}

std::unordered_map<std::string, int> LoadVocabMap(const std::string& path)
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

}  // namespace

transformer_core::Tensor ToTensor(const cnpy::NpyArray& array)
{
    transformer_core::Tensor out(ToShape(array), 0.0F);

    if (array.word_size == sizeof(float))
    {
        CopyTensorData<float>(out, array);
        return out;
    }

    if (array.word_size == sizeof(double))
    {
        CopyTensorData<double>(out, array);
        return out;
    }

    throw std::runtime_error("Unsupported NPY dtype in generic state dict loader.");
}

transformer_core::StateDict LoadStateDict(const cnpy::npz_t& weights)
{
    transformer_core::StateDict state_dict;

    for (const auto& kv : weights)
    {
        state_dict.emplace(kv.first, ToTensor(kv.second));
    }

    return state_dict;
}

LoadedStateDictArtifact LoadStateDictArtifact(const core::ArtifactBundle& artifact)
{
    return LoadStateDictArtifact(artifact.Inspect());
}

LoadedStateDictArtifact LoadStateDictArtifact(const core::ArtifactSpec& artifact)
{
    RequireArtifactPaths(artifact);

    LoadedStateDictArtifact loaded;
    loaded.artifact   = artifact;
    loaded.metadata   = LoadJsonDocument(artifact.metadata_path.string());
    loaded.state_dict = LoadStateDict(LoadNpzArchive(artifact.weights_path.string()));

    if (!artifact.tokenizer_path.empty())
    {
        loaded.vocab = LoadVocabMap(artifact.tokenizer_path.string());
    }

    return loaded;
}

}  // namespace inference::artifacts::npz
