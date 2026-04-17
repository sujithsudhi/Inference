/// \file
/// \brief Generic NPZ state-dict artifact loader implementation.

#include "inference/artifacts/npz/state_dict_loader.hpp"

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
    loaded.metadata   = LoadJson(artifact.metadata_path.string());
    loaded.state_dict = LoadStateDict(LoadNpz(artifact.weights_path.string()));

    if (!artifact.tokenizer_path.empty())
    {
        loaded.vocab = LoadVocab(artifact.tokenizer_path.string());
    }

    return loaded;
}

}  // namespace inference::artifacts::npz
