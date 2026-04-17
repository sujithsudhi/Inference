#include <cstdio>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "inference/core/artifact.hpp"
#include "inference/runtime/model_runner.hpp"

namespace
{

using inference::transformer_core::IndexTensor;
using inference::transformer_core::Tensor;

void Expect(bool condition,
            const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

nlohmann::json LoadJson(const std::filesystem::path& path)
{
    std::ifstream handle(path);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open JSON file: " + path.string());
    }

    nlohmann::json payload;
    handle >> payload;
    return payload;
}

IndexTensor LoadIndexTensor(const nlohmann::json& values)
{
    const std::int64_t rows = static_cast<std::int64_t>(values.size());
    const std::int64_t cols = rows > 0 ? static_cast<std::int64_t>(values[0].size()) : 0;

    IndexTensor tensor({rows, cols}, 0);
    for (std::int64_t row = 0; row < rows; ++row)
    {
        for (std::int64_t col = 0; col < cols; ++col)
        {
            tensor.at({row, col}) = values[row][col].get<std::int64_t>();
        }
    }

    return tensor;
}

Tensor LoadFloatTensor(const nlohmann::json& values)
{
    const std::int64_t rows = static_cast<std::int64_t>(values.size());
    const std::int64_t cols = rows > 0 ? static_cast<std::int64_t>(values[0].size()) : 0;

    Tensor tensor({rows, cols}, 0.0F);
    for (std::int64_t row = 0; row < rows; ++row)
    {
        for (std::int64_t col = 0; col < cols; ++col)
        {
            tensor.at({row, col}) = values[row][col].get<float>();
        }
    }

    return tensor;
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            throw std::runtime_error("Usage: imdb_encoder_classifier_e2e_test <artifact-dir>");
        }

        const std::filesystem::path artifact_dir = std::filesystem::path(argv[1]);
        const std::filesystem::path sample_path  = artifact_dir / "sample.json";

        if (!std::filesystem::exists(artifact_dir / "artifact.json") || !std::filesystem::exists(sample_path))
        {
            std::printf("Skipping IMDB e2e test: fixture files are not present.\n");
            return 0;
        }

        inference::core::ArtifactBundle bundle(artifact_dir);
        inference::runtime::ModelRunner runner;
        const auto                     load_status = runner.Load(bundle);
        Expect(load_status.ok(),
               "IMDB fixture should load through ModelRunner.");
        Expect(runner.HasEncoderClassifier(),
               "IMDB fixture should resolve to an encoder-classifier model.");

        const auto sample = LoadJson(sample_path);
        const IndexTensor inputs         = LoadIndexTensor(sample["inputs"]);
        const Tensor      attention_mask = LoadFloatTensor(sample["attention_mask"]);
        const Tensor      logits         = runner.RunEncoderClassifier(inputs, attention_mask);

        Expect(logits.shape() == std::vector<std::int64_t>({1, 1}),
               "IMDB end-to-end logits shape mismatch.");

        const float expected = sample["expected_logits"][0][0].get<float>();
        const float actual   = logits.at({0, 0});

        Expect(std::fabs(actual - expected) < 5e-2F,
               "IMDB end-to-end logit mismatch. Expected "
               + std::to_string(expected)
               + " but received "
               + std::to_string(actual)
               + ".");

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
