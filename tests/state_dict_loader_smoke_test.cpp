/// \file
/// \brief Smoke test for the generic NPZ state-dict loader.

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include "inference/artifacts/npz/state_dict_loader.hpp"
#include "inference/core/artifact.hpp"

namespace
{

void Expect(bool condition,
            const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void WriteLittleEndianUint64(std::ofstream& handle,
                             const std::uint64_t value)
{
    for (std::size_t index = 0; index < 8; ++index)
    {
        const auto byte = static_cast<unsigned char>((value >> (index * 8U)) & 0xFFU);
        handle.put(static_cast<char>(byte));
    }
}

void WriteSafeTensorArtifact(const std::filesystem::path& root,
                             const std::vector<float>&    linear_weight,
                             const std::vector<float>&    linear_bias)
{
    std::filesystem::create_directories(root);

    const nlohmann::json manifest_json = {
        {"schema_version", "inference.artifact/1"},
        {"artifact_name",  root.filename().string()},
        {"model_family",   "test"},
        {"task",           "test"},
        {"weight_format",  "safetensors"},
        {"files",          {{"metadata", "model.json"},
                            {"weights",  "weights.safetensors"}}}
    };

    std::ofstream(root / "artifact.json") << manifest_json.dump(2) << "\n";
    std::ofstream(root / "model.json") << "{}\n";

    const std::uint64_t weight_bytes = static_cast<std::uint64_t>(linear_weight.size()) * sizeof(float);
    const std::uint64_t bias_bytes   = static_cast<std::uint64_t>(linear_bias.size()) * sizeof(float);

    nlohmann::json header_json = nlohmann::json::object();
    header_json["linear.weight"] = {{"dtype",        "F32"},
                                    {"shape",        {2, 3}},
                                    {"data_offsets", {0, weight_bytes}}};
    header_json["linear.bias"] = {{"dtype",        "F32"},
                                  {"shape",        {2}},
                                  {"data_offsets", {weight_bytes, weight_bytes + bias_bytes}}};

    const std::string header_payload = header_json.dump();

    std::ofstream handle(root / "weights.safetensors", std::ios::binary);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open safetensors smoke-test file for writing.");
    }

    WriteLittleEndianUint64(handle, static_cast<std::uint64_t>(header_payload.size()));
    handle.write(header_payload.data(),
                 static_cast<std::streamsize>(header_payload.size()));
    handle.write(reinterpret_cast<const char*>(linear_weight.data()),
                 static_cast<std::streamsize>(weight_bytes));
    handle.write(reinterpret_cast<const char*>(linear_bias.data()),
                 static_cast<std::streamsize>(bias_bytes));
}

}  // namespace

int main()
{
    try
    {
        const auto temp_dir = std::filesystem::temp_directory_path() / "inference_state_dict_loader_test";
        std::filesystem::create_directories(temp_dir);

        const auto npz_path = temp_dir / "weights.npz";
        std::filesystem::remove(npz_path);

        std::vector<float> linear_weight = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
        std::vector<float> linear_bias   = {0.5F, -0.5F};

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
        cnpy::npz_save(npz_path.string(), "linear.weight", linear_weight.data(), {2UL, 3UL}, "w");
        cnpy::npz_save(npz_path.string(), "linear.bias", linear_bias.data(), {2UL}, "a");
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

        const auto state_dict = inference::artifacts::npz::LoadStateDict(cnpy::npz_load(npz_path.string()));

        const auto weight_it = state_dict.find("linear.weight");
        const auto bias_it   = state_dict.find("linear.bias");

        Expect(weight_it != state_dict.end(), "Missing linear.weight in generic state dict.");
        Expect(bias_it != state_dict.end(), "Missing linear.bias in generic state dict.");

        Expect(weight_it->second.shape() == std::vector<std::int64_t>({2, 3}),
               "linear.weight shape mismatch.");
        Expect(bias_it->second.shape() == std::vector<std::int64_t>({2}),
               "linear.bias shape mismatch.");
        Expect(weight_it->second.at({1, 2}) == 6.0F,
               "linear.weight value mismatch.");
        Expect(bias_it->second.at({1}) == -0.5F,
               "linear.bias value mismatch.");

        const auto safetensors_root = temp_dir / "safetensors_bundle";
        std::filesystem::remove_all(safetensors_root);
        WriteSafeTensorArtifact(safetensors_root,
                                linear_weight,
                                linear_bias);

        const auto safetensors_artifact =
            inference::artifacts::npz::LoadStateDictArtifact(inference::core::ArtifactBundle(safetensors_root));

        const auto safetensors_weight_it = safetensors_artifact.state_dict.find("linear.weight");
        const auto safetensors_bias_it   = safetensors_artifact.state_dict.find("linear.bias");

        Expect(safetensors_weight_it != safetensors_artifact.state_dict.end(),
               "Missing linear.weight in safetensors state dict.");
        Expect(safetensors_bias_it != safetensors_artifact.state_dict.end(),
               "Missing linear.bias in safetensors state dict.");
        Expect(safetensors_weight_it->second.shape() == std::vector<std::int64_t>({2, 3}),
               "Safetensors linear.weight shape mismatch.");
        Expect(safetensors_bias_it->second.shape() == std::vector<std::int64_t>({2}),
               "Safetensors linear.bias shape mismatch.");
        Expect(safetensors_weight_it->second.at({1, 2}) == 6.0F,
               "Safetensors linear.weight value mismatch.");
        Expect(safetensors_bias_it->second.at({1}) == -0.5F,
               "Safetensors linear.bias value mismatch.");

        std::filesystem::remove_all(temp_dir);
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
