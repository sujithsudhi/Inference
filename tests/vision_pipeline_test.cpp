#include <iostream>
#include <filesystem>

#include <cnpy.h>

#include "inference/artifacts/npz/state_dict_loader.hpp"
#include "inference/transformer_core/vision.hpp"

int main()
{
    try
    {
        const auto test_dir = std::filesystem::path("test_vision");
        const auto npz_path = test_dir / "weights.npz";
        const auto image_npz = test_dir / "image.npz";

        if (!std::filesystem::exists(npz_path) || !std::filesystem::exists(image_npz))
        {
            std::cout << "Skipping vision pipeline test: local fixture files are not present." << std::endl;
            return 0;
        }

        // Load state dict
        const auto npz_data = cnpy::npz_load(npz_path.string());
        const auto state_dict = inference::artifacts::npz::LoadStateDict(npz_data);

        // Create PatchEmbedding
        inference::transformer_core::PatchEmbedding patch_embed(224, 16, 3, 1024, true);

        // Load parameters
        patch_embed.LoadParameters(state_dict, "patch_embed.");

        // Load image tensor
        auto image_data = cnpy::npz_load(image_npz.string());
        const auto& image_array = image_data["image"];
        std::vector<std::int64_t> shape(image_array.shape.begin(), image_array.shape.end());
        const float* data = image_array.data<float>();
        inference::transformer_core::Tensor image_tensor(shape, 0.0F);
        for (std::size_t i = 0; i < image_tensor.numel(); ++i)
        {
            image_tensor.flat(i) = data[i];
        }

        // Run forward
        const auto output = patch_embed.Forward(image_tensor);

        std::cout << "Input shape: ";
        for (auto s : image_tensor.shape()) std::cout << s << " ";
        std::cout << std::endl;
        std::cout << "Output shape: ";
        for (auto s : output.shape()) std::cout << s << " ";
        std::cout << std::endl;
        std::cout << "First few output values: ";
        for (size_t i = 0; i < 10; ++i) {
            std::cout << output.flat(i) << " ";
        }
        std::cout << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
