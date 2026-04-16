#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include "inference/artifacts/npz/state_dict_loader.hpp"
#include "inference/transformer_core/vision.hpp"

using Json = nlohmann::json;

std::unique_ptr<inference::transformer_core::Model> CreateModel(const std::string& model_type, const Json& config_json) {
    if (model_type == "vision") {
        inference::transformer_core::VisionTransformerConfig config = config_json;
        return std::make_unique<inference::transformer_core::VisionTransformer>(config);
    } else if (model_type == "text") {
        inference::transformer_core::TextTransformerConfig config = config_json;
        return std::make_unique<inference::transformer_core::TextTransformer>(config);
    } else {
        throw std::invalid_argument("Unknown model type: " + model_type);
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc < 7)
        {
            std::cerr << "Usage: inference --model <model_type> --config <config.json> --checkpoint <checkpoint_path> --input <input_path>" << std::endl;
            return 1;
        }

        std::string model_type;
        std::string config_path;
        std::string checkpoint_path;
        std::string input_path;

        for (int i = 1; i < argc; ++i)
        {
            if (std::string(argv[i]) == "--model" && i + 1 < argc)
            {
                model_type = argv[i + 1];
                ++i;
            }
            else if (std::string(argv[i]) == "--config" && i + 1 < argc)
            {
                config_path = argv[i + 1];
                ++i;
            }
            else if (std::string(argv[i]) == "--checkpoint" && i + 1 < argc)
            {
                checkpoint_path = argv[i + 1];
                ++i;
            }
            else if (std::string(argv[i]) == "--input" && i + 1 < argc)
            {
                input_path = argv[i + 1];
                ++i;
            }
        }

        if (model_type.empty() || config_path.empty() || checkpoint_path.empty() || input_path.empty())
        {
            std::cerr << "All --model, --config, --checkpoint, and --input must be provided." << std::endl;
            return 1;
        }

        // Load config
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            std::cerr << "Failed to open config file: " << config_path << std::endl;
            return 1;
        }
        Json config_json;
        config_file >> config_json;

        // Create model
        std::unique_ptr<inference::transformer_core::Model> model = CreateModel(model_type, config_json);

        // Load checkpoint
        const auto npz_data = cnpy::npz_load(checkpoint_path);
        const auto state_dict = inference::artifacts::npz::LoadStateDict(npz_data);

        // Load parameters
        model->LoadParameters(state_dict, "");

        // Load input
        auto input_data = cnpy::npz_load(input_path);
        const auto& input_array = input_data[model_type == "vision" ? "image" : "input"];
        std::vector<std::int64_t> shape;
        for (auto s : input_array.shape) shape.push_back(static_cast<std::int64_t>(s));
        inference::transformer_core::Tensor input_tensor(shape, 0.0F);
        const float* data = input_array.data<float>();
        for (std::size_t i = 0; i < input_tensor.numel(); ++i)
        {
            input_tensor.flat(i) = data[i];
        }

        // Run forward
        const auto output = model->Forward(input_tensor);

        std::cout << "Inference successful!" << std::endl;
        std::cout << "Model type: " << model_type << std::endl;
        std::cout << "Input shape: ";
        for (auto s : input_tensor.shape()) std::cout << s << " ";
        std::cout << std::endl;
        std::cout << "Output shape: ";
        for (auto s : output.shape()) std::cout << s << " ";
        std::cout << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}