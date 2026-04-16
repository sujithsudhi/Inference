#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cnpy.h>

#include "inference/artifacts/npz/state_dict_loader.hpp"

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

        cnpy::npz_save(npz_path.string(), "linear.weight", linear_weight.data(), {2UL, 3UL}, "w");
        cnpy::npz_save(npz_path.string(), "linear.bias", linear_bias.data(), {2UL}, "a");

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

        std::filesystem::remove_all(temp_dir);
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
