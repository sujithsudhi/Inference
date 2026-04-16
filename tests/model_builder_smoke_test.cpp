#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "inference/model_builder.hpp"

namespace
{

using inference::model_builder::BuildEncoderClassifier;
using inference::model_builder::InferModelType;
using inference::model_builder::ModelBuilderRegistry;
using inference::models::EncoderClassifier;
using inference::models::EncoderClassifierConfig;
using inference::transformer_core::IndexTensor;
using inference::transformer_core::StateDict;
using inference::transformer_core::Tensor;
using inference::transformer_core::TensorSpec;

void Expect(bool condition,
            const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

StateDict BuildStateDict(const std::vector<TensorSpec>& specs)
{
    StateDict state_dict;
    float     next_value = 0.01F;

    for (const auto& spec : specs)
    {
        Tensor tensor(spec.shape, 0.0F);
        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            tensor.flat(i) = next_value;
            next_value += 0.01F;
        }
        state_dict.emplace(spec.name, std::move(tensor));
    }

    return state_dict;
}

}  // namespace

int main()
{
    try
    {
        EncoderClassifierConfig config;
        config.vocab_size        = 16;
        config.max_length        = 5;
        config.embed_dim         = 8;
        config.depth             = 2;
        config.num_heads         = 2;
        config.mlp_ratio         = 2.0F;
        config.mlp_hidden_dim    = 16;
        config.dropout           = 0.0F;
        config.attention_dropout = 0.0F;
        config.use_rope          = false;
        config.cls_head_dim      = 8;
        config.num_outputs       = 1;

        EncoderClassifier reference_model(config);
        StateDict         state_dict = BuildStateDict(reference_model.ParameterSpecs());

        nlohmann::json metadata = {
            {"builder", {{"model_type", "transformers.encoder_classifier"}}},
            {"config", {{"model", {{"vocab_size", 16},
                                   {"max_length", 5},
                                   {"embed_dim", 8},
                                   {"depth", 2},
                                   {"num_heads", 2},
                                   {"mlp_ratio", 2.0},
                                   {"mlp_hidden_dim", 16},
                                   {"dropout", 0.0},
                                   {"attention_dropout", 0.0},
                                   {"qkv_bias", true},
                                   {"pre_norm", true},
                                   {"layer_norm_eps", 1e-5},
                                   {"drop_path", 0.0},
                                   {"use_cls_token", true},
                                   {"cls_head_dim", 8},
                                   {"num_outputs", 1},
                                   {"pooling", "cls"},
                                   {"use_rope", false},
                                   {"rope_base", 10000}}}}}
        };

        Expect(InferModelType(metadata, state_dict) == "transformers.encoder_classifier",
               "InferModelType should resolve the encoder-classifier builder.");

        ModelBuilderRegistry registry = ModelBuilderRegistry::CreateDefault();
        auto built = registry.Build(metadata, state_dict);
        Expect(built.HasEncoderClassifier(),
               "Registry should build an encoder-classifier instance.");
        Expect(built.model_type == "transformers.encoder_classifier",
               "BuiltModel should preserve the builder key.");

        IndexTensor inputs({1, 4}, 0);
        inputs.at({0, 0}) = 1;
        inputs.at({0, 1}) = 2;
        inputs.at({0, 2}) = 3;

        Tensor attention_mask({1, 4}, 0.0F);
        attention_mask.at({0, 0}) = 1.0F;
        attention_mask.at({0, 1}) = 1.0F;
        attention_mask.at({0, 2}) = 1.0F;

        Tensor logits = built.encoder_classifier->Forward(inputs, attention_mask);
        Expect(logits.shape() == std::vector<std::int64_t>({1, 1}),
               "Encoder-classifier forward shape mismatch.");

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
