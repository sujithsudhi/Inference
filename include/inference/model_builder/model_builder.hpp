#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <nlohmann/json.hpp>

#include "inference/artifacts/npz/state_dict_loader.hpp"
#include "inference/models/encoder_classifier.hpp"
#include "inference/models/vision_detector.hpp"

namespace inference::model_builder
{

using Json = nlohmann::json;

/// Result of building a concrete runtime model from metadata plus a state dict.
struct BuiltModel
{
    std::string                                model_type;
    std::unique_ptr<models::EncoderClassifier> encoder_classifier;
    std::unique_ptr<models::VisionDetector>    vision_detector;

    /// Indicate whether the built model contains an encoder classifier instance.
    bool HasEncoderClassifier() const noexcept;

    /// Indicate whether the built model contains a vision detector instance.
    bool HasVisionDetector() const noexcept;
};

/// Registry of model factories keyed by a stable model type string.
class ModelBuilderRegistry
{
public:
    /// Builder callback signature used for registered model factories.
    using BuilderFn = std::function<BuiltModel(const Json&, const transformer_core::StateDict&)>;

    /// Register a builder for one model type.
    void Register(std::string model_type,
                  BuilderFn   builder);

    /// Build a model from metadata and a generic state dict.
    BuiltModel Build(const Json&                         metadata,
                     const transformer_core::StateDict& state_dict) const;

    /// Build a model from a loaded state-dict artifact.
    BuiltModel Build(const artifacts::npz::LoadedStateDictArtifact& artifact) const;

    /// Create the default registry with the built-in transformer model builders.
    static ModelBuilderRegistry CreateDefault();

private:
    std::unordered_map<std::string, BuilderFn> builders_;
};

/// Infer the builder key to use for a given metadata/state_dict pair.
std::string InferModelType(const Json&                         metadata,
                           const transformer_core::StateDict& state_dict);

/// Resolve the encoder-classifier configuration from metadata and checkpoint tensors.
models::EncoderClassifierConfig ResolveEncoderClassifierConfig(const Json&                         metadata,
                                                              const transformer_core::StateDict& state_dict);

/// Build the built-in encoder-classifier model from metadata and checkpoint tensors.
BuiltModel BuildEncoderClassifier(const Json&                         metadata,
                                  const transformer_core::StateDict& state_dict);

/// Resolve the vision-detector configuration from metadata and checkpoint tensors.
models::VisionDetectorConfig ResolveVisionDetectorConfig(const Json&                         metadata,
                                                        const transformer_core::StateDict& state_dict);

/// Build the built-in vision detector from metadata and checkpoint tensors.
BuiltModel BuildVisionDetector(const Json&                         metadata,
                               const transformer_core::StateDict& state_dict);

}  // namespace inference::model_builder
