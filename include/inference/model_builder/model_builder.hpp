#pragma once

/// \file
/// \brief Registry-backed artifact model-builder interfaces.

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

/// \brief JSON alias used by artifact-backed model builders.
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

    /// \brief Register a builder callback for one stable model-type key.
    void Register(std::string model_type,
                  BuilderFn   builder);

    /// \brief Build one runtime model from explicit metadata plus a generic state dict.
    BuiltModel Build(const Json&                         metadata,
                     const transformer_core::StateDict& state_dict) const;

    /// \brief Build one runtime model from an already-loaded artifact bundle.
    BuiltModel Build(const artifacts::npz::LoadedStateDictArtifact& artifact) const;

    /// \brief Create the default registry with the built-in encoder-classifier and vision-detector builders.
    static ModelBuilderRegistry CreateDefault();

private:
    std::unordered_map<std::string, BuilderFn> builders_;
};

/// \brief Infer the builder key to use for one metadata/state-dict pair.
std::string InferModelType(const Json&                         metadata,
                           const transformer_core::StateDict& state_dict);

/// \brief Resolve one encoder-classifier config from artifact metadata and checkpoint tensors.
models::EncoderClassifierConfig ResolveEncoderClassifierConfig(const Json&                         metadata,
                                                              const transformer_core::StateDict& state_dict);

/// \brief Build the built-in encoder-classifier runtime model.
BuiltModel BuildEncoderClassifier(const Json&                         metadata,
                                  const transformer_core::StateDict& state_dict);

/// \brief Resolve one vision-detector config from artifact metadata and checkpoint tensors.
models::VisionDetectorConfig ResolveVisionDetectorConfig(const Json&                         metadata,
                                                        const transformer_core::StateDict& state_dict);

/// \brief Build the built-in vision-detector runtime model.
BuiltModel BuildVisionDetector(const Json&                         metadata,
                               const transformer_core::StateDict& state_dict);

}  // namespace inference::model_builder
