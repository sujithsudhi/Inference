#pragma once

#include <optional>
#include <string>

#include "inference/artifacts/npz/state_dict_loader.hpp"
#include "inference/core/artifact.hpp"
#include "inference/core/status.hpp"
#include "inference/model_builder/model_builder.hpp"
#include "inference/models/encoder_classifier.hpp"
#include "inference/models/vision_detector.hpp"
#include "inference/transformer_core/tensor.hpp"

namespace inference::runtime
{

/// Runtime boundary that owns a built model and exposes execution-only entry points.
class ModelRunner
{
public:
    /// Construct one runner with the default registry-backed model builder.
    ModelRunner();

    /// Construct one runner with an explicitly configured model-builder registry.
    explicit ModelRunner(model_builder::ModelBuilderRegistry registry);

    /// Load one artifact bundle, resolve it through the model builder, and retain the runnable model.
    core::Status Load(const core::ArtifactBundle& artifact);

    /// Load one already-decoded state-dict artifact and retain the runnable model.
    core::Status Load(const artifacts::npz::LoadedStateDictArtifact& artifact);

    /// Indicate whether the runner currently owns a loaded model.
    bool loaded() const noexcept;

    /// Return the stable model type of the currently loaded model, or an empty string when unloaded.
    const std::string& model_type() const noexcept;

    /// Indicate whether the loaded model is an encoder classifier.
    bool HasEncoderClassifier() const noexcept;

    /// Indicate whether the loaded model is a vision detector.
    bool HasVisionDetector() const noexcept;

    /// Access the loaded encoder-classifier config.
    const models::EncoderClassifierConfig& encoder_classifier_config() const;

    /// Access the loaded vision-detector config.
    const models::VisionDetectorConfig& vision_detector_config() const;

    /// Run the loaded encoder-classifier model.
    transformer_core::Tensor RunEncoderClassifier(
        const transformer_core::IndexTensor&          inputs,
        const std::optional<transformer_core::Tensor>& attention_mask = std::nullopt) const;

    /// Run only the loaded vision-detector backbone.
    models::VisionBackboneOutput RunVisionBackbone(const transformer_core::Tensor& images) const;

    /// Run the loaded vision-detector model.
    models::VisionDetectionOutput RunVisionDetector(const transformer_core::Tensor& images) const;

private:
    void Reset() noexcept;

    models::EncoderClassifier& RequireEncoderClassifier() const;

    models::VisionDetector& RequireVisionDetector() const;

    model_builder::ModelBuilderRegistry          registry_;
    std::string                                  model_type_;
    std::unique_ptr<models::EncoderClassifier>   encoder_classifier_;
    std::unique_ptr<models::VisionDetector>      vision_detector_;
};

}  // namespace inference::runtime
