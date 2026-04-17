/// \file
/// \brief Checkpoint-backed model runner implementation.

#include "inference/runtime/model_runner.hpp"

#include <stdexcept>
#include <utility>

namespace inference::runtime
{

ModelRunner::ModelRunner()
    : ModelRunner(model_builder::ModelBuilderRegistry::CreateDefault())
{
}

ModelRunner::ModelRunner(model_builder::ModelBuilderRegistry registry)
    : registry_(std::move(registry))
{
}

core::Status ModelRunner::Load(const core::ArtifactBundle& artifact)
{
    if (!artifact.Exists())
    {
        Reset();
        return core::Status::NotFound("Artifact root does not exist: " + artifact.root().string());
    }

    try
    {
        return Load(artifacts::npz::LoadStateDictArtifact(artifact));
    }
    catch (const std::exception& ex)
    {
        Reset();
        return core::Status::InvalidArgument(ex.what());
    }
}

core::Status ModelRunner::Load(const artifacts::npz::LoadedStateDictArtifact& artifact)
{
    try
    {
        model_builder::BuiltModel built = registry_.Build(artifact);

        model_type_          = built.model_type;
        encoder_classifier_  = std::move(built.encoder_classifier);
        vision_detector_     = std::move(built.vision_detector);
        return core::Status::Ok();
    }
    catch (const std::exception& ex)
    {
        Reset();
        return core::Status::InvalidArgument(ex.what());
    }
}

bool ModelRunner::loaded() const noexcept
{
    return !model_type_.empty();
}

const std::string& ModelRunner::model_type() const noexcept
{
    return model_type_;
}

bool ModelRunner::HasEncoderClassifier() const noexcept
{
    return static_cast<bool>(encoder_classifier_);
}

bool ModelRunner::HasVisionDetector() const noexcept
{
    return static_cast<bool>(vision_detector_);
}

const models::EncoderClassifierConfig& ModelRunner::encoder_classifier_config() const
{
    return RequireEncoderClassifier().config();
}

const models::VisionDetectorConfig& ModelRunner::vision_detector_config() const
{
    return RequireVisionDetector().config();
}

transformer_core::Tensor ModelRunner::RunEncoderClassifier(
    const transformer_core::IndexTensor&          inputs,
    const std::optional<transformer_core::Tensor>& attention_mask) const
{
    return RequireEncoderClassifier().Forward(inputs, attention_mask);
}

models::VisionBackboneOutput ModelRunner::RunVisionBackbone(const transformer_core::Tensor& images) const
{
    return RequireVisionDetector().ForwardBackbone(images);
}

models::VisionDetectionOutput ModelRunner::RunVisionDetector(const transformer_core::Tensor& images) const
{
    return RequireVisionDetector().Forward(images);
}

void ModelRunner::Reset() noexcept
{
    model_type_.clear();
    encoder_classifier_.reset();
    vision_detector_.reset();
}

models::EncoderClassifier& ModelRunner::RequireEncoderClassifier() const
{
    if (!encoder_classifier_)
    {
        throw std::logic_error("ModelRunner does not currently own an encoder-classifier model.");
    }

    return *encoder_classifier_;
}

models::VisionDetector& ModelRunner::RequireVisionDetector() const
{
    if (!vision_detector_)
    {
        throw std::logic_error("ModelRunner does not currently own a vision-detector model.");
    }

    return *vision_detector_;
}

}  // namespace inference::runtime
