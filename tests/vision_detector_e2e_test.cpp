/// \file
/// \brief End-to-end test for the vision-detector runtime path.

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <cnpy.h>

#include "inference/core/artifact.hpp"
#include "inference/runtime/model_runner.hpp"

namespace
{

using inference::transformer_core::Tensor;

void Expect(bool               condition,
            const std::string& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

Tensor LoadTensor(const cnpy::NpyArray& array)
{
    std::vector<std::int64_t> shape;
    shape.reserve(array.shape.size());
    for (const auto dim : array.shape)
    {
        shape.push_back(static_cast<std::int64_t>(dim));
    }

    Tensor tensor(shape, 0.0F);
    const float* data = array.data<float>();
    for (std::size_t index = 0; index < tensor.numel(); ++index)
    {
        tensor.flat(index) = data[index];
    }
    return tensor;
}

float MaxAbsDiff(const Tensor& lhs,
                 const Tensor& rhs)
{
    Expect(lhs.shape() == rhs.shape(), "Tensor shape mismatch while computing max-abs diff.");

    float max_diff = 0.0F;
    for (std::size_t index = 0; index < lhs.numel(); ++index)
    {
        max_diff = std::max(max_diff, std::fabs(lhs.flat(index) - rhs.flat(index)));
    }
    return max_diff;
}

}  // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc != 2)
        {
            throw std::runtime_error("Usage: vision_detector_e2e_test <artifact-dir>");
        }

        const std::filesystem::path artifact_dir = std::filesystem::path(argv[1]);
        const std::filesystem::path sample_path  = artifact_dir / "sample.npz";

        if (!std::filesystem::exists(artifact_dir / "artifact.json") || !std::filesystem::exists(sample_path))
        {
            std::printf("Skipping vision-detector e2e test: fixture files are not present.\n");
            return 0;
        }

        inference::core::ArtifactBundle bundle(artifact_dir);
        inference::runtime::ModelRunner runner;
        const auto                     load_status = runner.Load(bundle);
        Expect(load_status.ok(),
               "Vision-detector fixture should load through ModelRunner.");
        Expect(runner.HasVisionDetector(),
               "Vision-detector fixture should resolve to a vision-detector model.");

        const auto sample = cnpy::npz_load(sample_path.string());
        const Tensor image                 = LoadTensor(sample.at("image"));
        const Tensor expected_sequence     = LoadTensor(sample.at("sequence_output"));
        const Tensor expected_queries      = LoadTensor(sample.at("query_features"));
        const Tensor expected_boxes        = LoadTensor(sample.at("pred_boxes"));
        const Tensor expected_objectness   = LoadTensor(sample.at("pred_objectness_logits"));
        const Tensor expected_class_logits = LoadTensor(sample.at("pred_class_logits"));
        const auto   actual_backbone       = runner.RunVisionBackbone(image);
        const auto   actual                = runner.RunVisionDetector(image);

        Expect(actual_backbone.sequence_output.shape() == expected_sequence.shape(),
               "Vision-detector backbone shape mismatch.");
        Expect(actual.query_features.shape() == expected_queries.shape(),
               "Vision-detector query-feature shape mismatch.");

        Expect(actual.pred_boxes.shape() == expected_boxes.shape(),
               "Vision-detector box shape mismatch.");
        Expect(actual.pred_objectness_logits.shape() == expected_objectness.shape(),
               "Vision-detector objectness shape mismatch.");
        Expect(actual.pred_class_logits.shape() == expected_class_logits.shape(),
               "Vision-detector class-logit shape mismatch.");

        const float backbone_diff   = MaxAbsDiff(actual_backbone.sequence_output, expected_sequence);
        const float query_diff      = MaxAbsDiff(actual.query_features, expected_queries);
        const float box_diff        = MaxAbsDiff(actual.pred_boxes, expected_boxes);
        const float objectness_diff = MaxAbsDiff(actual.pred_objectness_logits, expected_objectness);
        const float class_diff      = MaxAbsDiff(actual.pred_class_logits, expected_class_logits);

        Expect(backbone_diff < 5e-2F,
               "Vision-detector backbone output mismatch. Max abs diff: " + std::to_string(backbone_diff));
        Expect(query_diff < 5e-2F,
               "Vision-detector query-feature mismatch. Max abs diff: " + std::to_string(query_diff));
        Expect(box_diff < 5e-2F,
               "Vision-detector box output mismatch. Max abs diff: " + std::to_string(box_diff));
        Expect(objectness_diff < 5e-2F,
               "Vision-detector objectness output mismatch. Max abs diff: " + std::to_string(objectness_diff));
        Expect(class_diff < 5e-2F,
               "Vision-detector class output mismatch. Max abs diff: " + std::to_string(class_diff));

        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
