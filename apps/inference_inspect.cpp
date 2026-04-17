/// \file
/// \brief Artifact inspection CLI entry point.

#include <iostream>
#include <string>

#include "inference/core/artifact.hpp"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: inference_inspect <artifact-root>" << std::endl;
        return 1;
    }

    const inference::core::ArtifactBundle artifact(argv[1]);

    if (!artifact.Exists())
    {
        std::cerr << "Artifact root not found: " << artifact.root() << std::endl;
        return 1;
    }

    const auto spec = artifact.Inspect();

    std::cout << "schema_version: " << spec.schema_version << std::endl;
    std::cout << "layout        : " << spec.layout << std::endl;
    std::cout << "artifact_name : " << spec.artifact_name << std::endl;
    std::cout << "model_family  : " << spec.model_family << std::endl;
    std::cout << "task          : " << spec.task << std::endl;
    std::cout << "weight_format : " << spec.weight_format << std::endl;
    std::cout << "manifest_path : " << spec.manifest_path.string() << std::endl;
    std::cout << "metadata_path : " << spec.metadata_path.string() << std::endl;
    std::cout << "weights_path  : " << spec.weights_path.string() << std::endl;
    std::cout << "tokenizer_path: " << spec.tokenizer_path.string() << std::endl;
    std::cout << "files         : " << spec.files.size() << std::endl;

    for (const auto& file : spec.files)
    {
        std::cout << "  - " << file.role << " => " << file.path.string() << std::endl;
    }

    return 0;
}
