#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace inference::core
{

struct ArtifactFile
{
    std::string           role;
    std::filesystem::path path;
};

struct ArtifactSpec
{
    std::string               schema_version;
    std::string               layout;
    std::string               artifact_name;
    std::string               model_family;
    std::string               task;
    std::string               weight_format;
    std::filesystem::path     manifest_path;
    std::filesystem::path     metadata_path;
    std::filesystem::path     weights_path;
    std::filesystem::path     tokenizer_path;
    std::vector<ArtifactFile> files;
};

class ArtifactBundle
{
public:
    explicit ArtifactBundle(std::filesystem::path root);

    const std::filesystem::path& root() const noexcept;

    bool Exists() const noexcept;

    ArtifactSpec Inspect() const;

private:
    std::filesystem::path root_;
};

}  // namespace inference::core
