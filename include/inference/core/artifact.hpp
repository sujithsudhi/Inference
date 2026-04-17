#pragma once

/// \file
/// \brief Artifact bundle inspection types and helpers.

#include <filesystem>
#include <string>
#include <vector>

namespace inference::core
{

/// \brief One resolved file entry from an inspected artifact bundle.
struct ArtifactFile
{
    /// Stable logical role such as `metadata`, `weights`, or `tokenizer`.
    std::string           role;
    /// Absolute or root-relative path resolved during inspection.
    std::filesystem::path path;
};

/// \brief Fully inspected artifact metadata derived from a bundle root or legacy prefix.
struct ArtifactSpec
{
    /// Artifact schema version reported by the manifest or inferred by inspection.
    std::string               schema_version;
    /// Layout mode such as `manifest-dir`, `flat-dir`, `legacy-prefix`, or `unknown`.
    std::string               layout;
    /// Human-readable artifact name.
    std::string               artifact_name;
    /// High-level model-family hint such as `encoder`, `decoder`, or `vision-language`.
    std::string               model_family;
    /// High-level task hint such as `text`, `vision`, or `vlm`.
    std::string               task;
    /// Weight container format such as `npz`, `pt`, or `safetensors`.
    std::string               weight_format;
    /// Resolved path to `artifact.json` when present.
    std::filesystem::path     manifest_path;
    /// Resolved path to `model.json` or equivalent metadata file.
    std::filesystem::path     metadata_path;
    /// Resolved path to the primary weights file.
    std::filesystem::path     weights_path;
    /// Resolved path to the tokenizer artifact when present.
    std::filesystem::path     tokenizer_path;
    /// All discovered files keyed by their logical roles.
    std::vector<ArtifactFile> files;
};

/// \brief Inspector for artifact directories, flat layouts, and legacy checkpoint prefixes.
class ArtifactBundle
{
public:
    /// \brief Construct one bundle inspector from a root directory or legacy prefix path.
    explicit ArtifactBundle(std::filesystem::path root);

    /// \brief Return the original root or prefix path supplied at construction.
    const std::filesystem::path& root() const noexcept;

    /// \brief Indicate whether the bundle root or legacy prefix resolves to an existing artifact.
    bool Exists() const noexcept;

    /// \brief Inspect the artifact and return the resolved layout plus file-role metadata.
    ArtifactSpec Inspect() const;

private:
    std::filesystem::path root_;
};

}  // namespace inference::core
