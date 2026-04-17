/// \file
/// \brief Artifact bundle inspection implementation.

#include "inference/core/artifact.hpp"

#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <nlohmann/json.hpp>

namespace inference::core
{
namespace
{
using Json = nlohmann::json;

bool path_exists(const std::filesystem::path& path)
{
    std::error_code error;
    return std::filesystem::exists(path, error);
}

bool path_is_directory(const std::filesystem::path& path)
{
    std::error_code error;
    return std::filesystem::is_directory(path, error);
}

bool path_is_regular_file(const std::filesystem::path& path)
{
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

bool is_known_suffix(const std::filesystem::path& path)
{
    const auto extension = path.extension().string();
    return extension == ".json"        ||
           extension == ".npz"         ||
           extension == ".pt"          ||
           extension == ".safetensors";
}

std::filesystem::path with_appended_suffix(const std::filesystem::path& root,
                                           const std::string&           suffix)
{
    return std::filesystem::path(root.string() + suffix);
}

std::filesystem::path normalize_legacy_prefix(const std::filesystem::path& root)
{
    if (path_is_regular_file(root) && is_known_suffix(root))
    {
        return root.parent_path() / root.stem();
    }
    return root;
}

void register_file(ArtifactSpec&                spec,
                   std::string                  role,
                   const std::filesystem::path& path)
{
    if (path.empty() || !path_exists(path))
    {
        return;
    }

    spec.files.push_back(ArtifactFile{role, path});

    if (role == "manifest" && spec.manifest_path.empty())
    {
        spec.manifest_path = path;
    }
    else if (role == "metadata" && spec.metadata_path.empty())
    {
        spec.metadata_path = path;
    }
    else if (role == "weights" && spec.weights_path.empty())
    {
        spec.weights_path = path;
    }
    else if (role == "tokenizer" && spec.tokenizer_path.empty())
    {
        spec.tokenizer_path = path;
    }
}

std::optional<Json> try_load_json(const std::filesystem::path& path)
{
    if (!path_exists(path))
    {
        return std::nullopt;
    }

    std::ifstream handle(path);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open JSON file: " + path.string());
    }

    Json payload;
    handle >> payload;
    return payload;
}

void infer_weight_format(ArtifactSpec& spec)
{
    if (!spec.weights_path.empty())
    {
        const auto extension = spec.weights_path.extension().string();

        if (extension == ".npz")
        {
            spec.weight_format = "npz";
        }
        else if (extension == ".pt")
        {
            spec.weight_format = "pt";
        }
        else if (extension == ".safetensors")
        {
            spec.weight_format = "safetensors";
        }
    }
}

void apply_metadata_hints(ArtifactSpec& spec,
                          const Json&   metadata)
{
    if (spec.weight_format == "unknown" &&
        metadata.contains("format")     &&
        metadata["format"].is_string())
    {
        spec.weight_format = metadata["format"].get<std::string>();
    }

    if (spec.task == "unknown"             &&
        metadata.contains("config")        &&
        metadata["config"].is_object()     &&
        metadata["config"].contains("name") &&
        metadata["config"]["name"].is_string())
    {
        spec.task = metadata["config"]["name"].get<std::string>();
    }

    if (!metadata.contains("state_dict_keys") || !metadata["state_dict_keys"].is_array())
    {
        return;
    }

    bool has_encoder = false;
    bool has_decoder = false;
    bool has_vision  = false;

    for (const auto& item : metadata["state_dict_keys"])
    {
        if (!item.is_string())
        {
            continue;
        }

        const auto key = item.get<std::string>();

        if (key.rfind("encoder.", 0) == 0)
        {
            has_encoder = true;
        }
        if (key.rfind("decoder.", 0) == 0 || key.rfind("blocks.", 0) == 0)
        {
            has_decoder = true;
        }
        if (key.find("vision") != std::string::npos || key.find("image") != std::string::npos)
        {
            has_vision = true;
        }
    }

    if (has_vision)
    {
        spec.model_family = "vision-language";
        spec.task         = "vlm";
    }
    else if (has_decoder)
    {
        spec.model_family = "decoder";
        if (spec.task == "unknown")
        {
            spec.task = "text";
        }
    }
    else if (has_encoder)
    {
        spec.model_family = "encoder";
        if (spec.task == "unknown")
        {
            spec.task = "text";
        }
    }
}

ArtifactSpec make_default_spec(const std::filesystem::path& root,
                               const std::string&           layout)
{
    ArtifactSpec spec;
    spec.schema_version = "inference.artifact/1";
    spec.layout         = layout;
    spec.artifact_name  = root.filename().string();
    spec.model_family   = "unknown";
    spec.task           = "unknown";
    spec.weight_format  = "unknown";
    return spec;
}

ArtifactSpec inspect_manifest_dir(const std::filesystem::path& root)
{
    ArtifactSpec spec = make_default_spec(root, "manifest-dir");

    register_file(spec, "manifest", root / "artifact.json");

    const auto manifest = try_load_json(spec.manifest_path);
    if (!manifest.has_value())
    {
        return spec;
    }

    spec.schema_version = manifest->value("schema_version", spec.schema_version);
    spec.artifact_name  = manifest->value("artifact_name", spec.artifact_name);
    spec.model_family   = manifest->value("model_family", spec.model_family);
    spec.task           = manifest->value("task", spec.task);
    spec.weight_format  = manifest->value("weight_format", spec.weight_format);

    if (manifest->contains("files") && (*manifest)["files"].is_object())
    {
        for (const auto& [role, value] : (*manifest)["files"].items())
        {
            if (!value.is_string())
            {
                continue;
            }

            register_file(spec, role, root / value.get<std::string>());
        }
    }

    if (spec.metadata_path.empty())
    {
        register_file(spec, "metadata", root / "model.json");
    }
    if (spec.weights_path.empty())
    {
        register_file(spec, "weights", root / "weights.npz");
        register_file(spec, "weights", root / "model.npz");
        register_file(spec, "weights", root / "model.safetensors");
        register_file(spec, "weights", root / "model.pt");
    }
    if (spec.tokenizer_path.empty())
    {
        register_file(spec, "tokenizer", root / "tokenizer.json");
        register_file(spec, "tokenizer", root / "tokenizer" / "tokenizer.json");
    }

    if (const auto metadata = try_load_json(spec.metadata_path); metadata.has_value())
    {
        apply_metadata_hints(spec, *metadata);
    }

    infer_weight_format(spec);
    return spec;
}

bool has_flat_dir_artifact_files(const std::filesystem::path& root)
{
    return path_exists(root / "model.json")                  ||
           path_exists(root / "config.json")                 ||
           path_exists(root / "weights.npz")                 ||
           path_exists(root / "model.npz")                   ||
           path_exists(root / "model.safetensors")           ||
           path_exists(root / "model.pt")                    ||
           path_exists(root / "tokenizer.json")              ||
           path_exists(root / "tokenizer" / "tokenizer.json") ||
           path_exists(root / "vision_encoder.npz")          ||
           path_exists(root / "projector.npz");
}

ArtifactSpec inspect_flat_dir(const std::filesystem::path& root)
{
    ArtifactSpec spec = make_default_spec(root, "flat-dir");

    register_file(spec, "metadata", root / "model.json");
    register_file(spec, "metadata", root / "config.json");
    register_file(spec, "weights",  root / "weights.npz");
    register_file(spec, "weights",  root / "model.npz");
    register_file(spec, "weights",  root / "model.safetensors");
    register_file(spec, "weights",  root / "model.pt");
    register_file(spec, "tokenizer", root / "tokenizer.json");
    register_file(spec, "tokenizer", root / "tokenizer" / "tokenizer.json");
    register_file(spec, "vision-weights", root / "vision_encoder.npz");
    register_file(spec, "projector-weights", root / "projector.npz");

    if (const auto metadata = try_load_json(spec.metadata_path); metadata.has_value())
    {
        apply_metadata_hints(spec, *metadata);
    }

    if (path_exists(root / "vision_encoder.npz") || path_exists(root / "projector.npz"))
    {
        spec.task = "vlm";
    }

    infer_weight_format(spec);
    return spec;
}

bool legacy_prefix_exists(const std::filesystem::path& root)
{
    const auto prefix = normalize_legacy_prefix(root);

    return path_exists(with_appended_suffix(prefix, ".json"))        ||
           path_exists(with_appended_suffix(prefix, ".npz"))         ||
           path_exists(with_appended_suffix(prefix, ".pt"))          ||
           path_exists(with_appended_suffix(prefix, ".safetensors"));
}

ArtifactSpec inspect_legacy_prefix(const std::filesystem::path& root)
{
    const auto prefix = normalize_legacy_prefix(root);

    ArtifactSpec spec = make_default_spec(prefix, "legacy-prefix");

    register_file(spec, "metadata", with_appended_suffix(prefix, ".json"));
    register_file(spec, "weights",  with_appended_suffix(prefix, ".npz"));
    register_file(spec, "weights",  with_appended_suffix(prefix, ".pt"));
    register_file(spec, "weights",  with_appended_suffix(prefix, ".safetensors"));

    const auto parent = prefix.parent_path();

    register_file(spec, "tokenizer", parent / "tokenizer.json");
    register_file(spec, "tokenizer", parent / "tokenizer" / "tokenizer.json");
    register_file(spec, "tokenizer", parent / "bert_tokenizer" / "tokenizer.json");
    register_file(spec, "tokenizer", parent / prefix.filename() / "tokenizer.json");

    if (const auto metadata = try_load_json(spec.metadata_path); metadata.has_value())
    {
        apply_metadata_hints(spec, *metadata);
    }

    infer_weight_format(spec);
    return spec;
}

}  // namespace

ArtifactBundle::ArtifactBundle(std::filesystem::path root)
: root_(std::move(root))
{
}

const std::filesystem::path& ArtifactBundle::root() const noexcept
{
    return root_;
}

bool ArtifactBundle::Exists() const noexcept
{
    return path_exists(root_) || legacy_prefix_exists(root_);
}

ArtifactSpec ArtifactBundle::Inspect() const
{
    if (path_is_directory(root_))
    {
        if (path_exists(root_ / "artifact.json"))
        {
            return inspect_manifest_dir(root_);
        }
        if (has_flat_dir_artifact_files(root_))
        {
            return inspect_flat_dir(root_);
        }
    }

    if (legacy_prefix_exists(root_) || (path_is_regular_file(root_) && is_known_suffix(root_)))
    {
        return inspect_legacy_prefix(root_);
    }

    return make_default_spec(root_, "unknown");
}

}  // namespace inference::core
