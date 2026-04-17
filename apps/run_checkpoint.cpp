/// \file
/// \brief Direct checkpoint import and execution CLI entry point.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include "inference/core/artifact.hpp"
#include "inference/runtime/model_runner.hpp"
#include "inference/tokenization/tokenizer_factory.hpp"

namespace
{

using Json = nlohmann::json;
using inference::transformer_core::IndexTensor;
using inference::transformer_core::Tensor;

struct Options
{
    std::filesystem::path checkpoint_path;
    std::filesystem::path input_path;
    std::filesystem::path tokenizer_path;
    std::filesystem::path artifact_dir;
    std::filesystem::path output_path;
    std::string           python_executable = "python";
    std::string           task_hint;
    std::string           prompt;
    std::filesystem::path prompt_file_path;
    bool                  keep_artifact = false;
};

std::string Usage()
{
    return "Usage: run_checkpoint --checkpoint <model.pt> [--input <input.json|input.npz>] "
           "[--tokenizer <tokenizer.json|tokenizer.model>] [--prompt <text> | --prompt-file <file.txt>] "
           "[--artifact-dir <dir>] [--output <result.json>] [--python <python>] [--task <hint>] "
           "[--keep-artifact]";
}

std::string Quote(const std::string& value)
{
    std::string quoted = "\"";
    for (const char ch : value)
    {
        if (ch == '"')
        {
            quoted += "\\\"";
        }
        else
        {
            quoted += ch;
        }
    }
    quoted += "\"";
    return quoted;
}

std::string QuotePowerShell(const std::string& value)
{
    std::string quoted = "'";
    for (const char ch : value)
    {
        if (ch == '\'')
        {
            quoted += "''";
        }
        else
        {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

std::filesystem::path MakeTemporaryArtifactDir()
{
    std::random_device                  rd;
    std::uniform_int_distribution<int> dist(0, 15);
    constexpr const char*              hex = "0123456789abcdef";
    std::string                        suffix;
    suffix.reserve(16);

    for (int i = 0; i < 16; ++i)
    {
        suffix.push_back(hex[dist(rd)]);
    }

    return std::filesystem::temp_directory_path() / ("inference-artifact-" + suffix);
}

Options ParseArgs(int argc,
                  char** argv)
{
    Options options;

    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];

        auto require_value = [&](const std::string& flag) -> std::string
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error("Missing value for " + flag + ".\n" + Usage());
            }
            ++index;
            return argv[index];
        };

        if (arg == "--checkpoint")
        {
            options.checkpoint_path = require_value(arg);
        }
        else if (arg == "--input")
        {
            options.input_path = require_value(arg);
        }
        else if (arg == "--tokenizer")
        {
            options.tokenizer_path = require_value(arg);
        }
        else if (arg == "--artifact-dir")
        {
            options.artifact_dir = require_value(arg);
        }
        else if (arg == "--output")
        {
            options.output_path = require_value(arg);
        }
        else if (arg == "--python")
        {
            options.python_executable = require_value(arg);
        }
        else if (arg == "--task")
        {
            options.task_hint = require_value(arg);
        }
        else if (arg == "--prompt")
        {
            options.prompt = require_value(arg);
        }
        else if (arg == "--prompt-file")
        {
            options.prompt_file_path = require_value(arg);
        }
        else if (arg == "--keep-artifact")
        {
            options.keep_artifact = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            throw std::runtime_error(Usage());
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg + ".\n" + Usage());
        }
    }

    const bool has_text_prompt = !options.prompt.empty() || !options.prompt_file_path.empty();

    if (options.checkpoint_path.empty())
    {
        throw std::runtime_error("--checkpoint is required.\n" + Usage());
    }

    if (options.input_path.empty() && !has_text_prompt)
    {
        throw std::runtime_error("Provide either --input or a text prompt via --prompt/--prompt-file.\n" + Usage());
    }

    if (!options.prompt.empty() && !options.prompt_file_path.empty())
    {
        throw std::runtime_error("Use either --prompt or --prompt-file, not both.\n" + Usage());
    }

    if (has_text_prompt && options.tokenizer_path.empty())
    {
        throw std::runtime_error("A text prompt requires --tokenizer.\n" + Usage());
    }

    if (options.artifact_dir.empty())
    {
        options.artifact_dir = MakeTemporaryArtifactDir();
    }

    return options;
}

Json LoadJson(const std::filesystem::path& path)
{
    std::ifstream handle(path);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open JSON file: " + path.string());
    }

    Json payload;
    handle >> payload;
    return payload;
}

const Json& RequireMatrixField(const Json&        payload,
                               const std::string& primary_key,
                               const std::string& fallback_key = "")
{
    if (payload.contains(primary_key))
    {
        return payload.at(primary_key);
    }

    if (!fallback_key.empty() && payload.contains(fallback_key))
    {
        return payload.at(fallback_key);
    }

    if (fallback_key.empty())
    {
        throw std::runtime_error("Input JSON must contain '" + primary_key + "'.");
    }

    throw std::runtime_error("Input JSON must contain '" + primary_key + "' or '" + fallback_key + "'.");
}

void ValidateRectangularMatrix(const Json&        values,
                               const std::string& field_name)
{
    if (!values.is_array())
    {
        throw std::runtime_error(field_name + " must be a rank-2 array.");
    }

    std::size_t expected_cols = 0;
    for (std::size_t row = 0; row < values.size(); ++row)
    {
        if (!values[row].is_array())
        {
            throw std::runtime_error(field_name + " must be a rank-2 array.");
        }

        if (row == 0)
        {
            expected_cols = values[row].size();
        }
        else if (values[row].size() != expected_cols)
        {
            throw std::runtime_error(field_name + " must be rectangular.");
        }
    }
}

std::string ReadTextFile(const std::filesystem::path& path)
{
    std::ifstream handle(path, std::ios::binary);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open text file: " + path.string());
    }

    std::ostringstream buffer;
    buffer << handle.rdbuf();
    return buffer.str();
}

IndexTensor LoadIndexTensor(const Json& values)
{
    ValidateRectangularMatrix(values, "input_ids");

    const std::int64_t rows = static_cast<std::int64_t>(values.size());
    const std::int64_t cols = rows > 0 ? static_cast<std::int64_t>(values[0].size()) : 0;

    IndexTensor tensor({rows, cols}, 0);
    for (std::int64_t row = 0; row < rows; ++row)
    {
        for (std::int64_t col = 0; col < cols; ++col)
        {
            tensor.at({row, col}) = values[row][col].get<std::int64_t>();
        }
    }

    return tensor;
}

Tensor LoadFloatTensor(const Json& values)
{
    ValidateRectangularMatrix(values, "attention_mask");

    const std::int64_t rows = static_cast<std::int64_t>(values.size());
    const std::int64_t cols = rows > 0 ? static_cast<std::int64_t>(values[0].size()) : 0;

    Tensor tensor({rows, cols}, 0.0F);
    for (std::int64_t row = 0; row < rows; ++row)
    {
        for (std::int64_t col = 0; col < cols; ++col)
        {
            tensor.at({row, col}) = values[row][col].get<float>();
        }
    }

    return tensor;
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

    if (array.word_size == sizeof(float))
    {
        const float* data = array.data<float>();
        for (std::size_t index = 0; index < tensor.numel(); ++index)
        {
            tensor.flat(index) = data[index];
        }
        return tensor;
    }

    if (array.word_size == sizeof(double))
    {
        const double* data = array.data<double>();
        for (std::size_t index = 0; index < tensor.numel(); ++index)
        {
            tensor.flat(index) = static_cast<float>(data[index]);
        }
        return tensor;
    }

    throw std::runtime_error("Unsupported tensor dtype in input NPZ.");
}

std::pair<IndexTensor, std::optional<Tensor>> LoadTextInput(const std::filesystem::path& path)
{
    if (path.extension() == ".json")
    {
        const Json payload = LoadJson(path);
        const Json& input_ids = RequireMatrixField(payload, "input_ids", "inputs");

        std::optional<Tensor> attention_mask = std::nullopt;
        if (payload.contains("attention_mask"))
        {
            attention_mask = LoadFloatTensor(payload["attention_mask"]);
        }

        return {LoadIndexTensor(input_ids), attention_mask};
    }

    if (path.extension() == ".npz")
    {
        const auto npz = cnpy::npz_load(path.string());

        const auto input_ids_it = npz.find("input_ids");
        const auto inputs_it    = npz.find("inputs");
        if (input_ids_it == npz.end() && inputs_it == npz.end())
        {
            throw std::runtime_error("Text input NPZ must contain 'input_ids' or 'inputs'.");
        }

        const cnpy::NpyArray& input_array = input_ids_it != npz.end() ? input_ids_it->second : inputs_it->second;
        const Tensor          input_tensor = LoadTensor(input_array);

        if (input_tensor.rank() != 2)
        {
            throw std::runtime_error("Text input tensor must be rank-2.");
        }

        IndexTensor input_ids(input_tensor.shape(), 0);
        for (std::size_t index = 0; index < input_ids.numel(); ++index)
        {
            input_ids.data()[index] = static_cast<std::int64_t>(input_tensor.flat(index));
        }

        std::optional<Tensor> attention_mask = std::nullopt;
        const auto            mask_it = npz.find("attention_mask");
        if (mask_it != npz.end())
        {
            attention_mask = LoadTensor(mask_it->second);
        }

        return {std::move(input_ids), attention_mask};
    }

    throw std::runtime_error("Text input must be a .json or .npz file.");
}

Tensor LoadVisionInput(const std::filesystem::path& path)
{
    if (path.extension() == ".npz")
    {
        const auto npz = cnpy::npz_load(path.string());
        const auto it  = npz.find("image");
        if (it == npz.end())
        {
            throw std::runtime_error("Vision input NPZ must contain an 'image' tensor.");
        }
        return LoadTensor(it->second);
    }

    if (path.extension() == ".json")
    {
        const Json payload = LoadJson(path);
        if (!payload.contains("image"))
        {
            throw std::runtime_error("Vision input JSON must contain an 'image' field.");
        }

        const Json& values = payload["image"];
        if (!values.is_array()
            || values.empty()
            || !values[0].is_array()
            || values[0].empty()
            || !values[0][0].is_array()
            || values[0][0].empty()
            || !values[0][0][0].is_array())
        {
            throw std::runtime_error("Vision input JSON must have shape [batch, channels, height, width].");
        }

        const std::int64_t batch_size = static_cast<std::int64_t>(values.size());
        const std::int64_t channels   = static_cast<std::int64_t>(values[0].size());
        const std::int64_t height     = static_cast<std::int64_t>(values[0][0].size());
        const std::int64_t width      = static_cast<std::int64_t>(values[0][0][0].size());

        Tensor tensor({batch_size, channels, height, width}, 0.0F);
        for (std::int64_t batch = 0; batch < batch_size; ++batch)
        {
            for (std::int64_t channel = 0; channel < channels; ++channel)
            {
                for (std::int64_t row = 0; row < height; ++row)
                {
                    for (std::int64_t col = 0; col < width; ++col)
                    {
                        tensor.at({batch, channel, row, col}) =
                            values[batch][channel][row][col].get<float>();
                    }
                }
            }
        }

        return tensor;
    }

    throw std::runtime_error("Vision input must be a .json or .npz file.");
}

Json TensorToJsonRecursive(const Tensor& tensor,
                           std::size_t   dim,
                           std::size_t&  flat_index)
{
    Json out = Json::array();

    if (dim + 1 == static_cast<std::size_t>(tensor.rank()))
    {
        for (std::int64_t index = 0; index < tensor.dim(static_cast<std::int64_t>(dim)); ++index)
        {
            out.push_back(tensor.flat(flat_index));
            ++flat_index;
        }
        return out;
    }

    for (std::int64_t index = 0; index < tensor.dim(static_cast<std::int64_t>(dim)); ++index)
    {
        out.push_back(TensorToJsonRecursive(tensor, dim + 1, flat_index));
    }

    return out;
}

Json IndexTensorToJson(const IndexTensor& tensor)
{
    Json out = Json::array();

    if (tensor.rank() != 2)
    {
        throw std::runtime_error("Only rank-2 index tensors can be serialized to JSON.");
    }

    for (std::int64_t row = 0; row < tensor.dim(0); ++row)
    {
        Json row_values = Json::array();
        for (std::int64_t col = 0; col < tensor.dim(1); ++col)
        {
            row_values.push_back(tensor.at({row, col}));
        }
        out.push_back(std::move(row_values));
    }

    return out;
}

std::optional<int32_t> ResolvePadTokenId(const inference::tokenization::Tokenizer& tokenizer)
{
    static constexpr std::string_view kCandidatePadTokens[] =
        {"[PAD]", "<pad>", "<PAD>", "<|pad|>", "[pad]"};

    for (const auto token : kCandidatePadTokens)
    {
        if (const auto token_id = tokenizer.TokenToId(token); token_id.has_value())
        {
            return token_id;
        }
    }

    if (const auto vocab_size = tokenizer.VocabSize(); vocab_size.has_value() && *vocab_size > 0)
    {
        return 0;
    }

    return std::nullopt;
}

struct TokenizedPrompt
{
    IndexTensor input_ids;
    Tensor      attention_mask;
};

TokenizedPrompt EncodePrompt(const std::string&                    prompt,
                             const inference::tokenization::Tokenizer& tokenizer,
                             std::int64_t                          max_length)
{
    if (max_length <= 0)
    {
        throw std::runtime_error("Encoder-classifier max_length must be positive.");
    }

    const auto pad_token_id = ResolvePadTokenId(tokenizer);
    if (!pad_token_id.has_value())
    {
        throw std::runtime_error("Unable to determine a pad token id from the tokenizer. "
                                 "Expected a tokenizer with [PAD] or an explicit vocabulary.");
    }

    const std::vector<int32_t> encoded = tokenizer.Encode(prompt);
    const std::int64_t         length = std::min<std::int64_t>(static_cast<std::int64_t>(encoded.size()), max_length);

    IndexTensor input_ids({1, max_length}, *pad_token_id);
    Tensor      attention_mask({1, max_length}, 0.0F);

    for (std::int64_t index = 0; index < length; ++index)
    {
        input_ids.at({0, index}) = encoded[static_cast<std::size_t>(index)];
        attention_mask.at({0, index}) = 1.0F;
    }

    return {std::move(input_ids), std::move(attention_mask)};
}

Json TensorToJson(const Tensor& tensor)
{
    if (tensor.rank() == 0)
    {
        return Json(tensor.flat(0));
    }

    std::size_t flat_index = 0;
    return TensorToJsonRecursive(tensor, 0, flat_index);
}

int RunImporter(const Options& options)
{
    const std::filesystem::path importer_path =
        std::filesystem::path(INFERENCE_SOURCE_DIR) / "tools" / "import_pytorch_checkpoint.py";

    if (!std::filesystem::exists(importer_path))
    {
        throw std::runtime_error("Checkpoint importer script is missing: " + importer_path.string());
    }

    std::string inner_command = "& "
                                + QuotePowerShell(options.python_executable)
                                + " "
                                + QuotePowerShell(importer_path.string())
                                + " --checkpoint "
                                + QuotePowerShell(options.checkpoint_path.string())
                                + " --output-dir "
                                + QuotePowerShell(options.artifact_dir.string());

    if (!options.task_hint.empty())
    {
        inner_command += " --task " + QuotePowerShell(options.task_hint);
    }

    if (!options.tokenizer_path.empty())
    {
        inner_command += " --tokenizer " + QuotePowerShell(options.tokenizer_path.string());
    }

    std::ostringstream command;
    command << "powershell -NoProfile -Command " << Quote(inner_command);
    return std::system(command.str().c_str());
}

void WriteOutput(const Json&                  output,
                 const std::filesystem::path& path)
{
    std::ofstream handle(path);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    handle << output.dump(2) << "\n";
}

}  // namespace

int main(int argc, char** argv)
{
    std::optional<std::filesystem::path> cleanup_dir = std::nullopt;

    try
    {
        const Options options = ParseArgs(argc, argv);

        if (!std::filesystem::exists(options.checkpoint_path))
        {
            throw std::runtime_error("Checkpoint file not found: " + options.checkpoint_path.string());
        }

        if (!options.input_path.empty() && !std::filesystem::exists(options.input_path))
        {
            throw std::runtime_error("Input file not found: " + options.input_path.string());
        }

        if (!options.tokenizer_path.empty() && !std::filesystem::exists(options.tokenizer_path))
        {
            throw std::runtime_error("Tokenizer file not found: " + options.tokenizer_path.string());
        }

        if (!options.prompt_file_path.empty() && !std::filesystem::exists(options.prompt_file_path))
        {
            throw std::runtime_error("Prompt file not found: " + options.prompt_file_path.string());
        }

        std::filesystem::create_directories(options.artifact_dir);

        if (!options.keep_artifact)
        {
            cleanup_dir = options.artifact_dir;
        }

        const int importer_exit_code = RunImporter(options);
        if (importer_exit_code != 0)
        {
            throw std::runtime_error("Checkpoint importer failed with exit code "
                                     + std::to_string(importer_exit_code)
                                     + ".");
        }

        inference::core::ArtifactBundle bundle(options.artifact_dir);
        const inference::core::ArtifactSpec artifact_spec = bundle.Inspect();
        inference::runtime::ModelRunner     runner;
        const auto                         load_status = runner.Load(bundle);
        if (!load_status.ok())
        {
            throw std::runtime_error(load_status.message());
        }

        Json result = Json::object();
        result["model_type"]   = runner.model_type();
        result["checkpoint"]   = options.checkpoint_path.string();
        result["artifact_dir"] = options.artifact_dir.string();
        result["input_path"]   = options.input_path.string();

        if (runner.HasEncoderClassifier())
        {
            const bool has_prompt = !options.prompt.empty() || !options.prompt_file_path.empty();

            IndexTensor            inputs({1, 1}, 0);
            std::optional<Tensor>  attention_mask = std::nullopt;

            if (has_prompt)
            {
                const std::filesystem::path tokenizer_path = !options.tokenizer_path.empty()
                                                                 ? options.tokenizer_path
                                                                 : artifact_spec.tokenizer_path;
                const auto tokenizer = inference::tokenization::LoadTokenizer(tokenizer_path);
                const std::string prompt = !options.prompt.empty()
                                               ? options.prompt
                                               : ReadTextFile(options.prompt_file_path);

                const TokenizedPrompt tokenized = EncodePrompt(prompt,
                                                               *tokenizer,
                                                               runner.encoder_classifier_config().max_length);

                inputs = tokenized.input_ids;
                attention_mask = tokenized.attention_mask;
                result["prompt"] = prompt;
                result["tokenizer_path"] = tokenizer_path.string();
                result["input_ids"] = IndexTensorToJson(inputs);
                result["attention_mask"] = TensorToJson(*attention_mask);
            }
            else
            {
                const auto loaded_input = LoadTextInput(options.input_path);
                inputs = loaded_input.first;
                attention_mask = loaded_input.second;
            }

            const Tensor logits = runner.RunEncoderClassifier(inputs, attention_mask);
            result["logits"] = TensorToJson(logits);
        }
        else if (runner.HasVisionDetector())
        {
            if (options.input_path.empty())
            {
                throw std::runtime_error("Vision-detector checkpoints require --input with an image tensor.");
            }

            const Tensor image  = LoadVisionInput(options.input_path);
            const auto   output = runner.RunVisionDetector(image);

            result["pred_boxes"]             = TensorToJson(output.pred_boxes);
            result["pred_objectness_logits"] = TensorToJson(output.pred_objectness_logits);
            result["pred_class_logits"]      = TensorToJson(output.pred_class_logits);
        }
        else
        {
            throw std::runtime_error("Checkpoint importer produced an unsupported runtime model.");
        }

        if (!options.output_path.empty())
        {
            WriteOutput(result, options.output_path);
        }
        else
        {
            std::cout << result.dump(2) << "\n";
        }

        if (cleanup_dir.has_value())
        {
            std::filesystem::remove_all(*cleanup_dir);
        }

        return 0;
    }
    catch (const std::exception& ex)
    {
        if (cleanup_dir.has_value())
        {
            std::error_code error;
            std::filesystem::remove_all(*cleanup_dir, error);
        }

        std::fprintf(stderr, "%s\n", ex.what());
        return 1;
    }
}
