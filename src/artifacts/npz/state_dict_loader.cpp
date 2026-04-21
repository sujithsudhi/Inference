/// \file
/// \brief Generic NPZ state-dict artifact loader implementation.

#include "inference/artifacts/npz/state_dict_loader.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace inference::artifacts::npz
{

namespace
{

std::vector<std::int64_t> ToShape(const cnpy::NpyArray& array)
{
    std::vector<std::int64_t> shape;
    shape.reserve(array.shape.size());

    for (const auto dim : array.shape)
    {
        shape.push_back(static_cast<std::int64_t>(dim));
    }

    return shape;
}

template <typename T>
void CopyTensorData(transformer_core::Tensor& out,
                    const cnpy::NpyArray&     array)
{
    const T* src = array.data<T>();
    for (std::size_t i = 0; i < out.numel(); ++i)
    {
        out.flat(i) = static_cast<float>(src[i]);
    }
}

void RequireArtifactPaths(const core::ArtifactSpec& artifact)
{
    if (artifact.metadata_path.empty())
    {
        throw std::runtime_error("Artifact is missing a metadata path.");
    }
    if (artifact.weights_path.empty())
    {
        throw std::runtime_error("Artifact is missing a weights path.");
    }
}

Json LoadJsonDocument(const std::string& path)
{
    std::ifstream handle(path);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open JSON file: " + path);
    }

    Json payload;
    handle >> payload;
    return payload;
}

cnpy::npz_t LoadNpzArchive(const std::string& path)
{
    return cnpy::npz_load(path);
}

std::uint64_t ReadLittleEndianUint64(std::ifstream& handle)
{
    std::array<unsigned char, 8> bytes = {};
    handle.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));

    if (!handle)
    {
        throw std::runtime_error("Failed to read safetensors header size.");
    }

    std::uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }

    return value;
}

Json LoadSafeTensorHeader(std::ifstream& handle)
{
    const std::uint64_t header_size = ReadLittleEndianUint64(handle);
    if (header_size > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
    {
        throw std::runtime_error("Safetensors header is too large to load.");
    }

    std::string header_payload(static_cast<std::size_t>(header_size), '\0');
    handle.read(header_payload.data(),
                static_cast<std::streamsize>(header_payload.size()));

    if (!handle)
    {
        throw std::runtime_error("Failed to read safetensors header payload.");
    }

    return Json::parse(header_payload);
}

std::vector<std::int64_t> LoadSafeTensorShape(const Json& tensor_spec)
{
    if (!tensor_spec.contains("shape") || !tensor_spec["shape"].is_array())
    {
        throw std::runtime_error("Safetensors entry is missing a valid shape array.");
    }

    std::vector<std::int64_t> shape;
    shape.reserve(tensor_spec["shape"].size());

    for (const auto& dim_value : tensor_spec["shape"])
    {
        if (!dim_value.is_number_integer())
        {
            throw std::runtime_error("Safetensors shape dimensions must be integers.");
        }

        const auto dim = dim_value.get<std::int64_t>();
        if (dim < 0)
        {
            throw std::runtime_error("Safetensors shape dimensions must be non-negative.");
        }

        shape.push_back(dim);
    }

    return shape;
}

std::pair<std::uint64_t, std::uint64_t> LoadSafeTensorOffsets(const Json& tensor_spec)
{
    if (!tensor_spec.contains("data_offsets") || !tensor_spec["data_offsets"].is_array())
    {
        throw std::runtime_error("Safetensors entry is missing valid data offsets.");
    }

    const Json& offsets = tensor_spec["data_offsets"];
    if (offsets.size() != 2)
    {
        throw std::runtime_error("Safetensors data_offsets must contain exactly two values.");
    }

    if (!offsets[0].is_number_integer() || !offsets[1].is_number_integer())
    {
        throw std::runtime_error("Safetensors data offsets must be integers.");
    }

    const auto begin = offsets[0].get<std::int64_t>();
    const auto end   = offsets[1].get<std::int64_t>();

    if (begin < 0 || end < 0 || end < begin)
    {
        throw std::runtime_error("Safetensors data offsets must be non-negative and ordered.");
    }

    return {static_cast<std::uint64_t>(begin), static_cast<std::uint64_t>(end)};
}

std::uint64_t ByteWidthForSafeTensorDType(const std::string& dtype)
{
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8")
    {
        return 1;
    }

    if (dtype == "F16" || dtype == "BF16" || dtype == "U16" || dtype == "I16")
    {
        return 2;
    }

    if (dtype == "F32" || dtype == "I32" || dtype == "U32")
    {
        return 4;
    }

    if (dtype == "F64" || dtype == "I64" || dtype == "U64")
    {
        return 8;
    }

    throw std::runtime_error("Unsupported safetensors dtype '" + dtype + "'.");
}

float Float16ToFloat(std::uint16_t bits)
{
    const std::uint32_t sign     = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = static_cast<std::uint32_t>((bits >> 10U) & 0x1FU);
    const std::uint32_t mantissa = static_cast<std::uint32_t>(bits & 0x03FFU);

    std::uint32_t fp32_bits = 0;
    if (exponent == 0)
    {
        if (mantissa == 0)
        {
            fp32_bits = sign;
        }
        else
        {
            std::uint32_t normalized_mantissa = mantissa;
            std::uint32_t adjusted_exponent   = 127U - 15U + 1U;

            while ((normalized_mantissa & 0x0400U) == 0)
            {
                normalized_mantissa <<= 1U;
                --adjusted_exponent;
            }

            normalized_mantissa &= 0x03FFU;
            fp32_bits = sign
                        | (adjusted_exponent << 23U)
                        | (normalized_mantissa << 13U);
        }
    }
    else if (exponent == 0x1FU)
    {
        fp32_bits = sign | 0x7F800000U | (mantissa << 13U);
    }
    else
    {
        const std::uint32_t adjusted_exponent = exponent + (127U - 15U);
        fp32_bits = sign | (adjusted_exponent << 23U) | (mantissa << 13U);
    }

    float value = 0.0F;
    std::memcpy(&value, &fp32_bits, sizeof(value));
    return value;
}

float BFloat16ToFloat(std::uint16_t bits)
{
    const std::uint32_t fp32_bits = static_cast<std::uint32_t>(bits) << 16U;
    float               value     = 0.0F;
    std::memcpy(&value, &fp32_bits, sizeof(value));
    return value;
}

template <typename T>
std::vector<T> ReadTypedSafeTensorBlock(std::ifstream& handle,
                                        const std::uint64_t offset,
                                        const std::size_t   count)
{
    std::vector<T> values(count);

    if (count == 0)
    {
        return values;
    }

    handle.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!handle)
    {
        throw std::runtime_error("Failed to seek within safetensors weights payload.");
    }

    const std::uint64_t byte_count = static_cast<std::uint64_t>(count) * sizeof(T);
    if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()))
    {
        throw std::runtime_error("Safetensors tensor payload is too large to read.");
    }

    handle.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(byte_count));

    if (!handle)
    {
        throw std::runtime_error("Failed to read safetensors tensor payload.");
    }

    return values;
}

template <typename T>
void CopyTypedValues(transformer_core::Tensor& out,
                     const std::vector<T>&     values)
{
    if (values.size() != out.numel())
    {
        throw std::runtime_error("Safetensors tensor element count mismatch.");
    }

    for (std::size_t index = 0; index < values.size(); ++index)
    {
        out.flat(index) = static_cast<float>(values[index]);
    }
}

void CopyFloat16Values(transformer_core::Tensor&       out,
                       const std::vector<std::uint16_t>& values)
{
    if (values.size() != out.numel())
    {
        throw std::runtime_error("Safetensors tensor element count mismatch.");
    }

    for (std::size_t index = 0; index < values.size(); ++index)
    {
        out.flat(index) = Float16ToFloat(values[index]);
    }
}

void CopyBFloat16Values(transformer_core::Tensor&       out,
                        const std::vector<std::uint16_t>& values)
{
    if (values.size() != out.numel())
    {
        throw std::runtime_error("Safetensors tensor element count mismatch.");
    }

    for (std::size_t index = 0; index < values.size(); ++index)
    {
        out.flat(index) = BFloat16ToFloat(values[index]);
    }
}

transformer_core::Tensor LoadSafeTensor(std::ifstream& handle,
                                        const Json&     tensor_spec,
                                        const std::uint64_t data_offset)
{
    if (!tensor_spec.contains("dtype") || !tensor_spec["dtype"].is_string())
    {
        throw std::runtime_error("Safetensors entry is missing a dtype.");
    }

    const std::string             dtype = tensor_spec["dtype"].get<std::string>();
    const std::vector<std::int64_t> shape = LoadSafeTensorShape(tensor_spec);
    const auto [begin, end] = LoadSafeTensorOffsets(tensor_spec);
    const std::uint64_t           byte_width = ByteWidthForSafeTensorDType(dtype);
    transformer_core::Tensor      tensor(shape, 0.0F);

    const std::uint64_t element_count = static_cast<std::uint64_t>(tensor.numel());
    if (element_count > 0 &&
        byte_width > std::numeric_limits<std::uint64_t>::max() / element_count)
    {
        throw std::runtime_error("Safetensors tensor size overflow.");
    }

    const std::uint64_t expected_bytes = element_count * byte_width;
    const std::uint64_t actual_bytes   = end - begin;
    if (expected_bytes != actual_bytes)
    {
        throw std::runtime_error("Safetensors tensor byte range does not match its shape.");
    }

    const std::uint64_t absolute_offset = data_offset + begin;

    if (dtype == "F32")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<float>(handle,
                                                                absolute_offset,
                                                                static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "F64")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<double>(handle,
                                                                 absolute_offset,
                                                                 static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "F16")
    {
        CopyFloat16Values(tensor, ReadTypedSafeTensorBlock<std::uint16_t>(handle,
                                                                          absolute_offset,
                                                                          static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "BF16")
    {
        CopyBFloat16Values(tensor, ReadTypedSafeTensorBlock<std::uint16_t>(handle,
                                                                           absolute_offset,
                                                                           static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "I64")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<std::int64_t>(handle,
                                                                       absolute_offset,
                                                                       static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "I32")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<std::int32_t>(handle,
                                                                       absolute_offset,
                                                                       static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "I16")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<std::int16_t>(handle,
                                                                       absolute_offset,
                                                                       static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "I8")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<std::int8_t>(handle,
                                                                      absolute_offset,
                                                                      static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "U64")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<std::uint64_t>(handle,
                                                                        absolute_offset,
                                                                        static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "U32")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<std::uint32_t>(handle,
                                                                        absolute_offset,
                                                                        static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "U16")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<std::uint16_t>(handle,
                                                                        absolute_offset,
                                                                        static_cast<std::size_t>(element_count)));
    }
    else if (dtype == "U8" || dtype == "BOOL")
    {
        CopyTypedValues(tensor, ReadTypedSafeTensorBlock<std::uint8_t>(handle,
                                                                       absolute_offset,
                                                                       static_cast<std::size_t>(element_count)));
    }
    else
    {
        throw std::runtime_error("Unsupported safetensors dtype '" + dtype + "'.");
    }

    return tensor;
}

transformer_core::StateDict LoadSafeTensorArchive(const std::string& path)
{
    std::ifstream handle(path, std::ios::binary);
    if (!handle.is_open())
    {
        throw std::runtime_error("Failed to open safetensors weights file: " + path);
    }

    const Json          header = LoadSafeTensorHeader(handle);
    const std::uint64_t data_offset = static_cast<std::uint64_t>(handle.tellg());
    transformer_core::StateDict state_dict;

    for (const auto& [key, value] : header.items())
    {
        if (key == "__metadata__")
        {
            continue;
        }

        if (!value.is_object())
        {
            throw std::runtime_error("Safetensors tensor entry must be a JSON object.");
        }

        state_dict.emplace(key, LoadSafeTensor(handle, value, data_offset));
    }

    return state_dict;
}

std::unordered_map<std::string, int> LoadVocabMap(const std::string& path)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open vocab file: " + path);
    }

    Json vocab_json;
    file >> vocab_json;

    std::unordered_map<std::string, int> vocab;

    if (vocab_json.contains("model") && vocab_json["model"].contains("vocab"))
    {
        const auto& vocab_obj = vocab_json["model"]["vocab"];
        for (const auto& [token, id] : vocab_obj.items())
        {
            vocab[token] = id.get<int>();
        }
    }
    else if (vocab_json.is_object())
    {
        for (const auto& [token, id] : vocab_json.items())
        {
            if (id.is_number_integer())
            {
                vocab[token] = id.get<int>();
            }
        }
    }
    else
    {
        throw std::runtime_error("Unsupported vocab format in: " + path);
    }

    return vocab;
}

}  // namespace

transformer_core::Tensor ToTensor(const cnpy::NpyArray& array)
{
    transformer_core::Tensor out(ToShape(array), 0.0F);

    if (array.word_size == sizeof(float))
    {
        CopyTensorData<float>(out, array);
        return out;
    }

    if (array.word_size == sizeof(double))
    {
        CopyTensorData<double>(out, array);
        return out;
    }

    throw std::runtime_error("Unsupported NPY dtype in generic state dict loader.");
}

transformer_core::StateDict LoadStateDict(const cnpy::npz_t& weights)
{
    transformer_core::StateDict state_dict;

    for (const auto& kv : weights)
    {
        state_dict.emplace(kv.first, ToTensor(kv.second));
    }

    return state_dict;
}

LoadedStateDictArtifact LoadStateDictArtifact(const core::ArtifactBundle& artifact)
{
    return LoadStateDictArtifact(artifact.Inspect());
}

LoadedStateDictArtifact LoadStateDictArtifact(const core::ArtifactSpec& artifact)
{
    RequireArtifactPaths(artifact);

    LoadedStateDictArtifact loaded;
    loaded.artifact   = artifact;
    loaded.metadata   = LoadJsonDocument(artifact.metadata_path.string());

    if (artifact.weight_format == "safetensors" || artifact.weights_path.extension() == ".safetensors")
    {
        loaded.state_dict = LoadSafeTensorArchive(artifact.weights_path.string());
    }
    else if (artifact.weight_format == "npz" || artifact.weights_path.extension() == ".npz")
    {
        loaded.state_dict = LoadStateDict(LoadNpzArchive(artifact.weights_path.string()));
    }
    else
    {
        throw std::runtime_error("Unsupported artifact weight format '" + artifact.weight_format
                                 + "'. Expected npz or safetensors.");
    }

    if (!artifact.tokenizer_path.empty())
    {
        loaded.vocab = LoadVocabMap(artifact.tokenizer_path.string());
    }

    return loaded;
}

}  // namespace inference::artifacts::npz
