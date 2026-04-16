#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference/transformer_core/tensor.hpp"

namespace inference::transformer_core::detail
{

inline std::string JoinKey(const std::string& prefix,
                           const std::string& suffix)
{
    return prefix.empty() ? suffix : prefix + suffix;
}

inline bool ShapesEqual(const std::vector<std::int64_t>& lhs,
                        const std::vector<std::int64_t>& rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }

    for (std::size_t i = 0; i < lhs.size(); ++i)
    {
        if (lhs[i] != rhs[i])
        {
            return false;
        }
    }

    return true;
}

inline std::string ShapeToString(const std::vector<std::int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";

    for (std::size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << shape[i];
    }

    oss << "]";
    return oss.str();
}

inline void ValidateShape(const Tensor&                    tensor,
                          const std::vector<std::int64_t>& expected_shape,
                          const std::string&               key)
{
    if (!ShapesEqual(tensor.shape(), expected_shape))
    {
        throw std::invalid_argument("Tensor shape mismatch for '" + key
                                    + "'. Expected "
                                    + ShapeToString(expected_shape)
                                    + " but received "
                                    + ShapeToString(tensor.shape())
                                    + ".");
    }
}

inline bool HasTensor(const StateDict&   state_dict,
                      const std::string& key)
{
    return state_dict.find(key) != state_dict.end();
}

inline const Tensor& RequireTensor(const StateDict&                 state_dict,
                                   const std::string&               key,
                                   const std::vector<std::int64_t>& expected_shape = {})
{
    const auto it = state_dict.find(key);
    if (it == state_dict.end())
    {
        throw std::invalid_argument("Missing tensor '" + key + "' in state_dict.");
    }

    if (!expected_shape.empty())
    {
        ValidateShape(it->second, expected_shape, key);
    }

    return it->second;
}

inline std::vector<std::int64_t> ReplaceLastDim(std::vector<std::int64_t> shape,
                                                std::int64_t              last_dim)
{
    if (shape.empty())
    {
        throw std::invalid_argument("Tensor rank must be at least 1.");
    }

    shape.back() = last_dim;
    return shape;
}

inline void AppendSpecs(std::vector<TensorSpec>&       dst,
                        const std::vector<TensorSpec>& src)
{
    dst.insert(dst.end(), src.begin(), src.end());
}

inline Tensor ConcatSequence(const Tensor& lhs, const Tensor& rhs)
{
    const auto batch = lhs.dim(0);
    const auto seq_l = lhs.dim(1);
    const auto seq_r = rhs.dim(1);
    const auto dim = lhs.dim(2);

    Tensor result({batch, seq_l + seq_r, dim}, 0.0F);

    for (std::int64_t b = 0; b < batch; ++b)
    {
        for (std::int64_t s = 0; s < seq_l; ++s)
        {
            for (std::int64_t d = 0; d < dim; ++d)
            {
                result.at({b, s, d}) = lhs.at({b, s, d});
            }
        }
        for (std::int64_t s = 0; s < seq_r; ++s)
        {
            for (std::int64_t d = 0; d < dim; ++d)
            {
                result.at({b, seq_l + s, d}) = rhs.at({b, s, d});
            }
        }
    }

    return result;
}

}  // namespace inference::transformer_core::detail
