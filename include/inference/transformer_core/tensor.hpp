#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace inference::transformer_core
{

/// Dense floating-point tensor used by the lightweight transformer kernels.
class Tensor
{
public:
    /// Construct an empty tensor.
    Tensor() = default;

    /// Construct a tensor with the provided shape and fill value.
    explicit Tensor(std::vector<std::int64_t> shape,
                    float                     fill_value = 0.0F)
    {
        Resize(std::move(shape), fill_value);
    }

    /// Resize the tensor and fill it with one scalar value.
    void Resize(std::vector<std::int64_t> shape,
                float                     fill_value = 0.0F)
    {
        shape_ = std::move(shape);

        std::size_t size = 1;
        for (const auto dim : shape_)
        {
            if (dim < 0)
            {
                throw std::invalid_argument("Tensor dimensions must be non-negative.");
            }
            size *= static_cast<std::size_t>(dim);
        }

        data_.assign(size, fill_value);
    }

    /// Return the tensor shape.
    const std::vector<std::int64_t>& shape() const noexcept
    {
        return shape_;
    }

    /// Return the number of dimensions.
    std::size_t rank() const noexcept
    {
        return shape_.size();
    }

    /// Return the length of one dimension.
    std::int64_t dim(std::size_t axis) const
    {
        if (axis >= shape_.size())
        {
            throw std::out_of_range("Tensor axis out of range.");
        }
        return shape_[axis];
    }

    /// Return the total number of elements.
    std::size_t numel() const noexcept
    {
        return data_.size();
    }

    /// Indicate whether the tensor stores no elements.
    bool empty() const noexcept
    {
        return data_.empty();
    }

    /// Return a mutable pointer to the flat storage.
    float* data() noexcept
    {
        return data_.data();
    }

    /// Return an immutable pointer to the flat storage.
    const float* data() const noexcept
    {
        return data_.data();
    }

    /// Return one element by flat index.
    float& flat(std::size_t index)
    {
        return data_.at(index);
    }

    /// Return one immutable element by flat index.
    const float& flat(std::size_t index) const
    {
        return data_.at(index);
    }

    /// Resolve a multidimensional index into one flat offset.
    std::size_t Offset(std::initializer_list<std::int64_t> indices) const
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument("Tensor index rank does not match tensor rank.");
        }

        std::size_t       offset = 0;
        std::size_t       stride = 1;
        auto shape_it            = shape_.rbegin();
        auto index_it            = indices.end();

        while (shape_it != shape_.rend())
        {
            --index_it;
            const auto index = *index_it;

            if (index < 0 || index >= *shape_it)
            {
                throw std::out_of_range("Tensor index out of bounds.");
            }

            offset += static_cast<std::size_t>(index) * stride;
            stride *= static_cast<std::size_t>(*shape_it);
            ++shape_it;
        }

        return offset;
    }

    /// Return one element by multidimensional index.
    float& at(std::initializer_list<std::int64_t> indices)
    {
        return data_.at(Offset(indices));
    }

    /// Return one immutable element by multidimensional index.
    const float& at(std::initializer_list<std::int64_t> indices) const
    {
        return data_.at(Offset(indices));
    }

private:
    std::vector<std::int64_t> shape_;
    std::vector<float>        data_;
};

/// Dense integer tensor used for token ids and similar index inputs.
class IndexTensor
{
public:
    /// Construct an empty integer tensor.
    IndexTensor() = default;

    /// Construct an integer tensor with the provided shape and fill value.
    explicit IndexTensor(std::vector<std::int64_t> shape,
                         std::int64_t              fill_value = 0)
    {
        Resize(std::move(shape), fill_value);
    }

    /// Resize the tensor and fill it with one integer value.
    void Resize(std::vector<std::int64_t> shape,
                std::int64_t              fill_value = 0)
    {
        shape_ = std::move(shape);

        std::size_t size = 1;
        for (const auto dim : shape_)
        {
            if (dim < 0)
            {
                throw std::invalid_argument("IndexTensor dimensions must be non-negative.");
            }
            size *= static_cast<std::size_t>(dim);
        }

        data_.assign(size, fill_value);
    }

    /// Return the tensor shape.
    const std::vector<std::int64_t>& shape() const noexcept
    {
        return shape_;
    }

    /// Return the number of dimensions.
    std::size_t rank() const noexcept
    {
        return shape_.size();
    }

    /// Return the length of one dimension.
    std::int64_t dim(std::size_t axis) const
    {
        if (axis >= shape_.size())
        {
            throw std::out_of_range("IndexTensor axis out of range.");
        }
        return shape_[axis];
    }

    /// Return the total number of elements.
    std::size_t numel() const noexcept
    {
        return data_.size();
    }

    /// Return a mutable pointer to the flat storage.
    std::int64_t* data() noexcept
    {
        return data_.data();
    }

    /// Return an immutable pointer to the flat storage.
    const std::int64_t* data() const noexcept
    {
        return data_.data();
    }

    /// Resolve a multidimensional index into one flat offset.
    std::size_t Offset(std::initializer_list<std::int64_t> indices) const
    {
        if (indices.size() != shape_.size())
        {
            throw std::invalid_argument("IndexTensor index rank does not match tensor rank.");
        }

        std::size_t       offset = 0;
        std::size_t       stride = 1;
        auto shape_it            = shape_.rbegin();
        auto index_it            = indices.end();

        while (shape_it != shape_.rend())
        {
            --index_it;
            const auto index = *index_it;

            if (index < 0 || index >= *shape_it)
            {
                throw std::out_of_range("IndexTensor index out of bounds.");
            }

            offset += static_cast<std::size_t>(index) * stride;
            stride *= static_cast<std::size_t>(*shape_it);
            ++shape_it;
        }

        return offset;
    }

    /// Return one mutable element by multidimensional index.
    std::int64_t& at(std::initializer_list<std::int64_t> indices)
    {
        return data_.at(Offset(indices));
    }

    /// Return one immutable element by multidimensional index.
    const std::int64_t& at(std::initializer_list<std::int64_t> indices) const
    {
        return data_.at(Offset(indices));
    }

private:
    std::vector<std::int64_t> shape_;
    std::vector<std::int64_t> data_;
};

/// Optional key/value cache used by autoregressive attention layers.
struct KeyValueCache
{
    Tensor key;
    Tensor value;

    /// Indicate whether either cached tensor is empty.
    bool empty() const noexcept
    {
        return key.empty() || value.empty();
    }
};

/// Expected tensor name and shape pair used for state-dict contracts.
struct TensorSpec
{
    std::string               name;
    std::vector<std::int64_t> shape;
};

using StateDict = std::unordered_map<std::string, Tensor>;

/// Extract only the parameter names from a list of tensor specs.
inline std::vector<std::string> SpecNames(const std::vector<TensorSpec>& specs)
{
    std::vector<std::string> names;
    names.reserve(specs.size());

    for (const auto& spec : specs)
    {
        names.push_back(spec.name);
    }

    return names;
}

}  // namespace inference::transformer_core
