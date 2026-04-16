#pragma once

#include <memory>
#include <string>

#include "inference/core/artifact.hpp"
#include "inference/core/status.hpp"
#include "inference/runtime/request.hpp"
#include "inference/runtime/response.hpp"
#include "inference/tokenization/tokenizer.hpp"

namespace inference::runtime
{

class ModelAdapter
{
public:
    virtual ~ModelAdapter() = default;

    virtual std::string Name() const = 0;

    virtual core::Status Load(const core::ArtifactBundle& artifact) = 0;

    virtual Response Run(const Request&                    request,
                         const tokenization::Tokenizer&    tokenizer) = 0;
};

using ModelAdapterPtr = std::unique_ptr<ModelAdapter>;

}  // namespace inference::runtime
