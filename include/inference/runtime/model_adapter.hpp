#pragma once

/// \file
/// \brief Adapter interface for generic runtime session flows.

#include <memory>
#include <string>

#include "inference/core/artifact.hpp"
#include "inference/core/status.hpp"
#include "inference/runtime/request.hpp"
#include "inference/runtime/response.hpp"
#include "inference/tokenization/tokenizer.hpp"

namespace inference::runtime
{

/// \brief Adapter interface used by `Session` for generic text or multimodal runtime flows.
class ModelAdapter
{
public:
    /// \brief Virtual destructor for adapter subclasses.
    virtual ~ModelAdapter() = default;

    /// \brief Return one stable adapter name for logging or diagnostics.
    virtual std::string Name() const = 0;

    /// \brief Load one artifact bundle into the adapter.
    virtual core::Status Load(const core::ArtifactBundle& artifact) = 0;

    /// \brief Execute one runtime request using the supplied tokenizer.
    virtual Response Run(const Request&                    request,
                         const tokenization::Tokenizer&    tokenizer) = 0;
};

/// \brief Owning pointer to one runtime adapter instance.
using ModelAdapterPtr = std::unique_ptr<ModelAdapter>;

}  // namespace inference::runtime
