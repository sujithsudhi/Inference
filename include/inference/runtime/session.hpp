#pragma once

/// \file
/// \brief Adapter-oriented runtime session API.

#include "inference/core/artifact.hpp"
#include "inference/core/status.hpp"
#include "inference/runtime/model_adapter.hpp"
#include "inference/runtime/response.hpp"
#include "inference/tokenization/tokenizer.hpp"

namespace inference::runtime
{

/// \brief Adapter-oriented runtime session that owns one model adapter plus tokenizer.
class Session
{
public:
    /// \brief Construct one session from an adapter/tokenizer pair.
    Session(ModelAdapterPtr            adapter,
            tokenization::TokenizerPtr tokenizer);

    /// \brief Load one artifact bundle into the owned adapter.
    core::Status Load(const core::ArtifactBundle& artifact);

    /// \brief Run one request through the loaded adapter.
    Response Run(const Request& request);

    /// \brief Return the owned adapter.
    const ModelAdapter& adapter() const noexcept;

    /// \brief Return the owned tokenizer.
    const tokenization::Tokenizer& tokenizer() const noexcept;

private:
    ModelAdapterPtr            adapter_;
    tokenization::TokenizerPtr tokenizer_;
    bool                       loaded_ = false;
};

}  // namespace inference::runtime
