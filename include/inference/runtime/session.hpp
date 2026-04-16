#pragma once

#include "inference/core/artifact.hpp"
#include "inference/core/status.hpp"
#include "inference/runtime/model_adapter.hpp"
#include "inference/runtime/response.hpp"
#include "inference/tokenization/tokenizer.hpp"

namespace inference::runtime
{

class Session
{
public:
    Session(ModelAdapterPtr            adapter,
            tokenization::TokenizerPtr tokenizer);

    core::Status Load(const core::ArtifactBundle& artifact);

    Response Run(const Request& request);

    const ModelAdapter& adapter() const noexcept;

    const tokenization::Tokenizer& tokenizer() const noexcept;

private:
    ModelAdapterPtr            adapter_;
    tokenization::TokenizerPtr tokenizer_;
    bool                       loaded_ = false;
};

}  // namespace inference::runtime
