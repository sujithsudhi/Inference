#include "inference/runtime/session.hpp"

#include <stdexcept>
#include <utility>

namespace inference::runtime
{

Session::Session(ModelAdapterPtr            adapter,
                 tokenization::TokenizerPtr tokenizer)
: adapter_(std::move(adapter)),
  tokenizer_(std::move(tokenizer))
{
    if (!adapter_)
    {
        throw std::invalid_argument("Session requires a model adapter.");
    }

    if (!tokenizer_)
    {
        throw std::invalid_argument("Session requires a tokenizer.");
    }
}

core::Status Session::Load(const core::ArtifactBundle& artifact)
{
    if (!artifact.Exists())
    {
        loaded_ = false;
        return core::Status::NotFound("Artifact root does not exist: " + artifact.root().string());
    }

    const auto status = adapter_->Load(artifact);
    loaded_           = status.ok();
    return status;
}

Response Session::Run(const Request& request)
{
    if (!loaded_)
    {
        Response response;
        response.status = core::Status::NotFound("No artifact has been loaded into this session.");
        return response;
    }

    if (request.segments.empty())
    {
        Response response;
        response.status = core::Status::InvalidArgument("Request must contain at least one input segment.");
        return response;
    }

    return adapter_->Run(request, *tokenizer_);
}

const ModelAdapter& Session::adapter() const noexcept
{
    return *adapter_;
}

const tokenization::Tokenizer& Session::tokenizer() const noexcept
{
    return *tokenizer_;
}

}  // namespace inference::runtime
