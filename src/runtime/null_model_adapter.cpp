/// \file
/// \brief Placeholder model-adapter implementation used by smoke tests.

#include "inference/runtime/null_model_adapter.hpp"

#include <sstream>

namespace inference::runtime
{
namespace
{

std::string collect_text(const Request& request)
{
    std::ostringstream stream;

    for (const auto& segment : request.segments)
    {
        if (segment.kind != InputKind::Text)
        {
            continue;
        }

        if (!stream.str().empty())
        {
            stream << ' ';
        }

        stream << segment.text;
    }

    return stream.str();
}

}  // namespace

std::string NullModelAdapter::Name() const
{
    return "null";
}

core::Status NullModelAdapter::Load(const core::ArtifactBundle& artifact)
{
    loaded_artifact_root_ = artifact.root().string();
    return core::Status::Ok();
}

Response NullModelAdapter::Run(const Request&                 request,
                               const tokenization::Tokenizer& tokenizer)
{
    Response response;
    response.text = "Adapter 'null' loaded artifact root: " + loaded_artifact_root_;

    if (request.return_token_ids)
    {
        response.prompt_token_ids = tokenizer.Encode(collect_text(request));
    }

    response.status = core::Status::NotImplemented(
        "NullModelAdapter is a placeholder. Add a concrete decoder or VLM adapter next.");

    return response;
}

}  // namespace inference::runtime
