#pragma once

#include <string>

#include "inference/runtime/model_adapter.hpp"

namespace inference::runtime
{

class NullModelAdapter final : public ModelAdapter
{
public:
    std::string Name() const override;

    core::Status Load(const core::ArtifactBundle& artifact) override;

    Response Run(const Request&                 request,
                 const tokenization::Tokenizer& tokenizer) override;

private:
    std::string loaded_artifact_root_;
};

}  // namespace inference::runtime
