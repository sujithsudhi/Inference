#pragma once

/// \file
/// \brief Placeholder model adapter used by smoke tests and early runtime wiring.

#include <string>

#include "inference/runtime/model_adapter.hpp"

namespace inference::runtime
{

/// \brief Placeholder adapter used for smoke tests and early runtime wiring.
class NullModelAdapter final : public ModelAdapter
{
public:
    /// \brief Return the adapter name.
    std::string Name() const override;

    /// \brief Record the artifact root and report success without building a real model.
    core::Status Load(const core::ArtifactBundle& artifact) override;

    /// \brief Return a placeholder response and optional prompt token ids.
    Response Run(const Request&                 request,
                 const tokenization::Tokenizer& tokenizer) override;

private:
    std::string loaded_artifact_root_;
};

}  // namespace inference::runtime
