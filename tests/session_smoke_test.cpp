#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>

#include "inference/core/artifact.hpp"
#include "inference/runtime/null_model_adapter.hpp"
#include "inference/runtime/request.hpp"
#include "inference/runtime/session.hpp"
#include "inference/tokenization/whitespace_tokenizer.hpp"

int main()
{
    const auto root = std::filesystem::temp_directory_path() / "inference-session-smoke";

    std::filesystem::create_directories(root);
    std::ofstream(root / "artifact.json") << "{}";
    std::ofstream(root / "tokenizer.json") << "{}";
    std::ofstream(root / "weights.npz") << "";

    inference::runtime::Session session(std::make_unique<inference::runtime::NullModelAdapter>(),
                                        std::make_shared<inference::tokenization::WhitespaceTokenizer>());

    const auto load_status = session.Load(inference::core::ArtifactBundle(root));
    if (!load_status.ok())
    {
        std::cerr << "Expected load to succeed but got: " << load_status.message() << std::endl;
        return 1;
    }

    inference::runtime::Request request;
    request.return_token_ids = true;
    request.segments.push_back({inference::runtime::InputKind::Text, "hello generic runtime", {}});

    const auto response = session.Run(request);

    if (response.prompt_token_ids.empty())
    {
        std::cerr << "Expected prompt token ids to be populated." << std::endl;
        return 1;
    }

    if (response.status.code() != inference::core::StatusCode::NotImplemented)
    {
        std::cerr << "Expected placeholder adapter to return NotImplemented." << std::endl;
        return 1;
    }

    std::filesystem::remove_all(root);
    return 0;
}
