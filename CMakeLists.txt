cmake_minimum_required(VERSION 3.18)

project(Inference LANGUAGES CXX)

include(FetchContent)
cmake_policy(SET CMP0169 OLD)

option(BUILD_APPS  "Build command-line applications." ON)
option(BUILD_TESTS "Build test binaries." ON)
option(BUILD_TRANSFORMER_NPZ_LOADER "Build NPZ-backed artifact loaders." ON)
option(INFERENCE_ENABLE_TOKENIZERS_CPP "Build the tokenizers-cpp backend for text tokenization." ON)

FetchContent_Declare(json
                    GIT_REPOSITORY https://github.com/nlohmann/json.git
                    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

add_library(inference STATIC
    src/core/status.cpp
    src/core/artifact.cpp
    src/runtime/model_runner.cpp
    src/runtime/session.cpp
    src/runtime/null_model_adapter.cpp
    src/tokenization/tokenizer_factory.cpp
    src/tokenization/whitespace_tokenizer.cpp
    src/transformer_core/common.cpp
    src/transformer_core/text.cpp
    src/transformer_core/vision.cpp
    src/models/encoder_classifier.cpp
    src/models/vision_detector.cpp
    src/model_builder/model_builder.cpp
)

if (BUILD_TRANSFORMER_NPZ_LOADER)
    FetchContent_Declare(cnpy
                        GIT_REPOSITORY https://github.com/rogersce/cnpy.git
                        GIT_TAG master
    )
    FetchContent_Declare(zlib
                        GIT_REPOSITORY https://github.com/madler/zlib.git
                        GIT_TAG v1.3.1
    )
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "Disable zlib example targets." FORCE)
    FetchContent_GetProperties(cnpy)
    if (NOT cnpy_POPULATED)
        FetchContent_Populate(cnpy)
    endif()

    FetchContent_MakeAvailable(zlib)

    if (TARGET zlibstatic)
        set(INFERENCE_ZLIB_TARGET zlibstatic)
    elseif (TARGET zlib)
        set(INFERENCE_ZLIB_TARGET zlib)
    elseif (TARGET ZLIB::ZLIB)
        set(INFERENCE_ZLIB_TARGET ZLIB::ZLIB)
    else()
        message(FATAL_ERROR "Unable to determine a zlib target for cnpy.")
    endif()

    add_library(cnpy STATIC
        ${cnpy_SOURCE_DIR}/cnpy.cpp
    )
    target_include_directories(cnpy
        PUBLIC
            ${cnpy_SOURCE_DIR}
    )
    target_link_libraries(cnpy
        PUBLIC
            ${INFERENCE_ZLIB_TARGET}
    )
    target_compile_features(cnpy PUBLIC cxx_std_17)

    target_sources(inference
        PRIVATE
            src/artifacts/npz/state_dict_loader.cpp
    )

    target_include_directories(inference
        PUBLIC
            ${cnpy_SOURCE_DIR}
    )

    target_link_libraries(inference
        PUBLIC
            cnpy
    )

    target_compile_definitions(inference
        PUBLIC
            INFERENCE_BUILD_TRANSFORMER_NPZ_LOADER
    )
endif()

if (INFERENCE_ENABLE_TOKENIZERS_CPP)
    find_program(INFERENCE_CARGO_EXECUTABLE cargo)

    if (INFERENCE_CARGO_EXECUTABLE)
        FetchContent_Declare(tokenizers_cpp
                            GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
                            GIT_TAG v0.1.1
        )
        FetchContent_MakeAvailable(tokenizers_cpp)

        target_link_libraries(inference
            PUBLIC
                tokenizers_cpp
        )

        target_compile_definitions(inference
            PUBLIC
                INFERENCE_HAS_TOKENIZERS_CPP
        )
    else()
        message(WARNING
                "INFERENCE_ENABLE_TOKENIZERS_CPP is ON but cargo was not found. "
                "The build will continue without tokenizers-cpp support.")
    endif()
endif()

target_include_directories(inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(inference PUBLIC cxx_std_17)

target_link_libraries(inference
    PUBLIC
        nlohmann_json::nlohmann_json
)

if (MSVC)
    target_compile_options(inference PRIVATE /W4 /permissive-)
else()
    target_compile_options(inference PRIVATE -Wall -Wextra -Wpedantic)
endif()

if (BUILD_APPS)
    add_executable(inference_inspect
        apps/inference_inspect.cpp
    )

    target_link_libraries(inference_inspect
        PRIVATE
            inference
    )

    add_executable(run_checkpoint
        apps/run_checkpoint.cpp
    )

    target_link_libraries(run_checkpoint
        PRIVATE
            inference
    )

    target_compile_definitions(run_checkpoint
        PRIVATE
            INFERENCE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()

if (BUILD_TESTS)
    enable_testing()

    add_executable(transformer_core_smoke_test
        tests/transformer_core_smoke_test.cpp
    )

    target_link_libraries(transformer_core_smoke_test
        PRIVATE
            inference
    )

    add_executable(transformer_core_regression_test
        tests/transformer_core_regression_test.cpp
    )

    target_link_libraries(transformer_core_regression_test
        PRIVATE
            inference
    )

    add_executable(state_dict_loader_smoke_test
        tests/state_dict_loader_smoke_test.cpp
    )

    target_link_libraries(state_dict_loader_smoke_test
        PRIVATE
            inference
    )

    add_executable(model_builder_smoke_test
        tests/model_builder_smoke_test.cpp
    )

    target_link_libraries(model_builder_smoke_test
        PRIVATE
            inference
    )

    add_executable(encoder_e2e_test
        tests/encoder_e2e_test.cpp
    )

    target_link_libraries(encoder_e2e_test
        PRIVATE
            inference
    )

    add_test(NAME session_smoke_test
             COMMAND model_builder_smoke_test session_smoke)
    add_test(NAME artifact_bundle_smoke_test
             COMMAND model_builder_smoke_test artifact_bundle)
    add_test(NAME transformer_core_smoke_test COMMAND transformer_core_smoke_test)
    add_test(NAME transformer_core_regression_test COMMAND transformer_core_regression_test)
    add_test(NAME state_dict_loader_smoke_test COMMAND state_dict_loader_smoke_test)
    add_test(NAME model_builder_smoke_test COMMAND model_builder_smoke_test)
    add_test(NAME model_runner_smoke_test
             COMMAND model_builder_smoke_test model_runner)

    add_test(NAME encoder_e2e_test
             COMMAND encoder_e2e_test
                     ${CMAKE_CURRENT_BINARY_DIR}/testdata/imdb_fixture)

    add_test(NAME vision_detector_e2e_test
             COMMAND encoder_e2e_test
                     vision_detector_e2e
                     ${CMAKE_CURRENT_BINARY_DIR}/testdata/vision_detector_fixture)
endif()
