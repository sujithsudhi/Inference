cmake_minimum_required(VERSION 3.18)

project(Inference LANGUAGES CXX)

include(FetchContent)
cmake_policy(SET CMP0169 OLD)

option(BUILD_APPS  "Build command-line applications." ON)
option(BUILD_TESTS "Build test binaries." ON)
option(BUILD_TRANSFORMER_NPZ_LOADER "Build the migrated transformer NPZ loader." ON)

FetchContent_Declare(json
                    GIT_REPOSITORY https://github.com/nlohmann/json.git
                    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

add_library(inference STATIC
    src/core/status.cpp
    src/core/artifact.cpp
    src/runtime/session.cpp
    src/runtime/null_model_adapter.cpp
    src/tokenization/whitespace_tokenizer.cpp
    src/transformer_core/common.cpp
    src/transformer_core/text.cpp
    src/transformer_core/vision.cpp
    src/models/encoder_classifier.cpp
    src/models/vision_detector.cpp
    src/model_builder/model_builder.cpp
)

if (BUILD_TRANSFORMER_NPZ_LOADER)
    FetchContent_Declare(eigen
                        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                        GIT_TAG 3.4.0
    )
    FetchContent_Declare(cnpy
                        GIT_REPOSITORY https://github.com/rogersce/cnpy.git
                        GIT_TAG master
    )
    FetchContent_Declare(zlib
                        GIT_REPOSITORY https://github.com/madler/zlib.git
                        GIT_TAG v1.3.1
    )
    set(ZLIB_BUILD_EXAMPLES OFF CACHE BOOL "Disable zlib example targets." FORCE)
    FetchContent_GetProperties(eigen)
    if (NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()

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

    add_library(inference_eigen INTERFACE)
    target_include_directories(inference_eigen
        INTERFACE
            ${eigen_SOURCE_DIR}
    )

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
            src/artifacts/npz/transformer_loader.cpp
            src/artifacts/npz/state_dict_loader.cpp
    )

    target_include_directories(inference
        PUBLIC
            ${cnpy_SOURCE_DIR}
    )

    target_link_libraries(inference
        PUBLIC
            inference_eigen
            cnpy
    )

    target_compile_definitions(inference
        PUBLIC
            INFERENCE_BUILD_TRANSFORMER_NPZ_LOADER
    )
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

    add_executable(run_inference
        apps/inference.cpp
    )

    target_link_libraries(run_inference
        PRIVATE
            inference
    )
endif()

if (BUILD_TESTS)
    enable_testing()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)

    add_executable(session_smoke_test
        tests/session_smoke_test.cpp
    )

    target_link_libraries(session_smoke_test
        PRIVATE
            inference
    )

    add_executable(artifact_bundle_smoke_test
        tests/artifact_bundle_smoke_test.cpp
    )

    target_link_libraries(artifact_bundle_smoke_test
        PRIVATE
            inference
    )

    add_executable(transformer_core_smoke_test
        tests/transformer_core_smoke_test.cpp
    )

    target_link_libraries(transformer_core_smoke_test
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

    add_executable(vision_pipeline_test
        tests/vision_pipeline_test.cpp
    )

    target_link_libraries(vision_pipeline_test
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

    add_executable(imdb_encoder_classifier_e2e_test
        tests/imdb_encoder_classifier_e2e_test.cpp
    )

    target_link_libraries(imdb_encoder_classifier_e2e_test
        PRIVATE
            inference
    )

    add_executable(vision_detector_e2e_test
        tests/vision_detector_e2e_test.cpp
    )

    target_link_libraries(vision_detector_e2e_test
        PRIVATE
            inference
    )

    add_test(NAME session_smoke_test COMMAND session_smoke_test)
    add_test(NAME artifact_bundle_smoke_test COMMAND artifact_bundle_smoke_test)
    add_test(NAME transformer_core_smoke_test COMMAND transformer_core_smoke_test)
    add_test(NAME state_dict_loader_smoke_test COMMAND state_dict_loader_smoke_test)
    add_test(NAME vision_pipeline_test COMMAND vision_pipeline_test)
    add_test(NAME model_builder_smoke_test COMMAND model_builder_smoke_test)

    add_test(NAME imdb_export_fixture_test
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/support/export_imdb_fixture.py
                     ${CMAKE_CURRENT_BINARY_DIR}/testdata/imdb_fixture)

    add_test(NAME imdb_encoder_classifier_e2e_test
             COMMAND imdb_encoder_classifier_e2e_test
                     ${CMAKE_CURRENT_BINARY_DIR}/testdata/imdb_fixture)
    set_tests_properties(imdb_encoder_classifier_e2e_test PROPERTIES DEPENDS imdb_export_fixture_test)

    add_test(NAME vision_detector_export_fixture_test
             COMMAND ${Python3_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/tests/support/export_detection_fixture.py
                     ${CMAKE_CURRENT_BINARY_DIR}/testdata/vision_detector_fixture)

    add_test(NAME vision_detector_e2e_test
             COMMAND vision_detector_e2e_test
                     ${CMAKE_CURRENT_BINARY_DIR}/testdata/vision_detector_fixture)
    set_tests_properties(vision_detector_e2e_test PROPERTIES DEPENDS vision_detector_export_fixture_test)
endif()
