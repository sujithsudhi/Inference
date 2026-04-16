# Inference

A lightweight C++ inference library for the model families that live in the `Transformers` repo.

This repo is the standalone runtime side of the stack. The goal is to keep training, export, and experimentation in `Transformers`, while moving reusable inference code into a smaller library that both LLM and VLM applications can build on.

## Goals

- keep the runtime generic across decoder-only LLMs and future VLM adapters
- keep the dependency surface small
- separate reusable library code from demo or app-specific binaries
- make exported artifacts a stable contract between training and inference repos

## What Is In Scope First

- core runtime interfaces
- artifact bundle inspection
- artifact bundle contract
- generic model builder entry points
- tokenizer abstraction
- model adapter abstraction
- a small CLI for artifact inspection and future smoke tests

## What Stays In `Transformers` For Now

- training loops
- dataset code
- checkpoint export scripts
- model research code

## Layout

```text
Inference/
|-- apps/                  # Small binaries built on top of the library
|-- docs/                  # Architecture and migration notes
|-- include/inference/     # Public headers
|-- src/                   # Library implementation
|-- tests/                 # Smoke tests
`-- CMakeLists.txt
```

## Artifact Contract

The canonical layout is now a directory bundle rooted at an artifact folder:

```text
my-artifact/
|-- artifact.json
|-- model.json
|-- weights.npz
`-- tokenizer/tokenizer.json
```

`artifact.json` is the stable contract between `Transformers` export code and this repo's runtime. The exact schema and compatibility notes live in `docs/artifact-contract.md`.

For migration convenience, the repo also understands the current legacy export prefix layout from `Transformers`, where a checkpoint is emitted as sibling files such as:

```text
exports/
|-- imdb_checkpoint.json
|-- imdb_checkpoint.npz
`-- bert_tokenizer/tokenizer.json
```

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## First Migration Targets

- move `Transformers/inference/cpp/load_params.cpp` into a dedicated artifact loader area
- split `Transformers/inference/cpp/model.hpp` into smaller runtime and artifact types
- replace `executer.cpp` with app binaries that depend on a reusable session API
- keep the current Python exporter in `Transformers` until the artifact format settles

The first migrated artifact loader now lives under `include/inference/artifacts/npz/` and `src/artifacts/npz/`.

The first generic registry-backed builder now lives under `include/inference/model_builder/` and `src/model_builder/`, with the initial IMDB-style encoder classifier under `include/inference/models/`.

More detail lives in `docs/artifact-contract.md`, `docs/architecture.md`, `docs/migration-from-transformers.md`, and `docs/model-builder.md`.
