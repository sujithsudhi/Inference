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

## Direct Checkpoint Run

The repo now includes a direct runner that accepts a PyTorch checkpoint plus one input file and
handles the temporary artifact conversion internally.

For text models, the intended path is now checkpoint + tokenizer + prompt:

```powershell
.\build-vs2022\Release\run_checkpoint.exe `
  --checkpoint C:\path\to\imdb_model.pt `
  --tokenizer C:\path\to\bert-base-uncased\tokenizer.json `
  --prompt "this movie was great"
```

For vision models, keep using a model-ready input file:

```powershell
.\build-vs2022\Release\run_checkpoint.exe `
  --checkpoint C:\path\to\model.pt `
  --input C:\path\to\input.json
```

Current input formats:

- encoder classifier:
  - preferred: `--tokenizer` plus `--prompt` or `--prompt-file`
  - fallback: `.json` or `.npz` with `input_ids` or `inputs`, plus optional `attention_mask`
- vision detector: `.json` or `.npz` with `image`

`run_checkpoint` uses [`mlc-ai/tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp) for
text tokenization when the build can find Rust/Cargo. If Rust is not installed, the repo still
builds, but raw prompt execution is unavailable until the tokenizer backend is enabled.

The importer currently supports the checkpoint families already implemented in this repo:

- IMDB-style encoder classifier checkpoints
- fixed-query vision detector checkpoints

## First Migration Targets

- move `Transformers/inference/cpp/load_params.cpp` into a dedicated artifact loader area
- split `Transformers/inference/cpp/model.hpp` into smaller runtime and artifact types
- replace `executer.cpp` with app binaries that depend on a reusable session API
- keep export-time graph and checkpoint conversion outside this repo's runtime library

The first migrated artifact loader now lives under `include/inference/artifacts/npz/` and `src/artifacts/npz/`.

The first generic registry-backed builder now lives under `include/inference/model_builder/` and `src/model_builder/`, with the initial IMDB-style encoder classifier under `include/inference/models/`.

More detail lives in `docs/artifact-contract.md`, `docs/architecture.md`, `docs/migration-from-transformers.md`, and `docs/model-builder.md`.
