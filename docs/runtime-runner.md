# Runtime Runner

`Inference` now separates artifact resolution into two explicit stages:

1. `model_builder::ModelBuilderRegistry`
   This layer validates `model.json`, resolves `builder.graph`, and constructs one concrete C++
   runtime model from metadata plus weights.

2. `runtime::ModelRunner`
   This layer owns the built model and exposes execution-only entry points for the supported
   runtime families.

## Why The Split Exists

- The builder should understand artifact structure, graph metadata, and state-dict conventions.
- The runner should understand inference-time inputs and outputs.
- Apps should depend on the runner instead of wiring builder internals directly.

This keeps `apps/run_checkpoint.cpp` focused on CLI concerns such as prompt loading, JSON/NPZ input
parsing, and result formatting, while the reusable runtime boundary lives in the library.

## Relation To `Session`

- `runtime::ModelRunner` is the checkpoint-backed execution path for built-in artifact families.
- `runtime::Session` remains the adapter-oriented boundary for future decoder or multimodal flows
  that do not map directly onto the current built-in checkpoint runners.

## Current Runner Surface

`runtime::ModelRunner` supports:

- `Load(const core::ArtifactBundle&)`
- `Load(const artifacts::npz::LoadedStateDictArtifact&)`
- `RunEncoderClassifier(...)`
- `RunVisionBackbone(...)`
- `RunVisionDetector(...)`

The current stable runtime families are:

- `transformers.encoder_classifier`
- `vlm.vision_detector`

## Supported Transformer-Core Building Blocks

The checked-in low-level transformer surface now has direct regression coverage for:

- `Linear`
- `LayerNorm`
- `RotaryEmbedding`
- `MultiHeadSelfAttention`
- `FeedForward`
- `TransformerEncoderLayer`
- `TransformerDecoderLayer`
- `PatchEmbedding`
- `ViTEncoderLayer`
- `VisionTransformer`
- `TextTransformer`

The task-specific runtime models build on top of these blocks:

- `models::EncoderClassifier`
- `models::VisionDetector`
