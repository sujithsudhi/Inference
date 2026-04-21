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

- `runtime::ModelRunner` is the artifact-backed execution path for built-in runtime families.
- `runtime::Session` remains the adapter-oriented boundary for future decoder or multimodal flows
  that do not map directly onto the current built-in artifact runners.

The decoder-runtime draft extends that direction with a dedicated typed generation surface before it
is wrapped by one adapter-backed `Session` flow:

- `docs/decoder-runtime-design.md`

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

`apps/run_checkpoint.cpp` can still accept a raw `.pt` checkpoint at the CLI boundary, but it
imports that source into a temporary artifact bundle before handing execution off to
`runtime::ModelRunner`.

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
