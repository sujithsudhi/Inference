# Architecture

## Principles

- Keep the public API smaller than the application layer.
- Treat model export artifacts as an external contract.
- Support both text-only and multimodal workloads through adapters instead of branching the whole runtime.
- Keep demo logic out of the core library.

## Layers

### `artifacts`

Owns artifact manifests, layout compatibility, and model-family-specific loaders such as the migrated NPZ transformer loader.

### `core`

Owns artifact discovery and shared status/result types.

### `tokenization`

Owns tokenizer interfaces. The initial whitespace tokenizer is only a development fallback so the repo can build and test without external dependencies.

### `runtime`

Owns request and response types, adapter interfaces, and the session object that applications use.

### `apps`

Small binaries for inspection, smoke testing, and later interactive or server-style inference entry points.

## Intended Growth Path

1. Keep the artifact contract stable while the exporter side is still evolving in `Transformers`.
2. Add a decoder adapter for TinyStories-style generation.
3. Add a shared tokenization package boundary for HuggingFace tokenizer support.
4. Add a VLM adapter that can compose a vision encoder, projector, and text decoder without changing the core session API.

## Deliberate Non-Goals For The First Cut

- reproducing the entire current C++ inference tree immediately
- bundling training or export code into this repo
- locking the final on-disk artifact schema before the runtime seams are stable
