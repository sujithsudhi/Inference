# Model Builder

`Inference` now includes a lightweight model-builder layer for checkpoint-backed runtime models.

The builder and the execution boundary are intentionally separate:

- `model_builder::ModelBuilderRegistry` validates metadata and builds concrete models
- `runtime::ModelRunner` owns those built models and exposes execution-only entry points

## Graph Builder

`model.json` can now describe a runtime build using `builder.graph` instead of only a fixed
`builder.model_type`. The graph builder is intentionally module-level rather than raw-op-level:

- it follows PyTorch-style module boundaries and parameter prefixes
- it validates graph connectivity through named tensor inputs and outputs
- it resolves the graph into one of the built-in runtime models backed by the existing C++ layer implementations

The initial graph schema version is `inference.graph/1`.

Each node supports:

- `name`
- `op`
- `inputs`
- `outputs`
- `param_prefix`
- `attrs`

The current supported graph patterns are:

- `token_embedding` + `transformer_encoder` + `layer_norm` + `classifier_head`
  - resolves to `transformers.encoder_classifier`
  - optional helper nodes: `cls_token`, `positional_encoding`
- `patch_embedding` + `vision_backbone` + `detection_head`
  - resolves to `vlm.vision_detector`
  - optional helper node: `cls_token`

## Current Built-In Model Types

- `transformers.encoder_classifier`
  - Intended for the IMDB-style encoder classifier used in `Transformers/models/classifier.py`
  - Expects PyTorch-style parameter names such as:
    - `token_embedding.weight`
    - `cls_token`
    - `position.positional_table`
    - `encoder.<idx>.residual_attention.module.w_q.weight`
    - `norm.weight`
    - `head.0.weight`
    - `head.3.weight`

- `vlm.vision_detector`
  - Intended for the fixed-query detector used in `VLM/App/object_detection.py`
  - Expects PyTorch-style parameter names such as:
    - `backbone.patch_embed.proj.weight`
    - `backbone.pos_embed`
    - `backbone.cls_token`
    - `backbone.blocks.<idx>.residual_attention.module.w_q.weight`
    - `backbone.norm.weight`
    - `detection_head.query_embed`
    - `detection_head.cross_attention.in_proj_weight`
    - `detection_head.ffn.0.weight`
    - `detection_head.box_head.1.weight`
    - `detection_head.objectness_head.1.weight`
    - `detection_head.class_head.1.weight`

## Artifact Path

The builder consumes the existing artifact bundle contract:

- `artifact.json`
- `model.json`
- `weights.npz`

For generic PyTorch-style exports, `model.json` should include:

```json
{
  "builder": {
    "model_type": "transformers.encoder_classifier"
  },
  "config": {
    "name": "imdb",
    "model": {
      "vocab_size": 30522,
      "max_length": 513,
      "embed_dim": 128,
      "depth": 4,
      "num_heads": 4
    }
  }
}
```

If `builder.model_type` is omitted, the registry currently infers the encoder-classifier builder from the state-dict key layout.

## Graph Artifact Example

`model.json` can also carry a graph-backed build description:

```json
{
  "builder": {
    "model_type": "graph",
    "graph": {
      "version": "inference.graph/1",
      "inputs": ["input_ids", "attention_mask"],
      "outputs": ["logits"],
      "nodes": [
        {
          "name": "token_embedding",
          "op": "token_embedding",
          "inputs": ["input_ids"],
          "outputs": ["embedded_tokens"],
          "param_prefix": "token_embedding.",
          "attrs": {
            "vocab_size": 30522,
            "embed_dim": 128
          }
        },
        {
          "name": "encoder",
          "op": "transformer_encoder",
          "inputs": ["embedded_tokens", "attention_mask"],
          "outputs": ["encoded_tokens"],
          "param_prefix": "encoder.",
          "attrs": {
            "max_length": 513,
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 2.0,
            "pooling": "cls"
          }
        },
        {
          "name": "norm",
          "op": "layer_norm",
          "inputs": ["encoded_tokens"],
          "outputs": ["normalized_tokens"],
          "param_prefix": "norm."
        },
        {
          "name": "classifier",
          "op": "classifier_head",
          "inputs": ["normalized_tokens"],
          "outputs": ["logits"],
          "param_prefix": "head.",
          "attrs": {
            "num_outputs": 1
          }
        }
      ]
    }
  }
}
```

When `builder.graph` is present, the registry validates the graph and derives the target runtime model type from the node pattern.

## Direct Checkpoint Import

For execution flows, the repo now provides a direct checkpoint runner.

For text models, the preferred user-facing flow is checkpoint + tokenizer + prompt:

```powershell
.\build-vs2022\Release\run_checkpoint.exe `
  --checkpoint C:\path\to\model.pt `
  --tokenizer C:\path\to\tokenizer.json `
  --prompt "this movie was great"
```

For model-ready tensor inputs or vision models, `--input` remains available:

```powershell
.\build-vs2022\Release\run_checkpoint.exe `
  --checkpoint C:\path\to\model.pt `
  --input C:\path\to\input.json
```

`run_checkpoint` uses the bundled importer at `tools/import_pytorch_checkpoint.py` to:

- load a supported PyTorch checkpoint
- materialize a temporary artifact bundle with `artifact.json`, `model.json`, and `weights.npz`
- copy the tokenizer into the temporary artifact bundle when one is provided
- load the artifact through `runtime::ModelRunner`, which internally delegates model construction to the graph builder
- tokenize the prompt for text models or load the provided tensor input
- execute inference on the resolved runtime input

Text prompt tokenization uses [`mlc-ai/tokenizers-cpp`](https://github.com/mlc-ai/tokenizers-cpp).
Its official build flow expects the project to be added via CMake and requires Rust to be
installed. The repo auto-enables that backend when Cargo is available and otherwise keeps the rest
of the runtime build working without raw prompt support.

## IMDB End-To-End Test Flow

The end-to-end encoder test uses:

- an artifact directory rooted at `artifact.json`
- the generic NPZ-to-state-dict loader
- `runtime::ModelRunner`, which internally delegates model construction to the builder
- the C++ encoder classifier runtime

The following test binary exercises the full path when an encoder fixture directory is present:

- `encoder_e2e_test`

## Vision Detector End-To-End Test Flow

The vision-detector end-to-end test uses:

- an artifact directory rooted at `artifact.json`
- `runtime::ModelRunner`, which internally delegates model construction to the builder
- the C++ detector runtime

The following test binary exercises the full path when a detector fixture directory is present:

- `vision_detector_e2e_test`

## Next Extension Points

- Add more registry entries for decoder language models.
- Add an artifact-conversion tool that emits graph-backed manifests directly.
- Add more runner entry points as new artifact-backed runtime families become executable.
