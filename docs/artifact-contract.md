# Artifact Contract

## Canonical Layout

The preferred artifact layout for this repo is a directory bundle:

```text
my-artifact/
|-- artifact.json
|-- model.json
|-- weights.npz
`-- tokenizer/tokenizer.json
```

This keeps the inference-side contract stable even if exporter internals change in `Transformers`.

`model.json` may contain either:

- direct model metadata plus `builder.model_type`
- a module-level graph under `builder.graph` that the runtime resolves into a supported model family

The current checkpoint-backed runtime flow is:

1. `core::ArtifactBundle` resolves the bundle layout.
2. `artifacts::npz::LoadStateDictArtifact(...)` loads metadata plus weights.
3. `model_builder::ModelBuilderRegistry` resolves the graph or model type.
4. `runtime::ModelRunner` owns the resulting executable model.

## Manifest Shape

`artifact.json` is the canonical manifest:

```json
{
  "schema_version": "inference.artifact/1",
  "artifact_name": "tinystories-small",
  "model_family": "decoder",
  "task": "text",
  "weight_format": "npz",
  "files": {
    "metadata": "model.json",
    "weights": "weights.npz",
    "tokenizer": "tokenizer/tokenizer.json"
  }
}
```

## Reserved File Roles

- `manifest`
- `metadata`
- `weights`
- `tokenizer`
- `vision-weights`
- `projector-weights`

More roles can be added later without changing the loader entry points, as long as the manifest remains explicit.

For text models, the tokenizer role may point to either:

- `tokenizer/tokenizer.json` for Hugging Face tokenizer exports
- `tokenizer/tokenizer.model` for SentencePiece exports

## Graph Metadata

The preferred way to describe PyTorch-style checkpoint structure going forward is a graph in
`model.json`:

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
        }
      ]
    }
  }
}
```

The graph is validated as a named-tensor DAG and each node can declare a `param_prefix` so the
artifact stays aligned with the original PyTorch module layout.

The currently supported graph targets are:

- `transformers.encoder_classifier`
- `vlm.vision_detector`

The current optional helper nodes recognized by the graph-backed builder are:

- `cls_token`
- `positional_encoding`

## Compatibility Mode

The repo currently supports the legacy prefix layout emitted by `Transformers/inference/pytorch_checkpoint_exporter.py`:

```text
exports/
|-- imdb_checkpoint.json
|-- imdb_checkpoint.npz
`-- bert_tokenizer/tokenizer.json
```

When an artifact bundle is created from the prefix path `exports/imdb_checkpoint`, the `ArtifactBundle` inspector resolves the metadata, weights, and tokenizer paths automatically.

## Current Recommendation

1. Keep runtime code in `Inference` consuming either manifest bundles or legacy prefixes.
2. Update the exporter in `Transformers` next so new artifacts are written as directory bundles with `artifact.json`.
3. Keep legacy compatibility during the transition so existing checkpoints continue to work.
