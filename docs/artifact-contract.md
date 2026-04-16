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
