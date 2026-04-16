# Model Builder

`Inference` now includes a lightweight model-builder layer for checkpoint-backed runtime models.

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

## IMDB End-To-End Test Flow

The end-to-end IMDB test uses:

- the provided PyTorch checkpoint
- a Python fixture exporter that writes a NumPy-compatible `.npz` archive without requiring NumPy
- the generic NPZ-to-state-dict loader
- the registry-based model builder
- the C++ encoder classifier runtime

When `INFERENCE_IMDB_CHECKPOINT` is set, the following tests exercise the full path:

- `imdb_export_fixture_test`
- `imdb_encoder_classifier_e2e_test`

The IMDB fixture exporter also falls back to the local checkpoint at
`Transformers/results/epoch-007_train-0.2458_val-0.3269.pt` when it exists.

## Vision Detector End-To-End Test Flow

The vision-detector end-to-end test uses:

- the provided VLM detector checkpoint
- a Python fixture exporter that writes an NPZ artifact bundle plus one deterministic image sample
- the registry-based detector builder
- the C++ detector runtime

When `INFERENCE_VISION_DETECTOR_CHECKPOINT` is set, or when the local
`VLM/artifacts/checkpoints/vision_detector/best.pt` exists, the following tests
exercise the full path:

- `vision_detector_export_fixture_test`
- `vision_detector_e2e_test`

## Next Extension Points

- Add more registry entries for decoder language models.
- Extend the fixture export flow into a reusable artifact-conversion tool.
