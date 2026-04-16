# Migration From `Transformers`

This repo starts as a clean extraction target, not a direct file-for-file copy.

## Mapping

- `Transformers/inference/cpp/load_params.cpp`
  Migrated into `src/artifacts/npz/transformer_loader.cpp` as the first model-family-specific loader.

- `Transformers/inference/cpp/model.hpp`
  Split into `include/inference/artifacts/npz/transformer_weights.hpp` and `include/inference/artifacts/npz/transformer_loader.hpp`. The tokenizer ownership piece is intentionally deferred until the tokenization boundary settles.

- `Transformers/inference/cpp/src/model/executer.*`
  Replace with library-backed apps. Interactive console behavior belongs in `apps/`, not in the reusable runtime.

- `Transformers/inference/cpp/tinystories.cpp`
  Replace with model-specific binaries or examples that wire a concrete adapter into `runtime::Session`.

- `Transformers/inference/python/pytorch_checkpoint_exporter.py`
  Keep in `Transformers` until the exported artifact contract is stable enough to share. The new canonical contract is documented in `docs/artifact-contract.md`, and the new repo currently keeps compatibility with the old prefix layout.

## Suggested Next Steps

1. Update the Python exporter in `Transformers` to emit bundle directories with `artifact.json`.
2. Add one concrete decoder adapter and one end-to-end smoke test using a small exported checkpoint.
3. Decide whether tokenizer support is optional CMake feature or a separate package dependency.
4. Start separating encoder, decoder, and multimodal weight schemas so one loader does not pretend to fit every model family.
