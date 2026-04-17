# Migration From `Transformers`

This repo starts as a clean extraction target, not a direct file-for-file copy.

## Mapping

- `Transformers/inference/cpp/load_params.cpp`
  The initial extraction work informed the current generic NPZ state-dict path under
  `src/artifacts/npz/state_dict_loader.cpp`. The older Eigen-backed transformer loader has now been
  removed because the active runtime no longer uses it.

- `Transformers/inference/cpp/model.hpp`
  The surviving runtime-facing split is now centered on `include/inference/artifacts/npz/state_dict_loader.hpp`
  plus the task-specific model/runtime headers. The earlier legacy transformer-weight view types
  were removed once they stopped having repo consumers.

- `Transformers/inference/cpp/src/model/executer.*`
  Replace with library-backed apps. Interactive console behavior belongs in `apps/`, not in the reusable runtime.

- `Transformers/inference/cpp/tinystories.cpp`
  Replace with model-specific binaries or examples. Checkpoint-backed apps should prefer
  `runtime::ModelRunner`, while interactive generation flows can still wire a concrete adapter into
  `runtime::Session`.

- `Transformers/inference/python/pytorch_checkpoint_exporter.py`
  Keep in `Transformers` until the exported artifact contract is stable enough to share. The new canonical contract is documented in `docs/artifact-contract.md`, and the new repo currently keeps compatibility with the old prefix layout.

## Suggested Next Steps

1. Update the Python exporter in `Transformers` to emit bundle directories with `artifact.json`.
2. Add one concrete decoder path, either as an artifact-backed `ModelRunner` target or as a
   `runtime::Session` adapter, plus one end-to-end smoke test using a small exported checkpoint.
3. Decide whether tokenizer support is optional CMake feature or a separate package dependency.
4. Keep the generic state-dict loader focused on artifact ingestion and let model-specific structure
   live in builder/runtime code instead of resurrecting one-off legacy weight schemas.
