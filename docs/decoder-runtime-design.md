# Decoder Runtime Design

This document drafts the next runtime shape for generative decoder models in `Inference`.

The goal is to add decoder support without turning the repo into a generic graph-execution engine.
The proposed design stays consistent with the current architecture:

- exporters produce stable manifest bundles
- `model_builder` validates typed graph metadata and constructs concrete runtime models
- `runtime` owns inference-time state such as KV cache growth and token generation

## Goals

- Add one first-class decoder target: `transformers.decoder_lm`
- Keep manifests typed and architecture-aware instead of inferring arbitrary execution graphs
- Support efficient autoregressive inference with explicit `Prefill()` and `DecodeStep()`
- Move KV cache ownership out of ad-hoc tensor concatenation and into one runtime-managed cache object
- Leave room for grouped-query attention, multi-query attention, RoPE, learned position embeddings, and later VLM decoders

## Non-Goals

- Building a general ONNX-style operator executor
- Encoding runtime-only concepts such as cache pages or decode scheduling directly into the manifest
- Supporting every decoder family in the first cut

## Proposed Layering

- `artifacts`
  - Owns the on-disk manifest contract for decoder exports.
- `model_builder`
  - Validates the decoder graph description and builds one `models::DecoderLM`.
- `models`
  - Owns the concrete decoder model implementation and layer wiring.
- `runtime`
  - Owns `DecoderSession`, `DecoderCache`, prefill/decode flow, and later sampling policy.
- `apps`
  - Stay thin and call runtime-facing APIs instead of managing cache logic directly.

This keeps model construction and generation-time state separate in the same way the current
builder/runner split keeps checkpoint import separate from execution.

## Manifest Direction

The current repo already uses `builder.model_type` plus an optional `builder.graph`. For decoders,
the draft direction is:

- make `builder.model_type` explicit
- use `builder.graph` for validation and structured construction
- stop inferring decoder identity from loose node patterns

The first decoder target should be:

- `transformers.decoder_lm`

### Proposed `model.json`

```json
{
  "builder": {
    "model_type": "transformers.decoder_lm",
    "graph": {
      "version": "inference.graph/2",
      "inputs": ["input_ids", "attention_mask"],
      "outputs": ["logits"],
      "nodes": [
        {
          "name": "token_embedding",
          "op": "token_embedding",
          "inputs": ["input_ids"],
          "outputs": ["embedded_tokens"],
          "param_prefix": "embed_tokens.",
          "attrs": {
            "vocab_size": 32000,
            "embed_dim": 4096
          }
        },
        {
          "name": "rope",
          "op": "rotary_embedding",
          "inputs": ["embedded_tokens"],
          "outputs": ["rotary_context"],
          "attrs": {
            "head_dim": 128,
            "base": 10000.0,
            "max_position_embeddings": 8192
          }
        },
        {
          "name": "decoder",
          "op": "transformer_decoder",
          "inputs": ["embedded_tokens", "attention_mask"],
          "outputs": ["decoded_tokens"],
          "param_prefix": "layers.",
          "attrs": {
            "depth": 32,
            "embed_dim": 4096,
            "num_heads": 32,
            "num_kv_heads": 8,
            "mlp_ratio": 3.5,
            "activation": "swiglu",
            "norm": "rms_norm",
            "position_encoding": "rope",
            "causal": true
          }
        },
        {
          "name": "final_norm",
          "op": "rms_norm",
          "inputs": ["decoded_tokens"],
          "outputs": ["normalized_tokens"],
          "param_prefix": "norm."
        },
        {
          "name": "lm_head",
          "op": "lm_head",
          "inputs": ["normalized_tokens"],
          "outputs": ["logits"],
          "param_prefix": "lm_head.",
          "attrs": {
            "tie_weights": true
          }
        }
      ]
    }
  },
  "config": {
    "name": "decoder_lm",
    "model": {
      "vocab_size": 32000,
      "max_position_embeddings": 8192,
      "embed_dim": 4096,
      "depth": 32,
      "num_heads": 32,
      "num_kv_heads": 8,
      "intermediate_size": 14336,
      "dropout": 0.0,
      "norm_eps": 0.00001,
      "position_encoding": "rope",
      "rope_theta": 10000.0,
      "tie_word_embeddings": true,
      "bos_token_id": 1,
      "eos_token_id": 2,
      "pad_token_id": 0
    }
  }
}
```

### Why This Shape

- `builder.model_type` remains the stable runtime target.
- `builder.graph` remains useful for structural validation and consistent parameter-prefix mapping.
- Runtime-only concerns such as cache capacity, batch reuse, or speculative decoding stay out of the manifest.

## Decoder Graph Operations

The first decoder graph should stay intentionally small. The runtime does not need arbitrary nodes;
it needs a stable set of known module boundaries.

### Required Nodes

- `token_embedding`
- `transformer_decoder`
- `lm_head`

### Optional Helper Nodes

- `position_embedding`
- `rotary_embedding`
- `layer_norm`
- `rms_norm`

### Required Decoder Features

The first implementation should support these layer-level features:

- causal self-attention
- `num_heads` and `num_kv_heads`
- RoPE and learned absolute position embeddings
- `LayerNorm` and `RMSNorm`
- standard MLP and gated MLP variants such as `swiglu`
- tied or untied LM head weights

Cross-attention should stay out of the first `decoder_lm` cut and be added later as a separate
target such as `vlm.decoder_lm`.

## Cache Direction

The current transformer-core cache surface is layer-local:

- `transformer_core::KeyValueCache`

That is still useful as one per-layer forward result, but it is too small a concept to own
runtime-generation state across many decoder layers and many decode steps.

The proposed design introduces two higher-level runtime types:

- `runtime::LayerCache`
- `runtime::DecoderCache`

### Proposed C++ Draft

```cpp
namespace inference::runtime
{

struct LayerCache
{
    transformer_core::Tensor key_storage;
    transformer_core::Tensor value_storage;
    std::int64_t             length   = 0;
    std::int64_t             capacity = 0;
};

struct DecoderCacheConfig
{
    std::int64_t batch_size    = 1;
    std::int64_t num_layers    = 0;
    std::int64_t num_kv_heads  = 0;
    std::int64_t head_dim      = 0;
    std::int64_t max_length    = 0;
};

class DecoderCache
{
public:
    DecoderCache() = default;
    explicit DecoderCache(const DecoderCacheConfig& config);

    core::Status Reserve(const DecoderCacheConfig& config);
    void Reset() noexcept;

    bool loaded() const noexcept;
    std::int64_t length() const noexcept;
    std::int64_t capacity() const noexcept;
    std::size_t num_layers() const noexcept;

    LayerCache& layer(std::size_t layer_index);
    const LayerCache& layer(std::size_t layer_index) const;

    core::Status Append(std::size_t                     layer_index,
                        const transformer_core::Tensor& key_chunk,
                        const transformer_core::Tensor& value_chunk);

private:
    DecoderCacheConfig      config_;
    std::vector<LayerCache> layers_;
    std::int64_t            length_ = 0;
};

}  // namespace inference::runtime
```

### Cache Semantics

- Storage is preallocated once as `[batch, kv_heads, capacity, head_dim]`.
- `Prefill()` appends a full prompt chunk into each layer cache.
- `DecodeStep()` appends one or a few new tokens into each layer cache.
- Cache growth should copy only the new chunk into the reserved storage.
- Existing cached tokens should never be rebuilt by concatenating the entire history at each step.

### Why Not Reuse `KeyValueCache` Directly

`KeyValueCache` should stay as the layer-forward contract returned by transformer-core kernels.
`DecoderCache` is different:

- it owns storage across all layers
- it tracks current sequence length
- it validates batch size, KV head count, and capacity
- it becomes the runtime session state for iterative generation

That split lets low-level kernels stay simple while the runtime owns lifecycle and capacity policy.

## Runtime API Direction

The repo already has two runtime surfaces:

- `runtime::ModelRunner` for checkpoint-backed fixed-shape execution
- `runtime::Session` for adapter-oriented execution

Decoder generation fits the second category better than the first, because it needs persistent
state across many calls. The proposal is to add one dedicated runtime surface:

- `runtime::DecoderSession`

### Proposed C++ Draft

```cpp
namespace inference::runtime
{

struct DecoderPrefillInput
{
    transformer_core::IndexTensor           input_ids;
    std::optional<transformer_core::Tensor> attention_mask;
    std::optional<transformer_core::IndexTensor> position_ids;
    std::int64_t                            max_cache_length = 0;
    bool                                    return_logits    = true;
};

struct DecoderPrefillResult
{
    core::Status             status = core::Status::Ok();
    transformer_core::Tensor logits;
    std::int64_t             prompt_length = 0;
};

struct DecoderStepInput
{
    transformer_core::IndexTensor                input_ids;
    std::optional<transformer_core::IndexTensor> position_ids;
    bool                                         return_logits = true;
};

struct DecoderStepResult
{
    core::Status             status = core::Status::Ok();
    transformer_core::Tensor logits;
    std::int64_t             total_length = 0;
};

class DecoderSession
{
public:
    DecoderSession();

    core::Status Load(const core::ArtifactBundle& artifact);
    core::Status InitializeCache(std::int64_t batch_size,
                                 std::int64_t max_cache_length);
    void Reset() noexcept;

    bool loaded() const noexcept;
    bool cache_initialized() const noexcept;

    DecoderPrefillResult Prefill(const DecoderPrefillInput& input);
    DecoderStepResult DecodeStep(const DecoderStepInput& input);

    const DecoderCache& cache() const;
    const models::DecoderLMConfig& config() const;
};

}  // namespace inference::runtime
```

### Runtime Semantics

- `Load()` builds and stores one `models::DecoderLM`.
- `InitializeCache()` reserves per-layer KV storage for the requested batch and maximum sequence length.
- `Prefill()` runs the prompt chunk and fills the cache from length `0 -> prompt_length`.
- `DecodeStep()` consumes only the newly generated tokens and appends their KV slices.
- `Reset()` drops runtime state without unloading the model.

This separation is important because prompt processing and token-by-token decoding usually have
different performance tradeoffs.

## Relation To `Session` And `ModelAdapter`

The decoder path should not fork the public runtime shape permanently.

The intended progression is:

1. Add `DecoderSession` as the low-level typed generation API.
2. Add one decoder adapter that wraps `DecoderSession`.
3. Expose generic text generation through the existing `Session` + `ModelAdapter` layer.

That gives us:

- one efficient typed path for tests and low-level tools
- one adapter-backed path for generic user-facing inference flows

## Request And Response Growth

The existing `runtime::Request` and `runtime::Response` are already close to what decoder adapters
need. The likely extensions are modest:

- request-side stop-token controls
- response-side finish reason
- optional per-step logits or scores for debugging

Those belong in the adapter-facing layer, not in the low-level cache API.

## Implementation Phases

### Phase 1

- add `models::DecoderLM`
- add `transformers.decoder_lm` manifest support
- add `DecoderCache`
- add `DecoderSession::Prefill()` and `DecoderSession::DecodeStep()`
- support greedy decode only

### Phase 2

- add `RMSNorm` and gated MLP coverage where missing
- add grouped-query attention coverage
- add stop-token handling and finish reasons
- add one adapter-backed text-generation path through `Session`

### Phase 3

- add optional sliding-window cache policy
- add paged-cache or block-cache options if needed
- add sampler policies such as top-k, top-p, and temperature
- add multimodal decoder targets with cross-attention

## Summary

The recommended direction is:

- own our graph builder
- own our KV cache manager
- keep both typed and architecture-aware

That gives the repo a clean path toward decoder support without overcommitting to a generic graph
runtime that would be harder to optimize for autoregressive inference.
