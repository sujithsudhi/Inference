"""
Import one raw PyTorch checkpoint into an artifact bundle for the C++ graph builder.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_ENCODER_CONFIG = {"activation"        : "gelu",
                          "attention_dropout" : 0.1,
                          "cls_head_dim"      : 128,
                          "drop_path"         : 0.0,
                          "dropout"           : 0.1,
                          "layer_norm_eps"    : 1e-5,
                          "mlp_ratio"         : 2.0,
                          "num_heads"         : 4,
                          "pooling"           : "cls",
                          "pre_norm"          : True,
                          "qkv_bias"          : True,
                          "rope_base"         : 10000,
                          "use_cls_token"     : True,
                          "use_rope"          : False}

DEFAULT_VISION_CONFIG = {"backbone" : {"attention_dropout" : 0.0,
                                       "dropout"           : 0.0,
                                       "global_rope_base"  : 1_000_000,
                                       "layer_norm_eps"    : 1e-6,
                                       "local_rope_base"   : 10_000,
                                       "local_window_size" : 7,
                                       "mlp_ratio"         : 4.0,
                                       "num_heads"         : 8,
                                       "qkv_bias"          : True,
                                       "use_rope"          : True},
                         "head"     : {"dropout"        : 0.0,
                                       "layer_norm_eps" : 1e-5}}


def _count_indexed_prefix(state_dict : dict[str, torch.Tensor],
                          prefix     : str,
                         ) -> int:
    """
    Count distinct numeric indices under a dotted state-dict prefix.
    Args:
        state_dict : Mapping of checkpoint parameter names to tensors.
        prefix     : Prefix such as ``encoder.`` or ``backbone.blocks.``.
    Returns:
        Number of distinct module indices discovered under the prefix.
    """
    indices: set[int] = set()

    for key in state_dict.keys():
        if not key.startswith(prefix):
            continue

        index_str = key[len(prefix):].split(".", maxsplit = 1)[0]
        if index_str.isdigit():
            indices.add(int(index_str))

    return len(indices)


def _find_tensor(state_dict : dict[str, torch.Tensor],
                 key        : str,
                ) -> torch.Tensor | None:
    """
    Return one tensor from the state dict when it exists.
    Args:
        state_dict : Mapping of checkpoint parameter names to tensors.
        key        : Exact parameter name to look up.
    Returns:
        The tensor when present, otherwise ``None``.
    """
    return state_dict.get(key)


def _require_tensor(state_dict : dict[str, torch.Tensor],
                    key        : str,
                   ) -> torch.Tensor:
    """
    Return one tensor from the state dict or raise when it is missing.
    Args:
        state_dict : Mapping of checkpoint parameter names to tensors.
        key        : Exact parameter name to look up.
    Returns:
        The requested tensor.
    Raises:
        ValueError : Raised when the tensor does not exist.
    """
    tensor = _find_tensor(state_dict, key)
    if tensor is None:
        raise ValueError(f"Missing tensor '{key}' in checkpoint state_dict.")
    return tensor


def _extract_state_dict(checkpoint : Any) -> dict[str, torch.Tensor]:
    """
    Extract a PyTorch-style state dict from a loaded checkpoint object.
    Args:
        checkpoint : Object returned by ``torch.load(...)``.
    Returns:
        Dictionary mapping parameter names to CPU tensors.
    Raises:
        ValueError : Raised when the checkpoint does not expose a state dict.
    """
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        raw_state_dict = checkpoint
    else:
        raise ValueError("Checkpoint does not contain a supported state_dict payload.")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.detach().cpu().to(dtype = torch.float32)

    if not state_dict:
        raise ValueError("Checkpoint state_dict is empty.")

    return state_dict


def _infer_task(state_dict : dict[str, torch.Tensor],
                task_hint  : str | None,
               ) -> str:
    """
    Resolve which supported runtime family the checkpoint belongs to.
    Args:
        state_dict : Mapping of checkpoint parameter names to tensors.
        task_hint  : Optional explicit hint from the CLI.
    Returns:
        Stable internal task name for the importer.
    Raises:
        ValueError : Raised when the checkpoint cannot be mapped to a supported family.
    """
    if task_hint:
        normalized = task_hint.strip().lower().replace("-", "_")
        if normalized in {"encoder_classifier", "imdb", "sentiment"}:
            return "encoder_classifier"
        if normalized in {"vision_detector", "detector", "object_detection"}:
            return "vision_detector"
        raise ValueError(f"Unsupported task hint '{task_hint}'.")

    if "token_embedding.weight" in state_dict and _count_indexed_prefix(state_dict, "encoder.") > 0:
        return "encoder_classifier"

    if "backbone.patch_embed.proj.weight" in state_dict and "detection_head.query_embed" in state_dict:
        return "vision_detector"

    raise ValueError("Unable to infer a supported model family from the checkpoint.")


def _encoder_classifier_config(checkpoint : dict[str, Any],
                               state_dict : dict[str, torch.Tensor],
                              ) -> dict[str, Any]:
    """
    Build one resolved encoder-classifier config from checkpoint tensors.
    Args:
        checkpoint : Loaded checkpoint payload.
        state_dict : Mapping of parameter names to tensors.
    Returns:
        Serializable config dictionary for the C++ runtime.
    """
    config = dict(DEFAULT_ENCODER_CONFIG)

    token_embedding = _require_tensor(state_dict, "token_embedding.weight")
    position = _find_tensor(state_dict, "position.positional_table")
    cls_token = _find_tensor(state_dict, "cls_token")
    fc1_weight = _find_tensor(state_dict, "encoder.0.residual_mlp.module.fc1.weight")
    head0_weight = _find_tensor(state_dict, "head.0.weight")
    head3_weight = _find_tensor(state_dict, "head.3.weight")
    head_weight = _find_tensor(state_dict, "head.weight")

    config["vocab_size"] = int(token_embedding.shape[0])
    config["embed_dim"] = int(token_embedding.shape[1])
    config["depth"] = _count_indexed_prefix(state_dict, "encoder.")
    config["max_length"] = int(position.shape[1]) if position is not None else 512
    config["use_cls_token"] = cls_token is not None
    config["use_rope"] = position is None
    config["qkv_bias"] = _find_tensor(state_dict, "encoder.0.residual_attention.module.w_q.bias") is not None

    if fc1_weight is not None:
        config["mlp_hidden_dim"] = int(fc1_weight.shape[0])
        config["mlp_ratio"] = float(fc1_weight.shape[0]) / float(token_embedding.shape[1])

    if head0_weight is not None:
        config["cls_head_dim"] = int(head0_weight.shape[0])
    else:
        config["cls_head_dim"] = None

    if head3_weight is not None:
        config["num_outputs"] = int(head3_weight.shape[0])
    elif head_weight is not None:
        config["num_outputs"] = int(head_weight.shape[0])
    else:
        config["num_outputs"] = 1

    embedded_model_config = checkpoint.get("model_config")
    if isinstance(embedded_model_config, dict):
        for key in ("num_heads",
                    "dropout",
                    "attention_dropout",
                    "layer_norm_eps",
                    "pooling",
                    "pre_norm",
                    "rope_base",
                    "use_rope",
                    "use_cls_token"):
            if key in embedded_model_config:
                config[key] = embedded_model_config[key]

    return config


def _vision_detector_config(checkpoint : dict[str, Any],
                            state_dict : dict[str, torch.Tensor],
                           ) -> dict[str, Any]:
    """
    Build one resolved detector config from checkpoint tensors and embedded metadata.
    Args:
        checkpoint : Loaded checkpoint payload.
        state_dict : Mapping of parameter names to tensors.
    Returns:
        Serializable config dictionary for the C++ runtime.
    """
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        model_config = {}

    backbone_cfg = dict(DEFAULT_VISION_CONFIG["backbone"])
    head_cfg = dict(DEFAULT_VISION_CONFIG["head"])

    patch_weight = _require_tensor(state_dict, "backbone.patch_embed.proj.weight")
    pos_embed = _require_tensor(state_dict, "backbone.pos_embed")
    query_embed = _require_tensor(state_dict, "detection_head.query_embed")
    class_weight = _require_tensor(state_dict, "detection_head.class_head.1.weight")
    box_hidden = _find_tensor(state_dict, "detection_head.box_head.1.weight")
    ffn_hidden = _find_tensor(state_dict, "detection_head.ffn.0.weight")

    vision_backbone = model_config.get("vision_backbone", {})
    local_block = model_config.get("local_block", {})
    global_block = model_config.get("global_block", {})

    attention_cfg = local_block.get("attention", {}) if isinstance(local_block, dict) else {}
    mlp_cfg = local_block.get("mlp", {}) if isinstance(local_block, dict) else {}

    use_cls_token = bool(vision_backbone.get("use_cls_token", "backbone.cls_token" in state_dict))
    patch_tokens = int(pos_embed.shape[1]) - (1 if use_cls_token else 0)
    grid_size = int(round(patch_tokens ** 0.5))
    image_size = int(vision_backbone.get("image_size", patch_weight.shape[2] * grid_size))

    backbone_cfg["image_size"] = image_size
    backbone_cfg["patch_size"] = int(vision_backbone.get("patch_size", patch_weight.shape[2]))
    backbone_cfg["in_channels"] = int(vision_backbone.get("in_channels", patch_weight.shape[1]))
    backbone_cfg["embed_dim"] = int(attention_cfg.get("embedding_dim", patch_weight.shape[0]))
    backbone_cfg["num_layers"] = int(vision_backbone.get("num_layers", _count_indexed_prefix(state_dict, "backbone.blocks.")))
    backbone_cfg["num_heads"] = int(attention_cfg.get("num_heads", backbone_cfg["num_heads"]))
    backbone_cfg["attention_dropout"] = float(attention_cfg.get("dropout", backbone_cfg["attention_dropout"]))
    backbone_cfg["dropout"] = float(mlp_cfg.get("dropout", backbone_cfg["dropout"]))
    backbone_cfg["qkv_bias"] = bool(attention_cfg.get("qkv_bias", backbone_cfg["qkv_bias"]))
    backbone_cfg["use_cls_token"] = use_cls_token
    backbone_cfg["layer_norm_eps"] = float(vision_backbone.get("layer_norm_eps", backbone_cfg["layer_norm_eps"]))
    backbone_cfg["block_pattern"] = list(vision_backbone.get("block_pattern", ["local", "local", "local", "global"]))

    if "hidden_dim" in mlp_cfg:
        backbone_cfg["mlp_hidden_dim"] = int(mlp_cfg["hidden_dim"])
    elif "backbone.blocks.0.residual_mlp.module.fc1.weight" in state_dict:
        backbone_cfg["mlp_hidden_dim"] = int(state_dict["backbone.blocks.0.residual_mlp.module.fc1.weight"].shape[0])
    else:
        backbone_cfg["mlp_hidden_dim"] = int(backbone_cfg["embed_dim"] * backbone_cfg["mlp_ratio"])

    backbone_cfg["mlp_ratio"] = float(mlp_cfg.get("mlp_ratio",
                                                  float(backbone_cfg["mlp_hidden_dim"]) / float(backbone_cfg["embed_dim"])))

    if isinstance(global_block, dict) and global_block.get("rope_base_override") is not None:
        backbone_cfg["global_rope_base"] = int(global_block["rope_base_override"])

    if isinstance(local_block, dict) and local_block.get("rope_base_override") is not None:
        backbone_cfg["local_rope_base"] = int(local_block["rope_base_override"])

    if isinstance(local_block, dict) and local_block.get("window_size_override") is not None:
        backbone_cfg["local_window_size"] = int(local_block["window_size_override"])

    head_cfg["num_queries"] = int(query_embed.shape[1])
    head_cfg["num_classes"] = int(class_weight.shape[0])
    head_cfg["num_heads"] = int(attention_cfg.get("num_heads", backbone_cfg["num_heads"]))
    head_cfg["dropout"] = float(mlp_cfg.get("dropout", head_cfg["dropout"]))
    head_cfg["layer_norm_eps"] = float(model_config.get("classifier", {}).get("layer_norm_eps",
                                                                              head_cfg["layer_norm_eps"]))

    if ffn_hidden is not None:
        head_cfg["mlp_hidden_dim"] = int(ffn_hidden.shape[0])
    elif box_hidden is not None:
        head_cfg["mlp_hidden_dim"] = int(box_hidden.shape[0])
    else:
        head_cfg["mlp_hidden_dim"] = int(backbone_cfg["embed_dim"])

    return {"backbone" : backbone_cfg,
            "head"     : head_cfg}


def _encoder_graph(model_config : dict[str, Any]) -> dict[str, Any]:
    """
    Build graph metadata for the encoder-classifier runtime.
    Args:
        model_config : Resolved model config dictionary.
    Returns:
        Serializable graph section for ``model.json``.
    """
    nodes: list[dict[str, Any]] = [{"name"         : "token_embedding",
                                    "op"           : "token_embedding",
                                    "inputs"       : ["input_ids"],
                                    "outputs"      : ["embedded_tokens"],
                                    "param_prefix" : "token_embedding.",
                                    "attrs"        : {"vocab_size" : model_config["vocab_size"],
                                                      "embed_dim"  : model_config["embed_dim"]}}]

    if model_config.get("use_cls_token", False):
        nodes.append({"name"         : "cls_token",
                      "op"           : "cls_token",
                      "outputs"      : ["cls_token_tensor"],
                      "param_prefix" : "cls_token",
                      "attrs"        : {"use_cls_token" : True}})

    if not model_config.get("use_rope", True):
        nodes.append({"name"         : "position",
                      "op"           : "positional_encoding",
                      "inputs"       : ["embedded_tokens"],
                      "outputs"      : ["positioned_tokens"],
                      "param_prefix" : "position.",
                      "attrs"        : {"max_length" : model_config["max_length"],
                                        "use_rope"   : False}})

    encoder_input = "positioned_tokens" if not model_config.get("use_rope", True) else "embedded_tokens"
    nodes.append({"name"         : "encoder",
                  "op"           : "transformer_encoder",
                  "inputs"       : [encoder_input, "attention_mask"],
                  "outputs"      : ["encoded_tokens"],
                  "param_prefix" : "encoder.",
                  "attrs"        : {"depth"             : model_config["depth"],
                                    "num_heads"         : model_config["num_heads"],
                                    "mlp_ratio"         : model_config["mlp_ratio"],
                                    "mlp_hidden_dim"    : model_config.get("mlp_hidden_dim"),
                                    "dropout"           : model_config["dropout"],
                                    "attention_dropout" : model_config["attention_dropout"],
                                    "qkv_bias"          : model_config["qkv_bias"],
                                    "pre_norm"          : model_config["pre_norm"],
                                    "layer_norm_eps"    : model_config["layer_norm_eps"],
                                    "drop_path"         : model_config["drop_path"],
                                    "pooling"           : model_config["pooling"],
                                    "use_rope"          : model_config["use_rope"],
                                    "rope_base"         : model_config["rope_base"]}})
    nodes.append({"name"         : "norm",
                  "op"           : "layer_norm",
                  "inputs"       : ["encoded_tokens"],
                  "outputs"      : ["normalized_tokens"],
                  "param_prefix" : "norm."})
    nodes.append({"name"         : "classifier",
                  "op"           : "classifier_head",
                  "inputs"       : ["normalized_tokens"],
                  "outputs"      : ["logits"],
                  "param_prefix" : "head.",
                  "attrs"        : {"cls_head_dim" : model_config.get("cls_head_dim"),
                                    "num_outputs"  : model_config["num_outputs"]}})

    return {"version" : "inference.graph/1",
            "inputs"  : ["input_ids", "attention_mask"],
            "outputs" : ["logits"],
            "nodes"   : nodes}


def _vision_graph(model_config : dict[str, Any]) -> dict[str, Any]:
    """
    Build graph metadata for the vision-detector runtime.
    Args:
        model_config : Resolved detector config dictionary.
    Returns:
        Serializable graph section for ``model.json``.
    """
    backbone_cfg = model_config["backbone"]
    head_cfg = model_config["head"]

    return {"version" : "inference.graph/1",
            "inputs"  : ["image"],
            "outputs" : ["pred_boxes", "pred_objectness_logits", "pred_class_logits"],
            "nodes"   : [{"name"         : "patch_embed",
                          "op"           : "patch_embedding",
                          "inputs"       : ["image"],
                          "outputs"      : ["patch_tokens"],
                          "param_prefix" : "backbone.patch_embed.",
                          "attrs"        : {"image_size"  : backbone_cfg["image_size"],
                                            "patch_size"  : backbone_cfg["patch_size"],
                                            "in_channels" : backbone_cfg["in_channels"],
                                            "embed_dim"   : backbone_cfg["embed_dim"]}},
                         {"name"         : "backbone",
                          "op"           : "vision_backbone",
                          "inputs"       : ["patch_tokens"],
                          "outputs"      : ["vision_features"],
                          "param_prefix" : "backbone.",
                          "attrs"        : {"num_layers"        : backbone_cfg["num_layers"],
                                            "num_heads"         : backbone_cfg["num_heads"],
                                            "mlp_ratio"         : backbone_cfg["mlp_ratio"],
                                            "mlp_hidden_dim"    : backbone_cfg["mlp_hidden_dim"],
                                            "attention_dropout" : backbone_cfg["attention_dropout"],
                                            "dropout"           : backbone_cfg["dropout"],
                                            "qkv_bias"          : backbone_cfg["qkv_bias"],
                                            "use_cls_token"     : backbone_cfg["use_cls_token"],
                                            "use_rope"          : backbone_cfg["use_rope"],
                                            "layer_norm_eps"    : backbone_cfg["layer_norm_eps"],
                                            "local_window_size" : backbone_cfg["local_window_size"],
                                            "local_rope_base"   : backbone_cfg["local_rope_base"],
                                            "global_rope_base"  : backbone_cfg["global_rope_base"],
                                            "block_pattern"     : backbone_cfg["block_pattern"]}},
                         {"name"         : "head",
                          "op"           : "detection_head",
                          "inputs"       : ["vision_features"],
                          "outputs"      : ["pred_boxes", "pred_objectness_logits", "pred_class_logits"],
                          "param_prefix" : "detection_head.",
                          "attrs"        : {"num_queries"    : head_cfg["num_queries"],
                                            "num_classes"    : head_cfg["num_classes"],
                                            "num_heads"      : head_cfg["num_heads"],
                                            "mlp_hidden_dim" : head_cfg["mlp_hidden_dim"],
                                            "dropout"        : head_cfg["dropout"],
                                            "layer_norm_eps" : head_cfg["layer_norm_eps"]}}]}


def _copy_tokenizer(tokenizer_path : Path,
                    output_dir     : Path,
                   ) -> str:
    """
    Copy one tokenizer artifact into the output bundle and return its manifest-relative path.
    Args:
        tokenizer_path : Source tokenizer file supplied by the caller.
        output_dir     : Artifact directory receiving the copied tokenizer.
    Returns:
        Relative path recorded in ``artifact.json``.
    Raises:
        ValueError : Raised when the tokenizer format is unsupported.
    """
    tokenizer_dir = output_dir / "tokenizer"
    tokenizer_dir.mkdir(parents = True, exist_ok = True)

    if tokenizer_path.suffix == ".json":
        destination = tokenizer_dir / "tokenizer.json"
    elif tokenizer_path.suffix == ".model":
        destination = tokenizer_dir / "tokenizer.model"
    else:
        raise ValueError(f"Unsupported tokenizer file '{tokenizer_path}'. "
                         "Expected .json or .model.")

    shutil.copy2(tokenizer_path, destination)
    return str(destination.relative_to(output_dir))


def _write_artifact(output_dir     : Path,
                    checkpoint     : Path,
                    task           : str,
                    model_config   : dict[str, Any],
                    graph          : dict[str, Any],
                    state_dict     : dict[str, torch.Tensor],
                    tokenizer_path : Path | None,
                   ) -> None:
    """
    Write one manifest bundle for the imported checkpoint.
    Args:
        output_dir    : Artifact directory to populate.
        checkpoint    : Original checkpoint path for source metadata.
        task          : Stable internal task name.
        model_config  : Runtime config dictionary.
        graph         : Graph metadata dictionary.
        state_dict     : Mapping of checkpoint parameter names to tensors.
        tokenizer_path : Optional tokenizer file to copy into the bundle.
    """
    output_dir.mkdir(parents = True, exist_ok = True)

    weights_path = output_dir / "weights.npz"
    model_path = output_dir / "model.json"
    manifest_path = output_dir / "artifact.json"
    copied_tokenizer = _copy_tokenizer(tokenizer_path, output_dir) if tokenizer_path is not None else None

    np.savez(weights_path,
             **{key : tensor.detach().cpu().numpy()
                for key, tensor in state_dict.items()})

    model_payload = {"builder"         : {"model_type" : "graph",
                                          "graph"      : graph},
                     "config"          : {"name"  : task,
                                          "model" : model_config},
                     "format"          : "npz",
                     "source"          : {"framework"  : "pytorch",
                                          "checkpoint" : str(checkpoint),
                                          "tokenizer"  : str(tokenizer_path) if tokenizer_path is not None else None},
                     "state_dict_keys" : sorted(state_dict.keys())}

    files = {"metadata" : "model.json",
             "weights"  : "weights.npz"}
    if copied_tokenizer is not None:
        files["tokenizer"] = copied_tokenizer

    manifest_payload = {"schema_version" : "inference.artifact/1",
                        "artifact_name"  : checkpoint.stem,
                        "model_family"   : "vision-language" if task == "vision_detector" else "encoder",
                        "task"           : "vision" if task == "vision_detector" else "text",
                        "weight_format"  : "npz",
                        "files"          : files}

    model_path.write_text(json.dumps(model_payload, indent = 2) + "\n", encoding = "utf-8")
    manifest_path.write_text(json.dumps(manifest_payload, indent = 2) + "\n", encoding = "utf-8")


def main() -> None:
    """
    Import one supported raw PyTorch checkpoint into an artifact bundle.
    """
    parser = argparse.ArgumentParser(description = "Import one raw PyTorch checkpoint into an inference artifact bundle.")
    parser.add_argument("--checkpoint",
                        required = True,
                        help     = "Path to the raw .pt checkpoint file.")
    parser.add_argument("--output-dir",
                        required = True,
                        help     = "Directory that will receive artifact.json, model.json, and weights.npz.")
    parser.add_argument("--task",
                        default = None,
                        help    = "Optional task hint: encoder_classifier or vision_detector.")
    parser.add_argument("--tokenizer",
                        default = None,
                        help    = "Optional tokenizer.json or tokenizer.model to copy into the artifact.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    tokenizer_path = Path(args.tokenizer) if args.tokenizer is not None else None

    if checkpoint_path.is_dir():
        raise ValueError("tools/import_pytorch_checkpoint.py expects a raw checkpoint file, "
                         "not an artifact directory. Pass the artifact directly to run_checkpoint instead.")

    checkpoint = torch.load(checkpoint_path, map_location = "cpu")
    state_dict = _extract_state_dict(checkpoint)
    task = _infer_task(state_dict, args.task)

    if task == "encoder_classifier":
        model_config = _encoder_classifier_config(checkpoint, state_dict)
        graph = _encoder_graph(model_config)
    else:
        model_config = _vision_detector_config(checkpoint, state_dict)
        graph = _vision_graph(model_config)

    _write_artifact(output_dir   = output_dir,
                    checkpoint   = checkpoint_path,
                    task         = task,
                    model_config = model_config,
                    graph        = graph,
                    state_dict   = state_dict,
                    tokenizer_path = tokenizer_path)


if __name__ == "__main__":
    main()
