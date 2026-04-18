"""
Run one vision-detector checkpoint on one image and save raw plus rendered outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype = np.float32).reshape(3, 1, 1)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype = np.float32).reshape(3, 1, 1)

_LABEL_COLORS = ((46, 125, 50),
                 (21, 101, 192),
                 (173, 20, 87),
                 (230, 81, 0),
                 (106, 27, 154),
                 (2, 119, 189),
                 (85, 139, 47),
                 (198, 40, 40))

_DEFAULT_CLASS_NAMES = ("car",
                        "truck",
                        "bus",
                        "pedestrian",
                        "bicycle",
                        "motorcycle",
                        "traffic_cone",
                        "barrier")


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the detector wrapper.
    Args:
        None.
    Returns:
        Parsed namespace containing checkpoint, image, and output settings.
    """
    parser = argparse.ArgumentParser(
        description = "Wrap run_checkpoint.exe for image-based object detection inference.")

    parser.add_argument("--checkpoint",
                        required = True,
                        type = Path,
                        help = "Path to the PyTorch checkpoint (.pt).")
    parser.add_argument("--image",
                        required = True,
                        type = Path,
                        help = "Path to the input image.")
    parser.add_argument("--output-dir",
                        type = Path,
                        default = None,
                        help = "Directory receiving generated NPZ, JSON, and overlay outputs.")
    parser.add_argument("--run-checkpoint",
                        type = Path,
                        default = None,
                        help = "Optional explicit path to run_checkpoint.exe.")
    parser.add_argument("--python",
                        dest = "python_executable",
                        default = sys.executable,
                        help = "Python interpreter passed through to run_checkpoint.exe.")
    parser.add_argument("--image-size",
                        type = int,
                        default = None,
                        help = "Optional override for the resized square image size.")
    parser.add_argument("--score-threshold",
                        type = float,
                        default = 0.10,
                        help = "Minimum detection score kept after objectness/class fusion.")
    parser.add_argument("--nms-iou-threshold",
                        type = float,
                        default = 0.50,
                        help = "Label-wise NMS IoU threshold.")
    parser.add_argument("--top-k",
                        type = int,
                        default = 50,
                        help = "Maximum number of detections kept after NMS.")
    parser.add_argument("--class-names",
                        default = None,
                        help = "Optional comma-separated class names overriding built-in defaults.")
    parser.add_argument("--artifact-dir",
                        type = Path,
                        default = None,
                        help = "Optional artifact directory passed to run_checkpoint.exe.")
    parser.add_argument("--keep-artifact",
                        action = "store_true",
                        help = "Keep the generated artifact bundle on disk.")

    return parser.parse_args()


def _resolve_run_checkpoint(run_checkpoint_path : Path | None,
                            repo_root           : Path,
                           ) -> Path:
    """
    Resolve the run_checkpoint executable path from an override or common build folders.
    Args:
        run_checkpoint_path : Optional explicit executable path.
        repo_root           : Repository root used for default candidate probing.
    Returns:
        Existing path to ``run_checkpoint.exe``.
    Raises:
        FileNotFoundError : Raised when no usable executable can be found.
    """
    candidates: list[Path] = []

    if run_checkpoint_path is not None:
        candidates.append(run_checkpoint_path)

    candidates.extend([
        repo_root / "build-vs2022" / "Release" / "run_checkpoint.exe",
        repo_root / "build-clean" / "Release" / "run_checkpoint.exe",
        repo_root / "build" / "Release" / "run_checkpoint.exe",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    formatted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError("Unable to locate run_checkpoint.exe. Checked:\n" + formatted)


def _load_checkpoint_metadata(checkpoint_path : Path,
                             ) -> dict[str, Any]:
    """
    Load the raw checkpoint object and expose its metadata mapping.
    Args:
        checkpoint_path : Path to the PyTorch checkpoint file.
    Returns:
        Loaded checkpoint mapping.
    Raises:
        ValueError : Raised when the checkpoint is not a dictionary payload.
    """
    checkpoint = torch.load(checkpoint_path, map_location = "cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must deserialize to a dictionary payload.")
    return checkpoint


def _resolve_image_size(checkpoint_payload : dict[str, Any],
                        image_size_override : int | None,
                       ) -> int:
    """
    Resolve the resized image size expected by the detector.
    Args:
        checkpoint_payload  : Loaded checkpoint mapping.
        image_size_override : Optional CLI override.
    Returns:
        Positive square image size in pixels.
    Raises:
        ValueError : Raised when no valid image size can be determined.
    """
    if image_size_override is not None:
        if image_size_override <= 0:
            raise ValueError("--image-size must be positive.")
        return image_size_override

    model_config = checkpoint_payload.get("model_config", {})
    if isinstance(model_config, dict):
        vision_backbone = model_config.get("vision_backbone", {})
        if isinstance(vision_backbone, dict):
            image_size = vision_backbone.get("image_size")
            if isinstance(image_size, int) and image_size > 0:
                return image_size

    raise ValueError("Unable to resolve image size from the checkpoint. "
                     "Pass --image-size explicitly.")


def _prepare_image_npz(image_path  : Path,
                       output_path : Path,
                       image_size  : int,
                      ) -> None:
    """
    Resize and normalize one image into the detector's NPZ input format.
    Args:
        image_path  : Source RGB image path.
        output_path : Destination ``.npz`` file path.
        image_size  : Target square size expected by the model.
    Returns:
        None.
    """
    resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR

    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size),
                         resample = resample)

    tensor = np.asarray(image, dtype = np.float32).transpose(2, 0, 1) / 255.0
    tensor = ((tensor - _IMAGENET_MEAN) / _IMAGENET_STD)[None, ...].astype(np.float32)

    output_path.parent.mkdir(parents = True, exist_ok = True)
    np.savez(output_path, image = tensor)


def _run_checkpoint(run_checkpoint_path : Path,
                    checkpoint_path     : Path,
                    input_path          : Path,
                    output_path         : Path,
                    python_executable   : str,
                    artifact_dir        : Path | None,
                    keep_artifact       : bool,
                   ) -> None:
    """
    Invoke ``run_checkpoint.exe`` for one detector inference request.
    Args:
        run_checkpoint_path : Path to the executable wrapper.
        checkpoint_path     : Path to the PyTorch checkpoint.
        input_path          : Prepared vision input NPZ file.
        output_path         : Raw output JSON emitted by the executable.
        python_executable   : Python interpreter forwarded to the importer.
        artifact_dir        : Optional explicit artifact directory.
        keep_artifact       : Whether the generated artifact bundle should be preserved.
    Returns:
        None.
    Raises:
        CalledProcessError : Raised when ``run_checkpoint.exe`` fails.
    """
    command = [str(run_checkpoint_path),
               "--checkpoint", str(checkpoint_path),
               "--input", str(input_path),
               "--task", "vision_detector",
               "--output", str(output_path),
               "--python", python_executable]

    if artifact_dir is not None:
        command.extend(["--artifact-dir", str(artifact_dir)])

    if keep_artifact:
        command.append("--keep-artifact")

    subprocess.run(command,
                   check = True)


def _sigmoid(values : np.ndarray,
            ) -> np.ndarray:
    """
    Apply the logistic function elementwise.
    Args:
        values : Input array containing logits.
    Returns:
        Array containing sigmoid outputs.
    """
    return 1.0 / (1.0 + np.exp(-values))


def _softmax(values : np.ndarray,
            ) -> np.ndarray:
    """
    Apply the softmax function along the last axis.
    Args:
        values : Input array containing class logits.
    Returns:
        Array containing class probabilities.
    """
    shifted = values - np.max(values, axis = -1, keepdims = True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis = -1, keepdims = True)


def _canonical_box(box : np.ndarray,
                  ) -> np.ndarray:
    """
    Clamp and order one normalized box as ``[x_min, y_min, x_max, y_max]``.
    Args:
        box : Input array containing four normalized coordinates.
    Returns:
        Normalized box with sorted corners inside `[0, 1]`.
    """
    x0 = float(min(box[0], box[2]))
    y0 = float(min(box[1], box[3]))
    x1 = float(max(box[0], box[2]))
    y1 = float(max(box[1], box[3]))

    return np.array([min(max(x0, 0.0), 1.0),
                     min(max(y0, 0.0), 1.0),
                     min(max(x1, 0.0), 1.0),
                     min(max(y1, 0.0), 1.0)],
                    dtype = np.float32)


def _box_iou(lhs : np.ndarray,
             rhs : np.ndarray,
            ) -> float:
    """
    Compute IoU for two normalized corner boxes.
    Args:
        lhs : Left-hand box in ``[x_min, y_min, x_max, y_max]`` format.
        rhs : Right-hand box in ``[x_min, y_min, x_max, y_max]`` format.
    Returns:
        Intersection-over-union score in `[0, 1]`.
    """
    inter_x0 = max(float(lhs[0]), float(rhs[0]))
    inter_y0 = max(float(lhs[1]), float(rhs[1]))
    inter_x1 = min(float(lhs[2]), float(rhs[2]))
    inter_y1 = min(float(lhs[3]), float(rhs[3]))

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    lhs_area = max(0.0, float(lhs[2]) - float(lhs[0])) * max(0.0, float(lhs[3]) - float(lhs[1]))
    rhs_area = max(0.0, float(rhs[2]) - float(rhs[0])) * max(0.0, float(rhs[3]) - float(rhs[1]))
    union = lhs_area + rhs_area - inter_area

    if union <= 0.0:
        return 0.0
    return inter_area / union


def _label_wise_nms(detections        : list[dict[str, Any]],
                    iou_threshold     : float,
                    top_k             : int | None,
                   ) -> list[dict[str, Any]]:
    """
    Run score-sorted label-wise non-maximum suppression.
    Args:
        detections    : Candidate detections containing ``box``, ``score``, and ``label_id``.
        iou_threshold : Maximum allowed IoU between kept same-label boxes.
        top_k         : Optional cap on the number of returned detections.
    Returns:
        Filtered detections ordered by descending score.
    """
    kept: list[dict[str, Any]] = []

    for label_id in sorted({int(detection["label_id"]) for detection in detections}):
        label_detections = [detection for detection in detections if int(detection["label_id"]) == label_id]
        label_detections.sort(key = lambda item: float(item["score"]),
                              reverse = True)

        while label_detections:
            current = label_detections.pop(0)
            kept.append(current)

            current_box = np.asarray(current["box"], dtype = np.float32)
            survivors: list[dict[str, Any]] = []
            for candidate in label_detections:
                candidate_box = np.asarray(candidate["box"], dtype = np.float32)
                if _box_iou(current_box, candidate_box) <= iou_threshold:
                    survivors.append(candidate)
            label_detections = survivors

    kept.sort(key = lambda item: float(item["score"]),
              reverse = True)

    if top_k is not None:
        kept = kept[:top_k]

    return kept


def _class_name_map(class_count          : int,
                    class_names_override : str | None,
                   ) -> dict[int, str]:
    """
    Build one class-index to name mapping for the rendered overlay.
    Args:
        class_count          : Number of detector classes in the raw output.
        class_names_override : Optional comma-separated override string.
    Returns:
        Mapping from class index to class name.
    """
    if class_names_override:
        names = [item.strip() for item in class_names_override.split(",") if item.strip()]
    elif class_count == len(_DEFAULT_CLASS_NAMES):
        names = list(_DEFAULT_CLASS_NAMES)
    else:
        names = [f"class_{index}" for index in range(class_count)]

    if len(names) < class_count:
        names.extend(f"class_{index}" for index in range(len(names), class_count))

    return {index : names[index] for index in range(class_count)}


def _postprocess_detector_output(raw_payload           : dict[str, Any],
                                 score_threshold       : float,
                                 nms_iou_threshold     : float,
                                 top_k                 : int | None,
                                 class_names_override  : str | None,
                                ) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """
    Convert raw detector logits into scored detections with label-wise NMS.
    Args:
        raw_payload          : Parsed JSON payload produced by ``run_checkpoint.exe``.
        score_threshold      : Minimum score kept before NMS.
        nms_iou_threshold    : IoU threshold used during label-wise NMS.
        top_k                : Optional cap on kept detections.
        class_names_override : Optional comma-separated class-name override.
    Returns:
        Tuple containing filtered detections and the class-id to class-name mapping.
    """
    pred_boxes = np.asarray(raw_payload["pred_boxes"], dtype = np.float32)
    obj_logits = np.asarray(raw_payload["pred_objectness_logits"], dtype = np.float32)
    cls_logits = np.asarray(raw_payload["pred_class_logits"], dtype = np.float32)

    if pred_boxes.ndim != 3 or pred_boxes.shape[0] < 1:
        raise ValueError("pred_boxes must have shape [batch, queries, 4].")
    if obj_logits.ndim != 2 or obj_logits.shape[0] < 1:
        raise ValueError("pred_objectness_logits must have shape [batch, queries].")
    if cls_logits.ndim != 3 or cls_logits.shape[0] < 1:
        raise ValueError("pred_class_logits must have shape [batch, queries, classes].")

    batch_boxes = pred_boxes[0]
    batch_obj = _sigmoid(obj_logits[0])
    batch_cls_probs = _softmax(cls_logits[0])
    batch_labels = np.argmax(batch_cls_probs, axis = -1)
    batch_cls_scores = np.max(batch_cls_probs, axis = -1)
    batch_scores = batch_obj * batch_cls_scores

    class_map = _class_name_map(class_count          = int(batch_cls_probs.shape[-1]),
                                class_names_override = class_names_override)

    detections: list[dict[str, Any]] = []
    for query_index in range(batch_boxes.shape[0]):
        score = float(batch_scores[query_index])
        if score < score_threshold:
            continue

        label_id = int(batch_labels[query_index])
        canonical_box = _canonical_box(batch_boxes[query_index])

        detections.append({"query_index" : int(query_index),
                           "label_id"    : label_id,
                           "label_name"  : class_map.get(label_id, f"class_{label_id}"),
                           "score"       : score,
                           "box"         : canonical_box.tolist()})

    filtered = _label_wise_nms(detections    = detections,
                               iou_threshold = nms_iou_threshold,
                               top_k         = top_k)

    return filtered, class_map


def _measure_text(draw : ImageDraw.ImageDraw,
                  text : str,
                  font : ImageFont.ImageFont,
                 ) -> tuple[int, int]:
    """
    Measure rendered text dimensions for the overlay label plate.
    Args:
        draw : Active PIL drawing context.
        text : Text to measure.
        font : Font used for rendering.
    Returns:
        Pixel width and height.
    """
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0),
                                                 text,
                                                 font = font)
        return right - left, bottom - top
    return draw.textsize(text, font = font)


def _draw_overlay(image_path       : Path,
                  detections       : list[dict[str, Any]],
                  output_path      : Path,
                 ) -> None:
    """
    Draw normalized detections on top of the original RGB image.
    Args:
        image_path  : Source image path used for inference.
        detections  : Postprocessed detection list.
        output_path : Destination overlay image path.
    Returns:
        None.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    image_width, image_height = image.size

    if not detections:
        draw.rectangle((10, 10, 180, 34),
                       fill = (0, 0, 0))
        draw.text((18, 18),
                  "No detections",
                  fill = (255, 255, 255),
                  font = font)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        image.save(output_path)
        return

    for detection in detections:
        label_id = int(detection["label_id"])
        color = _LABEL_COLORS[label_id % len(_LABEL_COLORS)]
        box = np.asarray(detection["box"], dtype = np.float32)

        x_min = float(box[0]) * image_width
        y_min = float(box[1]) * image_height
        x_max = float(box[2]) * image_width
        y_max = float(box[3]) * image_height

        draw.rectangle((x_min, y_min, x_max, y_max),
                       outline = color,
                       width = 3)

        label_text = f"{detection['label_name']}: {float(detection['score']):.2f}"
        text_width, text_height = _measure_text(draw,
                                                label_text,
                                                font)
        label_top = max(y_min - text_height - 8, 0.0)

        draw.rectangle((x_min,
                        label_top,
                        x_min + text_width + 8,
                        label_top + text_height + 8),
                       fill = color)
        draw.text((x_min + 4, label_top + 4),
                  label_text,
                  fill = (255, 255, 255),
                  font = font)

    output_path.parent.mkdir(parents = True, exist_ok = True)
    image.save(output_path)


def _write_processed_summary(output_path          : Path,
                             checkpoint_path      : Path,
                             image_path           : Path,
                             input_npz_path       : Path,
                             raw_json_path        : Path,
                             overlay_path         : Path,
                             artifact_dir         : Path | None,
                             image_size           : int,
                             score_threshold      : float,
                             nms_iou_threshold    : float,
                             top_k                : int | None,
                             detections           : list[dict[str, Any]],
                            ) -> None:
    """
    Write one compact processed summary alongside the raw detector output.
    Args:
        output_path       : Destination summary JSON path.
        checkpoint_path   : Path to the input checkpoint.
        image_path        : Path to the source image.
        input_npz_path    : Path to the generated detector input NPZ.
        raw_json_path     : Path to the raw run_checkpoint JSON.
        overlay_path      : Path to the rendered overlay image.
        artifact_dir      : Optional artifact directory used for the importer.
        image_size        : Resized detector input dimension.
        score_threshold   : Score filter used before NMS.
        nms_iou_threshold : IoU threshold used during NMS.
        top_k             : Optional cap on detections retained.
        detections        : Filtered detections saved in normalized coordinates.
    Returns:
        None.
    """
    payload = {"checkpoint"         : str(checkpoint_path),
               "image"              : str(image_path),
               "input_npz"          : str(input_npz_path),
               "raw_output_json"    : str(raw_json_path),
               "overlay_image"      : str(overlay_path),
               "artifact_dir"       : str(artifact_dir) if artifact_dir is not None else None,
               "image_size"         : image_size,
               "score_threshold"    : score_threshold,
               "nms_iou_threshold"  : nms_iou_threshold,
               "top_k"              : top_k,
               "detections"         : detections}

    output_path.parent.mkdir(parents = True, exist_ok = True)
    output_path.write_text(json.dumps(payload, indent = 2) + "\n",
                           encoding = "utf-8")


def main() -> None:
    """
    Execute the end-to-end image-to-overlay detection wrapper flow.
    Args:
        None.
    Returns:
        None.
    """
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_checkpoint_path = _resolve_run_checkpoint(args.run_checkpoint,
                                                  repo_root)

    checkpoint_path = args.checkpoint.resolve()
    image_path = args.image.resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = (args.output_dir.resolve()
                  if args.output_dir is not None
                  else image_path.parent / "out")
    output_dir.mkdir(parents = True, exist_ok = True)

    checkpoint_payload = _load_checkpoint_metadata(checkpoint_path)
    image_size = _resolve_image_size(checkpoint_payload,
                                     args.image_size)

    image_stem = image_path.stem
    input_npz_path = output_dir / f"{image_stem}.input.npz"
    raw_json_path = output_dir / f"{image_stem}.raw.json"
    processed_json_path = output_dir / f"{image_stem}.detections.json"
    overlay_path = output_dir / f"{image_stem}.overlay.jpg"

    artifact_dir = args.artifact_dir
    if artifact_dir is None and args.keep_artifact:
        artifact_dir = output_dir / f"{checkpoint_path.stem}_artifact"
    if artifact_dir is not None:
        artifact_dir = artifact_dir.resolve()

    print("[1/4] Preparing normalized image tensor...", flush = True)
    _prepare_image_npz(image_path  = image_path,
                       output_path = input_npz_path,
                       image_size  = image_size)

    print("[2/4] Running run_checkpoint.exe...", flush = True)
    _run_checkpoint(run_checkpoint_path = run_checkpoint_path,
                    checkpoint_path     = checkpoint_path,
                    input_path          = input_npz_path,
                    output_path         = raw_json_path,
                    python_executable   = args.python_executable,
                    artifact_dir        = artifact_dir,
                    keep_artifact       = args.keep_artifact)

    if not raw_json_path.exists():
        raise FileNotFoundError("run_checkpoint.exe finished without creating the raw JSON output: "
                                + str(raw_json_path))

    print("[3/4] Postprocessing detections...", flush = True)
    raw_payload = json.loads(raw_json_path.read_text(encoding = "utf-8"))
    detections, _ = _postprocess_detector_output(raw_payload          = raw_payload,
                                                 score_threshold      = args.score_threshold,
                                                 nms_iou_threshold    = args.nms_iou_threshold,
                                                 top_k                = args.top_k,
                                                 class_names_override = args.class_names)

    print("[4/4] Rendering overlay image...", flush = True)
    _draw_overlay(image_path  = image_path,
                  detections  = detections,
                  output_path = overlay_path)

    _write_processed_summary(output_path       = processed_json_path,
                             checkpoint_path   = checkpoint_path,
                             image_path        = image_path,
                             input_npz_path    = input_npz_path,
                             raw_json_path     = raw_json_path,
                             overlay_path      = overlay_path,
                             artifact_dir      = artifact_dir if args.keep_artifact else None,
                             image_size        = image_size,
                             score_threshold   = args.score_threshold,
                             nms_iou_threshold = args.nms_iou_threshold,
                             top_k             = args.top_k,
                             detections        = detections)

    print("Done.", flush = True)
    print(f"  input npz     : {input_npz_path}", flush = True)
    print(f"  raw json      : {raw_json_path}", flush = True)
    print(f"  detections    : {processed_json_path}", flush = True)
    print(f"  overlay image : {overlay_path}", flush = True)
    print(f"  kept boxes    : {len(detections)}", flush = True)
    if args.keep_artifact and artifact_dir is not None:
        print(f"  artifact dir  : {artifact_dir}", flush = True)


if __name__ == "__main__":
    main()
