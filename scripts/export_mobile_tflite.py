#!/usr/bin/env python3
"""Export the trained MobileFaceLoRA development checkpoint to LiteRT/TFLite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mobilefacelora_core import (  # noqa: E402
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    build_model,
    make_experiment_config,
)


PRESET = "hybrid_teacher_qkvo_r8"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "mobilefacelora_trained.pth"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mobile_export"
FP32_NAME = "mobilefacelora_hybrid_r8_fp32.tflite"
INT8_NAME = "mobilefacelora_hybrid_r8_weight_int8.tflite"
MANIFEST_NAME = "mobilefacelora_manifest.json"


class MobileEmbedding(nn.Module):
    """Deployment graph: merged CLIP-ViT, projection, and L2 normalization."""

    def __init__(self, backbone: nn.Module, projection: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = projection

    def forward(self, face_image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=face_image).pooler_output
        embedding = self.projection(features)
        return F.normalize(embedding, p=2.0, dim=1, eps=1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Export and validate only the FP32 TFLite model.",
    )
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_lf(path: Path) -> str:
    text_content = path.read_text(encoding="utf-8")
    canonical = text_content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def enable_quantizer_name_compatibility() -> None:
    """Bridge AI Edge Quantizer 0.8 tensor names to LiteRT 2.1.6."""
    from ai_edge_quantizer.transformations import dequant_insert

    original = dequant_insert.insert_dequant
    if getattr(original, "_mobilefacelora_name_compat", False):
        return

    def insert_dequant_compatible(transformation_input: Any) -> Any:
        tensor = transformation_input.subgraph.tensors[
            transformation_input.tensor_id
        ]
        if isinstance(tensor.name, str):
            tensor.name = tensor.name.encode("utf-8")
        return original(transformation_input)

    insert_dequant_compatible._mobilefacelora_name_compat = True  # type: ignore[attr-defined]
    dequant_insert.insert_dequant = insert_dequant_compatible


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None
def git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())




def make_sample(input_size: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(42)
    rgb = torch.rand((1, 3, input_size, input_size), generator=generator)
    mean = torch.tensor(CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD).view(1, 3, 1, 1)
    return (rgb - mean) / std


def load_deployment_model(checkpoint: Path) -> tuple[MobileEmbedding, dict[str, Any]]:
    print(f"[Load] Checkpoint: {checkpoint}", flush=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "arcface.weight" not in state:
        raise ValueError("Expected a MobileFaceLoRA state dict with arcface.weight")

    num_classes, embed_dim = state["arcface.weight"].shape
    cfg = make_experiment_config(PRESET)
    if int(embed_dim) != int(cfg["embed_dim"]):
        raise ValueError(
            f"Checkpoint embedding width {embed_dim} != config {cfg['embed_dim']}"
        )

    model = build_model(cfg, int(num_classes), torch.device("cpu"))
    model.load_state_dict(state, strict=True)
    model.eval()

    if not hasattr(model.backbone, "merge_and_unload"):
        raise TypeError("Expected a PEFT backbone with merge_and_unload()")
    model.backbone = model.backbone.merge_and_unload()
    model.eval()

    deployment_model = MobileEmbedding(model.backbone, model.projection).eval()
    metadata = {
        "preset": PRESET,
        "identity_classes_in_training_head": int(num_classes),
        "input_size": int(cfg["input_size"]),
        "embedding_dimensions": int(cfg["embed_dim"]),
        "clip_model": cfg["clip_model"],
        "lora_rank": int(cfg["lora_rank"]),
        "lora_alpha": int(cfg["lora_alpha"]),
        "lora_targets": list(cfg["lora_targets"]),
    }
    print("[Load] Strict checkpoint load: PASS", flush=True)
    print("[Load] LoRA merge: PASS", flush=True)
    return deployment_model, metadata


def run_litert(path: Path, sample: np.ndarray, num_threads: int) -> dict[str, Any]:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
    interpreter.allocate_tensors()
    signatures = interpreter.get_signature_list()

    if signatures:
        signature_name = next(iter(signatures))
        signature = signatures[signature_name]
        input_name = signature["inputs"][0]
        output_name = signature["outputs"][0]
        outputs = interpreter.get_signature_runner(signature_name)(
            **{input_name: sample}
        )
        output = outputs[output_name]
    else:
        signature_name = None
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        input_name = input_detail["name"]
        output_name = output_detail["name"]
        interpreter.set_tensor(input_detail["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    return {
        "output": np.asarray(output),
        "signature": signature_name,
        "input_name": input_name,
        "input_shape": [int(value) for value in input_detail["shape"]],
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_name": output_name,
        "output_shape": [int(value) for value in output_detail["shape"]],
        "output_dtype": np.dtype(output_detail["dtype"]).name,
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_flat = reference.reshape(-1).astype(np.float64)
    candidate_flat = candidate.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat)
    cosine = float(np.dot(reference_flat, candidate_flat) / denominator)
    absolute = np.abs(reference_flat - candidate_flat)
    return {
        "cosine_similarity": cosine,
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "output_l2_norm": float(np.linalg.norm(candidate_flat)),
    }


def artifact_record(
    path: Path, runtime: dict[str, Any], metrics: dict[str, float]
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "runtime_contract": {
            key: value for key, value in runtime.items() if key != "output"
        },
        "numerical_validation": metrics,
    }


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_metadata = load_deployment_model(checkpoint)
    sample = make_sample(model_metadata["input_size"])
    with torch.inference_mode():
        reference = model(sample).cpu().numpy()

    print("[Convert] Importing LiteRT Torch...", flush=True)
    import litert_torch

    print("[Convert] Exporting FP32 LiteRT model...", flush=True)
    edge_model = litert_torch.convert(
        model,
        sample_kwargs={"face_image": sample},
        strict_export="auto",
        lightweight_conversion=True,
        enable_x64=False,
        runtime_constant_folding=False,
    )
    fp32_path = output_dir / FP32_NAME
    edge_model.export(str(fp32_path))

    sample_np = sample.numpy().astype(np.float32, copy=False)
    fp32_runtime = run_litert(fp32_path, sample_np, args.num_threads)
    fp32_metrics = compare(reference, fp32_runtime["output"])
    if fp32_runtime["input_shape"] != [1, 3, 224, 224]:
        raise RuntimeError(f"Unexpected FP32 input shape: {fp32_runtime['input_shape']}")
    if fp32_runtime["output_shape"] != [1, 512]:
        raise RuntimeError(f"Unexpected FP32 output shape: {fp32_runtime['output_shape']}")
    if fp32_metrics["cosine_similarity"] < 0.9999:
        raise RuntimeError(f"FP32 numerical validation failed: {fp32_metrics}")
    print(f"[Verify] FP32 cosine: {fp32_metrics['cosine_similarity']:.8f}", flush=True)

    artifacts = {
        "fp32": artifact_record(fp32_path, fp32_runtime, fp32_metrics),
    }

    if not args.skip_quantization:
        from ai_edge_quantizer import quantizer, recipe
        enable_quantizer_name_compatibility()

        int8_path = output_dir / INT8_NAME
        print("[Quantize] Applying weight-only INT8 / FP32 activations...", flush=True)
        qt = quantizer.Quantizer(str(fp32_path))
        qt.load_quantization_recipe(recipe.weight_only_wi8_afp32())
        qt.quantize(serialize_to_path=int8_path, enable_progress_report=True)
        int8_runtime = run_litert(int8_path, sample_np, args.num_threads)
        int8_metrics = compare(reference, int8_runtime["output"])
        if int8_runtime["input_shape"] != [1, 3, 224, 224]:
            raise RuntimeError(f"Unexpected INT8 input shape: {int8_runtime['input_shape']}")
        if int8_runtime["output_shape"] != [1, 512]:
            raise RuntimeError(f"Unexpected INT8 output shape: {int8_runtime['output_shape']}")
        if int8_metrics["cosine_similarity"] < 0.99:
            raise RuntimeError(f"INT8 numerical validation failed: {int8_metrics}")
        print(f"[Verify] INT8 cosine: {int8_metrics['cosine_similarity']:.8f}", flush=True)
        artifacts["weight_int8"] = artifact_record(
            int8_path, int8_runtime, int8_metrics
        )

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": git_commit(),
        "repository_dirty_at_export": git_is_dirty(),
        "exporter": {
            "path": "scripts/export_mobile_tflite.py",
            "sha256_lf": sha256_text_lf(Path(__file__).resolve()),
        },
        "source_checkpoint": {
            "path": str(checkpoint.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "model": model_metadata,
        "deployment_contract": {
            "input_layout": "NCHW",
            "input_shape": [1, 3, 224, 224],
            "input_dtype": "float32",
            "input_color_order": "RGB",
            "input_value_range_before_normalization": [0.0, 1.0],
            "normalization_mean": CLIP_IMAGE_MEAN,
            "normalization_std": CLIP_IMAGE_STD,
            "output_shape": [1, 512],
            "output_dtype": "float32",
            "output_l2_normalized": True,
            "similarity": "cosine similarity (dot product of normalized embeddings)",
            "decision_threshold": None,
            "decision_threshold_note": (
                "Calibrate on deployment-domain validation pairs before release."
            ),
            "arcface_training_head_included": False,
            "lora_adapters_merged": True,
        },
        "toolchain": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "litert_torch": package_version("litert-torch"),
            "ai_edge_litert": package_version("ai-edge-litert"),
            "ai_edge_quantizer": package_version("ai-edge-quantizer"),
        },
        "validation_sample": "seeded synthetic RGB image, seed=42",
        "artifacts": artifacts,
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[Done] Manifest: {manifest_path}", flush=True)
    for name, record in artifacts.items():
        print(
            f"[Done] {name}: {record['path']} ({record['bytes']:,} bytes)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
