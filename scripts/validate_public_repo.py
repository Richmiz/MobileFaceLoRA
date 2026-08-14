"""Fail-closed structural audit for the public MobileFaceLoRA repository."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 95 * 1024 * 1024
FORBIDDEN_PARTS = {"second run", ".idea", "edgeface-master", "__pycache__"}
FORBIDDEN_SUFFIXES = {
    ".pth",
    ".pt",
    ".ptf",
    ".onnx",
    ".tflite",
    ".ckpt",
    ".safetensors",
    ".pyc",
}
EXPECTED = {
    "README.md",
    "MobileFaceLoRA.ipynb",
    "mobilefacelora_core.py",
    "data/README.md",
    "docs/ARTIFACTS.md",
    "docs/PROJECT_STATUS.md",
    "docs/REPRODUCIBILITY.md",
    "outputs/README.md",
    "requirements.txt",
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def main() -> int:
    errors: list[str] = []
    tracked = tracked_files()
    relative = {path.relative_to(ROOT).as_posix() for path in tracked}

    for expected in sorted(EXPECTED - relative):
        errors.append(f"missing expected tracked file: {expected}")

    for path in tracked:
        rel = path.relative_to(ROOT)
        rel_text = rel.as_posix()
        if FORBIDDEN_PARTS.intersection(rel.parts):
            errors.append(f"forbidden tracked path: {rel_text}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or rel_text.endswith(".onnx.data"):
            errors.append(f"forbidden tracked artifact: {rel_text}")
        if rel.parts and rel.parts[0] == "data" and rel_text != "data/README.md":
            errors.append(f"dataset content is tracked: {rel_text}")
        if path.exists() and path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 95 MiB: {rel_text}")

    for path in tracked:
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"Python parse failed for {path.relative_to(ROOT)}: {exc}")
        elif path.suffix == ".ipynb":
            try:
                notebook = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(notebook.get("cells"), list):
                    raise ValueError("missing cells list")
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                errors.append(f"Notebook parse failed for {path.relative_to(ROOT)}: {exc}")

    if errors:
        print("PUBLIC REPOSITORY AUDIT: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"PUBLIC REPOSITORY AUDIT: PASS ({len(tracked)} tracked entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
