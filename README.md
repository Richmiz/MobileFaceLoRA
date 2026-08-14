# MobileFaceLoRA

Parameter-efficient face recognition using a frozen CLIP ViT backbone, LoRA adapters, an ArcFace training head, identity-balanced sampling, and optional EdgeFace teacher distillation.

This repository is a research workspace intended for continuation and verification. It includes the current source code, notebook, lightweight result figures, and reproducibility documentation. It intentionally excludes face datasets, trained checkpoints, generated deployment binaries, and archived development folders.

## Current development status

The latest recorded run uses the hybrid_teacher_qkvo_r8 preset:

- CLIP ViT-B/16 backbone
- rank-8 LoRA on the query, key, value, and output projections
- P x K sampling with 16 identities and 4 images per identity
- EdgeFace-S (edgeface_s_gamma_05) teacher distillation with weight 0.25
- 197,368 aligned training images across 540 identities
- seven configured epochs on an NVIDIA RTX 4060 Laptop GPU

The notebook completed training and restored epoch 4 as the best checkpoint. Its saved per-epoch evaluation record reports:

| Benchmark | Accuracy | Fold standard deviation |
|---|---:|---:|
| LFW | 98.12% | 0.65% |
| AgeDB-30 | 78.15% | 1.77% |
| CALFW | 85.98% | 1.22% |
| CPLFW | 87.82% | 2.31% |
| Mean | 87.52% | - |

These values come from the saved notebook execution, not a fresh independent rerun. Full final evaluation, a method-consistent ablation, mobile export validation, and physical-device latency evaluation remain open. See [Project status](docs/PROJECT_STATUS.md).

## Repository contents

~~~text
.
|-- MobileFaceLoRA.ipynb       # Current research notebook and saved run record
|-- mobilefacelora_core.py     # Model, data, training, evaluation, and export code
|-- data/README.md             # Expected dataset structure; no data is tracked
|-- outputs/                   # Lightweight figures and artifact documentation
|-- third_party/edgeface       # Pinned official EdgeFace Git submodule
|-- docs/                      # Status and reproducibility notes
|-- scripts/                   # Repository validation utility
+-- requirements.txt           # Recorded direct Python dependencies
~~~

## Clone and set up

Clone recursively so the pinned EdgeFace dependency is available:

~~~bash
git clone --recurse-submodules https://github.com/Richmiz/MobileFaceLoRA.git
cd MobileFaceLoRA
python -m venv .venv
~~~

Activate the environment and install the recorded direct dependencies:

~~~bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
~~~

If the repository was cloned without submodules:

~~~bash
git submodule update --init --recursive
~~~

The recorded experiment used Python 3.12, PyTorch 2.11.0, torchvision 0.26.0, transformers 5.6.2, and PEFT 0.19.1 under WSL with CUDA. PyTorch installation can be platform-specific; use the official PyTorch selector if the pinned wheel is unavailable for your operating system or accelerator.

## Prepare data

No dataset images or annotations are distributed here. Create the structure documented in [data/README.md](data/README.md) after obtaining each dataset under its original terms.

## Run

Open MobileFaceLoRA.ipynb and execute it in order. The current hybrid configuration is guarded in the notebook before training begins. For programmatic use:

~~~python
from mobilefacelora_core import make_experiment_config

cfg = make_experiment_config("hybrid_teacher_qkvo_r8")
~~~

Run the repository audit before committing public changes:

~~~bash
python scripts/validate_public_repo.py
~~~

## Model and deployment artifacts

Checkpoints (.pth, .pt), ONNX external-data files, PTF/TFLite exports, and datasets are ignored. The saved current checkpoint is a full research checkpoint of approximately 348 MB; it is not a compact deployment package. See [Artifact policy](docs/ARTIFACTS.md).

## EdgeFace

EdgeFace is connected to its [official repository](https://github.com/otroshi/edgeface) as a pinned Git submodule rather than copied into this project. EdgeFace retains its own BSD 3-Clause license and attribution.

## License

A project-level license has not yet been selected for the original MobileFaceLoRA code. Until one is added, the repository can be inspected and verified, but reuse rights are not granted beyond applicable law and GitHub's terms. The EdgeFace submodule is governed by its own license.
