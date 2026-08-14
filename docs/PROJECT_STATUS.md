# Project status

Status date: 2026-08-14

## Completed and recorded

- Implemented CLIP ViT-B/16 with LoRA adapters and an ArcFace training head.
- Added CLIP-native preprocessing presets and reproducibility seeding.
- Added P x K identity-balanced batch sampling.
- Added optional EdgeFace teacher distillation.
- Connected the official EdgeFace repository as a pinned submodule.
- Executed the hybrid_teacher_qkvo_r8 training notebook for seven configured epochs.
- Restored the best saved in-memory state from epoch 4 and wrote the research checkpoint locally.
- Recorded per-epoch 10-fold benchmark evaluation for LFW, AgeDB-30, CALFW, and CPLFW.

## Latest saved run

| Property | Recorded value |
|---|---|
| Preset | hybrid_teacher_qkvo_r8 |
| Training data | 197,368 images, 540 identities |
| LoRA targets | q, k, v, output projections |
| Trainable parameters | 1.26 million |
| Total parameters | 87.06 million |
| Best epoch | 4 |
| Best mean benchmark accuracy | 87.52% |
| Training duration | 313.8 minutes |
| Execution device | NVIDIA RTX 4060 Laptop GPU |

The values above are extracted from saved notebook outputs. They have not been independently reproduced from a clean clone in this repository revision.

## Open work

1. Execute a clean-clone smoke test and then a complete reproducibility run.
2. Run the full final-evaluation cell after explicitly loading the saved best checkpoint.
3. Rewrite the notebook ablation cell to use build_training_data(cfg) so every newly trained rank uses the same P x K and teacher-distillation method.
4. Preserve a valid model reference after ablation before executing the export cell.
5. Validate exported ONNX numerically against the PyTorch embedding path.
6. Produce a genuinely compact deployment artifact and record its complete file set and hash.
7. Measure latency and memory on the intended physical mobile device.
8. Select a project-level open-source license.

## Known notebook hazards

- The current ablation cell manually builds a conventional shuffled DataLoader. With the hybrid configuration, that omits teacher images and P x K sampling, so its new-rank results would not be method-consistent.
- The same ablation loop deletes model after each rank. A later export cell that assumes model still exists may fail or export the wrong state unless the selected checkpoint is rebuilt and loaded explicitly.
- Notebook outputs are evidence of a prior execution state; they are not a substitute for a clean rerun.
