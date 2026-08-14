# Artifact policy

Datasets, trained checkpoints, and deployment binaries are excluded from Git history. This keeps the repository cloneable, respects dataset distribution constraints, and avoids presenting unverified binaries as public release artifacts.

## Latest local research checkpoint

The checkpoint produced by the saved hybrid notebook execution was inspected in the development workspace before public-repository construction:

| Field | Value |
|---|---|
| Local path | outputs/mobilefacelora_trained.pth |
| Size | 348,419,836 bytes |
| SHA-256 | 10F82994B1C948264162A502A01C1E33BAB8B0F07BB28F3F7912A21EB3F00C82 |
| Publicly tracked | No |

This is a full research checkpoint, not a compact mobile package.

## Other excluded artifacts

Other .pth, .ptf, .onnx, and .onnx.data files from development runs are excluded from the public repository.

## Deployment-size caution

An earlier notebook output reported a small ONNX file while related exports used external tensor data. A valid deployment-size claim must count the complete set of files required for inference and must be tied to a hash-identified artifact. This repository does not currently publish or validate a self-contained compact deployment model.

## Recommended release procedure

Before publishing a model artifact:

1. Rebuild it from a named commit and recorded environment.
2. Compare embeddings numerically against the source PyTorch checkpoint.
3. Evaluate all benchmark metrics after conversion or quantization.
4. Record every required file, byte size, SHA-256 hash, input/output contract, and preprocessing definition.
5. Upload large binaries to a versioned model registry or release service rather than normal Git history.
