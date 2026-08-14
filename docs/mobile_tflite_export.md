# Mobile TFLite export

The deployment artifact is exported from the development checkpoint at
`outputs/mobilefacelora_trained.pth`. It is not derived from the legacy ONNX
file in the mobile application repository.

Run the exporter under Linux or WSL with Python 3.12:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-tflite.txt
python scripts/export_mobile_tflite.py
```

For an isolated overlay, install the TFLite requirements into a local target
and prepend it to `PYTHONPATH`:

```bash
python -m pip install --target .tflite-export-tools -r requirements-tflite.txt
PYTHONPATH=.tflite-export-tools python scripts/export_mobile_tflite.py
```

The generated `.tflite` files remain untracked. The small JSON manifest is
tracked and records checkpoint/artifact hashes, file sizes, tool versions,
the runtime tensor contract, and numerical comparison against PyTorch.

## Runtime contract

- Input: float32 RGB tensor, NCHW `[1, 3, 224, 224]`.
- Scale pixels to `[0, 1]`, then apply CLIP mean and standard deviation from
  the manifest.
- Output: float32 `[1, 512]` L2-normalized embedding.
- Similarity: dot product (cosine similarity for normalized embeddings).
- The ArcFace classification head is not included; LoRA adapters are merged.
- Calibrate the match threshold on deployment-domain validation pairs. Do not
  carry over an old ONNX/app threshold without validating this exported graph.
