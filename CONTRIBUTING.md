# Contributing

Contributions that improve reproducibility, evaluation rigor, deployment validation, or documentation are welcome.

Before opening a pull request:

1. Run python scripts/validate_public_repo.py.
2. Do not add datasets, face images, checkpoints, or deployment binaries to Git.
3. State the exact experiment preset, commit, environment, and data snapshot used for numerical results.
4. Keep historical results distinct from the current hybrid method.
5. Include hashes and equivalence tests for any externally hosted model artifact.

Please avoid committing notebook outputs that expose credentials, private paths beyond ordinary development metadata, or identifiable research subjects.
