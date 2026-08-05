# Local DLRM Patch Notes

The prototype currently uses the local DLRM repo at:

```text
C:\Users\han\Desktop\dlrm
```

To support the design document's cold start, resume, and evaluate workflow, the local
`torchrec_dlrm/dlrm_main.py` has been patched with checkpoint smoke support:

- `--checkpoint_save_dir`
- `--checkpoint_load_path`
- `--checkpoint_save_optimizer`
- `model.pt`
- optional `optimizer.pt`
- `metadata.json`
- `latest.json`

Check whether the local DLRM file contains the required patch:

```powershell
cd C:\Users\han\Desktop\prototype
powershell -ExecutionPolicy Bypass -File scripts/check_dlrm_checkpoint_patch.ps1
```

Current boundary:

- This patch is suitable for single-process smoke validation.
- Production multi-process sharded checkpointing still needs Torch Distributed Checkpoint.
