# External Dataset Licensing Checklist

| Dataset | Source / Contact | License (to confirm) | Action Items |
|---------|------------------|----------------------|--------------|
| SecFSM (2025) | Paper + release repo (FSM benchmark) | **TBD** (check arXiv supplemental / GitHub) | - Download archive under `data/external/secfsm/` once license allows.<br>- Save LICENSE text as `data/external/secfsm/LICENSE.txt`.<br>- Record attribution requirements in `external_sources.json`. |
| VeriThoughts (2025) | Formal + sim dataset (arXiv 2505.xxxx) | **TBD** | - Locate official repo (likely GitHub).<br>- Verify if MIT/Apache or custom research license.<br>- Note any “no commercial use” clauses. |
| CVDP (2024) | Cocotb protocol benchmark | **TBD** | - Check project page / scoreboard repo.<br>- If under BSD/MIT, mirror raw tests.<br>- Capture cocotb license compatibility. |
| VeriPrefer (2023) | RL dataset (testbench feedback) | **TBD** | - Confirm release terms (IEEE/publisher).<br>- Determine if evaluation scripts are redistributable. |

## Process
1. Locate the official repository / release page for each dataset.
2. Copy the exact LICENSE text into `data/external/<dataset>/LICENSE.txt`.
3. Add a short README in each folder describing the provenance and any usage restrictions.
4. Update `data/expanded_training/EXTERNAL_SOURCES.md` once licensing is confirmed.
5. Before using in continuous retrain, ensure metadata includes `metadata.source` pointing to the dataset.

> **Note:** Without confirmed license text we cannot ingest these datasets into the training pipeline. Please complete the checklist above before running the conversion scripts.
