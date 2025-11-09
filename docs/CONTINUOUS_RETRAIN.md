# Continuous Retrain Job (Design Draft)

## Goal

Keep the RTL generator aligned with the latest simulation‑verified behaviour by
harvesting successful pipeline runs on a regular cadence (e.g. nightly) and
feeding them into an incremental fine‑tuning stage.

## High‑Level Flow

```
1. Collect → 2. Curate → 3. Train → 4. Benchmark → 5. Publish
```

1. **Collect traces**
   - Scan `data/runs/*/trace.jsonl` for entries where `stage == "Simulation"` and
     `payload.status == "pass"`.
   - Extract `spec_hash`, `spec_path`, `rtl_file`, `testbench_file`,
     `simulation_compile_log`, `simulation_log` from the payload (added by the new
     orchestrator instrumentation).
   - Copy artefacts into a dated work directory (`work/continuous_retrain/<date>/`).

2. **Curate dataset**
   - De‑duplicate by `spec_hash` (latest run wins).
   - Validate artefact integrity (files exist & compile log contains no warnings
     above configured threshold).
   - Emit JSONL records with metadata:
     ```
     {
       "instruction": "...original spec text...",
       "output": "...rtl...",
       "testbench": "...optional TB...",
       "metadata": {
           "source": "continuous_retrain",
           "spec_hash": "...",
           "category": "...if known...",
           "sim_passed": true,
           "compile_success": true,
           "run_id": "...",
           "generated_at": "...utc iso..."
       }
     }
     ```

3. **Fine‑tune (nightly)**
   - Aggregate the new JSONL with the previous gold datasets (e.g. V4/V5).
   - Run a targeted QLoRA fine‑tune (short epochs) to absorb the delta.
   - Save adapters under `models/qwen_coder_rtl/nightly_<date>/`.

4. **Benchmark gate**
   - Execute the simulation benchmark suite (same as `benchmark_behavioral_model.py`
     + simulation).
   - Compare metrics vs. previous nightly run and the mainline model.
   - Require pass thresholds (e.g. functional ≥ last nightly, FSM ≥ 75 %).

5. **Publish + clean up**
   - On success: update symlink `models/qwen_coder_rtl/latest_nightly`.
   - On failure: archive artefacts for triage and alert the team (Slack/Email).
   - Purge raw copies older than N days after S3/off-site snapshot.

## Automation Hooks

| Stage      | Tooling                                         |
|------------|-------------------------------------------------|
| Collect    | `scripts/collect_passed_traces.py` (TODO)       |
| Curate     | validate with iverilog/vvp + metadata checks    |
| Fine-tune  | reuse `scripts/train_qwen_coder_qlora.py`       |
| Benchmark  | `scripts/benchmark_with_simulation.py`          |
| Publish    | Bash wrapper / GitHub Action / Cron             |

## Integration Notes

- Use cron or a CI runner (GitHub Actions, Jenkins, etc.) scheduled at 02:00 local.
- Parameterize nightly job with environment variables:
  - `CT_RETRAIN_MAX_RECORDS`, `CT_RETRAIN_MIN_DELTA_SCORE`, etc.
- Provide a dry-run flag to inspect what would be trained without actually running.
- Store run metadata (hyperparameters, metrics, adapter path) in `data/run_db/nightly/`.

## Open Questions

1. How much weight should the nightly delta have vs. the stable dataset?
2. Should we mix in failure-derived negative examples (e.g. compile errors with
   fixes) for contrastive tuning?
3. Formal verification integration: once `formal_runner.py` is implemented,
   include formal pass/fail signals in the record metadata.

---

**Next Steps**
1. Implement `collect_passed_traces.py` to materialise the JSONL described above.
2. Wire a cron job (or GitHub Action) that runs collect → curate → train → benchmark.
3. Decide retention policy and notification channel for nightly outcomes.
