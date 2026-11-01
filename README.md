# RS05: forward-looking VaR forecast-loss monitoring

RS05 asks whether information available at a forecast origin predicts a future period of **normalized VaR forecast-loss deterioration**. The canonical corrected empirical release is **Priority8 P8.1**, conditionally ready for a narrowly scoped manuscript with specified revisions.

## Scientific interpretation

Four fixed one-day 5% VaR families are evaluated: Historical Simulation (HS), EWMA, GARCH(1,1)-Student-t, and QRF. T1 flags a future H observed-session average normalized 5% pinball loss above the expanding 75th percentile of fully matured historical H-window losses,requiring 252 historical windows. Future forecasts enter losses when actually issued; there is no substitution of current VaR for every future date. T2 is a separate future breach-count event exceeding the 95% Binomial(H,.05) quantile.

Normalized loss is **not a proper cross-model VaR ranking score**. T1 is not proof of regulatory failure,coverage adequacy,causal effects or economic losses. Primary strict-CV S6-S4 AUROC gains survive for all four US VaR families. Some probability-score,normalization and secondary comparisons are null or negative; QRF reduced-monitor exceptions remain. System-warning superiority is unestablished and economic usefulness unproven. India tracks use local retraining on a selected surviving cohort,not frozen US-to-India transfer. Historical point-in-time membership,delistings,corporate-action/source-vintage and intraday-availability limitations remain.

## Start here

- `results/canonical/catalog.json`: one SHA256-verified pointer catalog for corrected predictions,tables,figures and reports. Large artifacts are not recopied.
- `paper/README.md`: journal materials and their status.
- `docs/RESEARCH_HISTORY.md`: scientific lineage and superseded results.
- `archive/ARCHIVE_INDEX.md`: immutable releases retained **at original paths** to preserve frozen commands and manifests.
- `docs/repository_audit/`: complete inventory,external backup verification,cleanup and integrity records.

## Environment and commands

Use PowerShell from the repository root and the existing `.venv/Scripts/python.exe`. The exact observed scientific versions are in `requirements.txt` and the P8 environment record. No dependency upgrade or replacement environment was performed. A clean-environment installation has not been certified.

```powershell
.\.venv\Scripts\python.exe -B scripts/rs05.py --help
.\.venv\Scripts\python.exe -B scripts/rs05.py status
.\.venv\Scripts\python.exe -B scripts/rs05.py artifact table/US_PRIMARY_INFERENCE --verify
.\.venv\Scripts\python.exe -B scripts/rs05.py verify
.\.venv\Scripts\python.exe -B tests/test_repository.py
.\.venv\Scripts\python.exe -B scripts/rs05.py smoke --out runs/my_unique_smoke
.\.venv\Scripts\python.exe -B scripts/rs05.py run-strict --out runs/my_unique_fit --phase primary
```

Outputs must be fresh paths. `smoke` independently refits one representative complete fold including tuning/calibration and compares it with immutable P8 evidence. `run-strict` clones only frozen source/config/input partitions into a new output root and runs the strict chronological pipeline. The changed output-root text produces an intentionally new driver digest. It does not refit upstream VaRs. `--prepare-only` prepares a run without fitting. `scripts/verify_repository.ps1` runs adapter regressions and the frozen release verifier. Optional editable installation is `python -m pip install --no-deps -e .`; this was not required or performed.

The old `run_monitor.py` is preserved for historical compatibility; it is **not** the canonical harmonized Priority8 pipeline.

## Architecture and availability

`src/rs05` provides importable adapters for data,features,targets,monitoring,calibration,evaluation and inference. They verify and delegate to a single indexed frozen implementation,avoiding competing numerical rewrites. `configs/` identifies the exact frozen protocol; `experiments/` documents domain entry points; `results/` indexes canonical artifacts; `paper/` indexes journal materials; `docs/` contains methodology/reproducibility; `archive/` records complete historical releases.

Full market data,models,environments and historical release payloads remain local. Redistribution rights have not been established. The Git repository is a local code/documentation shell; it is **not yet a self-contained public reproduction dataset**. No files are pushed or remotely uploaded. Git status and privacy review are documented; no scientific data are staged or committed.

## Recovery and integrity

See `docs/repository_audit/BACKUP_VERIFICATION.json` and `RESTORE_INSTRUCTIONS.md` for the independent external snapshot. Unique evidence and raw data are retained. Only verified backed-up disposable caches may be removed; all removals have per-file hashes/recovery paths. No frozen release is relocated or its contents edited. See `docs/CLEANUP_REPORT.md` for exact before/after counts and limitations.
