# Why these files are included

Scope: binary T1/T2 deterioration and coverage-failure monitoring of HS, EWMA, GARCH-t and QRF, as described by the supplied draft and implemented in the current repository. The manuscript is a scope reference; its embedded remaining-work items did not authorize changes to methodology or manuscript text.

## Included

- 28 existing scripts covering VaR reconstruction/corrections, regimes, causal feature/target panels, RS04b preflight, the main monitor, inference, comparisons, calibration, alerts, case studies, stronger-volatility comparison, ablation and Indian data acquisition.
- The exact raw U.S./VIX archive required by the upstream builders.
- Current RS01, RS03, RS04 and RS04b panels, outcomes and audit/manifest files. The five large CSV panels are losslessly compressed.
- Current RS05 result families, including draft-era main predictions and newer final predictions. Both are necessary until their lineage is reconciled.
- H5/H20, permutation controls, XGBoost-containing final outputs, ablation evidence and alert-budget/bootstrap variants.
- Indian acquisition data once, in the main data tree.
- The unmodified supplied draft and recorded environment/provenance.
- A new run_monitor.py launcher that writes main-monitor runs into fresh directories.

## Excluded from the new folder

Older conformal and exploratory branches, the separate continuous-target branch, RS02 exploratory event stages that are not dependencies of the retained pipeline, historical panel backups, dated duplicate output trees, source/result archive copies, cache files, old smoke runs, superseded exante COVID files, the narrative review notebook, the Git database and the installed virtual environment.

These exclusions only affect the new curated folder. All original files remain in D:\reliability_layer and in the full backup previously made.

## Code and data checks

Copied research code differs only in project-root resolution and compressed CSV filenames. An AST comparison verifies this restricted change set for all 28 scripts. Original-to-packaged file hashes are in file_manifest.json/CSV; the compressed-panel entries additionally match the original bytes after decompression.

The packaging regression test ran HS/logit, T1/T2, S4/S6 for 2020Q1 in scratch directories. All 5,952 prediction rows and 13 columns match exactly between the original and packaged code. This is a packaging test, not a full research verification.

## Important distinctions

The folder is the curated current working project, not a claim that the paper is finished. The draft and some saved results reflect earlier stages; existing run metadata can have historical/overwritten configuration. Preserving a result does not certify it.

The original virtual environment is reused for immediate execution. requirements.txt enables a separate installation later, which has not yet been tested. The copied research code and required data resolve within this folder rather than reading the old project.

Some historical scripts remain self-contained and share output filenames. The new launcher protects main-monitor results with fresh output directories; it does not redesign every upstream or evaluation script. Use their explicit output arguments or an isolated copy before a full reproduction.
