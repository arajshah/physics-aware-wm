# SIMPACT research agent instructions

This repository contains SIMPACT plus the SIMPACT → PHINO physics-representation experiments. Start with the README. The data contract is `docs/phino_pretraining_integration_contract.md`. The experiment record is `docs/phino_experiment_log.md`. On initial onboarding, `docs/phino_initial_handoff.md` is a historical snapshot from 2026-11-05; consult it only for that history. Source code and manifests take precedence over those notes. PHINO's training code is the `external/phino` submodule (`arajshah/phino`).

## Working rules

- Make one focused change per task. Inspect existing implementations before adding new ones. Prefer reusable adapters and tests over another simulation or model rewrite.
- Use descriptive commit messages such as `feat(phino): export validated SIMPACT fingerprints for training`; include a concise verification summary in the final response/PR.
- Do not overwrite any existing `results/`, grid manifests, checkpoints, or research outputs. Use new output directories and keep generated data, credentials, weights, and large artifacts out of Git.
- Respect manifest-declared action order, physics values, rejected statuses, and provenance. Never silently admit rejected action traces or fabricate missing data.
- Do not change validated action forces, cutoff defaults, mass/friction grids, validation rules, or evaluation splits without an explicit experiment specification.
- Keep dataset splits grouped by geometry; do not use test data to fit normalizers, calibrate hyperparameters, select features, or choose among methods.
- Keep static-image training claims separate from recoverability using privileged simulated trajectories. Identical images with different arbitrarily assigned mass/friction do **not** reveal those hidden properties.
- Code must fail clearly on invalid shapes, missing assets, inconsistent identifiers, non-finite values, duplicate entries, or wrong normalization/action counts.
- Add lightweight tests and self-tests for adapters. Run tests available in the current environment; accurately report what you could not execute. GPU/MuJoCo checks on Brown may require a human-run command.
- Do not launch long training or expensive simulation sweeps unless requested. The dataset export and the action-conditioned pilots in the experiment log are already done. Do not regenerate those grids or retrain those runs for a documentation change.
- When a research design choice is ambiguous or could make results scientifically invalid, document the issue and request human review rather than hiding it behind code.

## Research records

- Record reproducible experiments in `docs/phino_experiment_log.md`, including hypothesis, dataset and split, code revision, metrics, limitations, and conclusions. Do not fabricate results.
- Keep the README as the current setup and result summary. Preserve `docs/phino_initial_handoff.md` as a historical snapshot. Do not overwrite it or add a second onboarding document.

## Completed adapter

`scripts/export_phino_dataset.py` is the adapter from the seven-geometry grid to PHINO's dataset contract. See `docs/phino_pretraining_integration_contract.md`. Preserve the original grid. The static-image export is not the action-conditioned training result.