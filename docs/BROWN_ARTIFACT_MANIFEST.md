# Brown artifact manifest

Inventory of files that exist on the Brown checkout
`/nbu/people/ashah264/simpact` and are not in Git. Measured 2026-11-16.
Sizes are directory totals. None of these paths are tracked.
`docs/results/` is the tracked summary tree (59 files, 3.7 MB) and is not
listed again here.

Do not delete or overwrite these directories. Do not commit them. Do not
upload them to an external service until the research owner approves a
destination. Proposed sharing for people who already have a Brown account:
copy the directories on the shared filesystem (for example `rsync -a` from
`/nbu/people/ashah264/simpact/{results,data,external}`). That copy is a
proposal, not an upload that has been performed.

A clone of <https://github.com/arajshah/simpact> at
`phino-physics-experiments` has the code, the committed push example, and
`docs/results/`. It does not have the rows below.

## Perception weights (`external/`, gitignored, 15 GB)

| Path | Size | Needed to |
|---|---|---|
| `external/sam-3d-objects/` | 12.7 GB | Rerun SAM-3D. Includes `checkpoints/hf/slat_decoder_mesh.ckpt`. License: SAM license in that clone. |
| `external/checkpoints/sam2.1_hiera_large.pt` | 857 MB | Rerun Grounded-SAM-2. |
| `external/FoundationPose/` | 477 MB | Rerun FoundationPose. Includes refiner `weights/2023-10-28-18-33-37/` and scorer `weights/2024-01-11-20-02-45/`. |
| `.cache/huggingface/hub/models--IDEA-Research--grounding-dino-tiny/` | local HF cache | Open-vocab detection. Re-downloaded by `transformers` if missing. |
| `.venv/` | local env | The built rigid stack (`pytorch3d`, `nvdiffrast`, `kaolin`, FoundationPose `mycuda`/`mycpp`). Recreate with `bash scripts/setup_rigid_env.sh` on a machine with CUDA 12.8. |

## Exported datasets (`data/`, gitignored, 215 MB)

| Path | Size | Role |
|---|---|---|
| `data/phino_simpact_v0/` | 719 KB | PHINO contrastive export of the 1 m grid. `train/` and `heldout/`, plus `provenance.json`. Static-image interface smoke. Do not train on its similarity matrix. See `docs/phino_cursor_handoff.md`. |
| `data/phino_simpact_v3obs_v1/` | 114 MB | Earlier action-conditioned render. Friction leaked through sub-0.1 mm start offsets. Do not use. |
| `data/phino_simpact_v3obs_v2/` | 96 MB | Action-conditioned clips used by the Nov 6 learning and baseline pilots. One shared before-image per geometry. Bunny and sphere excluded. |

## Simulation grids and diagnostics (`results/`, gitignored, about 25 GB)

These are generated outputs. The write-ups that interpret them are in
`docs/phino_experiment_log.md` and `docs/results/`. A directory here is the
raw run. A file under `docs/results/` is the summary that was committed.
They are not interchangeable.

| Path | Size | Contents | How to read it |
|---|---|---|---|
| `results/phino_mesh_grid_cutoff_1m_v1/` | 3.9 MB | Current seven-geometry grid. `manifest.json`, `action_trajectories/`, `fingerprints/`, `isolated_scenes/`. 40/63 complete fingerprints. | Source for `data/phino_simpact_v0`. Do not regenerate or edit. |
| `results/phino_mesh_grid_v1/` | 2.7 MB | Earlier grid, 0.25 m displacement cap. | Preserved. Sphere pushes were systematically excluded. |
| `results/phino_mesh_scenes_v1/` | 2.4 MB | Isolated scene assets. `mesh_scene_manifest.json`. | Inputs to the grid generator. |
| `results/phino_multi_object_grid_v1/` | 765 KB | Carton grid. `manifest.json`. | Earlier than the seven-geometry grid. |
| `results/phino_pilot_grid_v1/` | 755 KB | Pilot grid. `manifest.json`. | Pilot, not the export source. |
| `results/phino_pilot/` | 28 KB | Two carton fingerprints. | Pilot. |
| `results/phino_pilot_v2/` | 30 KB | Two fingerprints plus `manifest.json`. | Pilot. |
| `results/phino_calibration/` | 5.2 MB | `calibration.json`, physics sweep trajectories. | Force/friction calibration diagnostics. |
| `results/phino_sphere_diagnostics_v1/` | 4.6 MB | Sphere trajectories and `summary.csv`. | Diagnostic. The 1 m grid already records the sphere outcome. |
| `results/phino_visual_observability_v1/` | 6.3 MB | Rendered push comparisons, `summary.json`. | Figures also in `docs/results/phino_visual_observability_v1/`. |
| `results/phino_settle_audit_v1/` | 132 KB | Ridge audit tables. | Same tables committed under `docs/results/phino_settle_audit_v1/`. |
| `results/phino_v3_identifiability_pilot_v1/` | 39 MB | `traces.npz`, verdicts, summary. Gate failed. | Summary committed under `docs/results/phino_v3_identifiability_pilot_v1/`. |
| `results/phino_reciprocity_feasibility_v1/` | 416 KB | CSVs plus `compliance.npz`. NO-GO. | Summary committed under `docs/results/phino_reciprocity_feasibility_v1/`. |
| `results/phino_action_learning_v1/` | 416 MB | One clip checkpoint and frozen features. Superseded by v2. | Do not cite in place of v2. |
| `results/phino_action_learning_v2/` | 8.1 GB | 15 leave-one-geometry-out runs. Checkpoints are about 336–354 MB each under `runs/*/checkpoint.pt`, plus feature banks. | Metrics in `docs/results/phino_action_learning_v2/`. |
| `results/phino_action_baselines_v1/` | 10.5 GB | InfoNCE and regression baselines. Same checkpoint size class under `runs/*/checkpoint.pt`. | Metrics in `docs/results/phino_action_baselines_v1/`. |
| `results/rigid_0103/` | 20 MB | Nov 4 perception output for `0103_push_0`: masks, crops, scaled meshes, poses, `scene.xml`, `pose_overlay.png`. | Execution log: `results/rigid_0103.log` ends in `RIGID_PIPELINE_OK`. |
| `results/smoke/` | 194 MB | `grounded_sam3d/` and `sam3d_sample/`. | Perception smoke dumps, not a validation report. |
| `results/foundationpose_debug/` | empty | — | Empty directory. |

`results/*.log` files (a few kilobytes each) are the console logs of the
runs above. `*.log` is gitignored everywhere.

## AnyGoal / SMWM diagnostics (`results/`, same tree)

These audits did not train a new world-model loss. Checkpoints below are
readouts or cached banks from the Nov 7–8 diagnostics. The dtype correction
is the current reading of the Insert-Peg forward model: float32 actions, not
the uint8 cast. See `docs/results/anygoal_training_contract_v1/`.

| Path | Size | Contents |
|---|---|---|
| `results/anygoal_contact_audit_v1/` | 188 KB | Stage A–C JSON and CSVs. |
| `results/anygoal_dynamics_factor_v1/` | 105 MB | `bank.npz` and integrator-condition banks. |
| `results/anygoal_insert_anchor_v1/` | 17 MB | Anchor bank. |
| `results/anygoal_insert_baselines_v1/` | 50 MB | Retrieval and short-horizon windows. |
| `results/anygoal_ranking_mechanism_v1/` | 8.0 MB | Original uint8-cast ranking arrays. Kept as the record of the bug. |
| `results/anygoal_training_contract_v1/` | 44 KB | Float32 rescore. The corrected regret is the one in `docs/results/anygoal_training_contract_v1/`. |
| `results/anygoal_state_estimator_v1/` | 3.1 GB | `estimator.pt`, `estimator_full.pt`, `estimator_v1.pt`, frame caches. 2 cm checkpoint was not met. |
| `results/anygoal_goal_pose_v1/` | 1.6 GB | `goal_pose.pt`, `goal_pose_8epoch.pt`, caches. Median errors 2.9 cm hand / 5.2 cm object. |
| `results/anygoal_proprio_baseline_v1/` | 6.6 KB | Ranking JSON only. |

The released SMWM weight `sensorimotor_insertpeg.ckpt` is not in this
SIMPACT tree. The audit pinned it by sha256 in `docs/phino_experiment_log.md`.
Recover it from the SMWM release, not from this repository.

## What a teammate still needs from Brown

To replay the committed push example: nothing beyond the Git clone and
`uv sync --extra dev`.

To rerun rigid perception: the `external/` weights, a rebuilt rigid env, and
a GPU with at least ~24 GB.

To reload a PHINO learning checkpoint or the seven-geometry grid: the matching
row above, plus the separate PHINO checkout at `/nbu/people/ashah264/phino`
when the script takes `--phino-root`.

## Local files that are not artifacts

| Path | What it is |
|---|---|
| `MUJOCO_LOG.TXT` | MuJoCo runtime log in the repo root. Gitignored. |
| `phino_simpact_pilot_v2.py` | Empty file in the repo root. The real script is `scripts/phino_simpact_pilot_v2.py`, which is already tracked. The empty file was not committed. |
