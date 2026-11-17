# Physics-aware world models

Research code for learning representations of physical behaviour from visual observations and simulated interactions. The simulation stack is [SIMPACT](https://arxiv.org/abs/2512.05955). The contrastive encoder, trajectory normalizer, and SoftInfoNCE loss come from PHINO, pinned as a submodule at `external/phino`.

The implemented path is:

```text
RGB-D observation → reconstruction → physical simulation → trajectory generation → representation training → linear-probe evaluation
```

A predictive world model and a planner that uses these representations are not implemented. SIMPACT's original test-time loop (`scripts/optimize.py`: propose, rollout, verify, regress) is still in the tree. It plans with a vision-language model and the simulator. It does not train the encoder studied below.

## Pipeline

SIMPACT turns one recorded RGB-D frame into a simulatable scene and rolls actions forward.

- Rigid objects: Grounded-SAM-2 masks, a SAM-3D mesh, FoundationPose 6-DoF, then MuJoCo. Driver: `scripts/run_rigid_pipeline.py`. Notes: [docs/RIGID_PIPELINE.md](docs/RIGID_PIPELINE.md), [docs/RIGID_ENV_SETUP.md](docs/RIGID_ENV_SETUP.md).
- Rope, dough, and sweep use the deformable builders in `simpact/real2sim/`. Those tasks are part of the upstream planner. The representation experiments below use rigid contacts.

The physics grids assign mass and friction in simulation. They are not measurements of real objects. A static image shared by several mass/friction assignments does not identify which assignment produced a trajectory. The training set that the tables below use is action-conditioned: `data/phino_simpact_v3obs_v2`. Each clip has a shared before-image and frames after a known action.

PHINO supplies the DINOv2-B/14 encoder (`dinov2_vitb14`, CLS concatenated with mean patch token), the projection head, `TrajectoryNormalizer`, and SoftInfoNCE (`external/phino/train/`). This repository calls that code with `--phino-root` (default: the submodule) and does not copy it into `simpact/`.

## Repository structure

```text
physics-aware-wm/
├── simpact/            # simulation, real2sim, perception, execution
├── scripts/            # reconstruction, grids, export, training, probes
├── tests/
├── examples/           # committed scenes and recorded rollouts
├── assets/             # camera calibration and the Franka gripper model
├── prompts/            # vision-language prompt templates
├── docs/               # pipeline notes and experiment reports
│   └── results/        # committed figures and metric tables
├── external/phino/     # PHINO submodule (encoder, loss, dataloader)
├── pyproject.toml
├── uv.lock
└── .gitmodules
```

`data/`, `results/`, and third-party weights under `external/` (other than the PHINO submodule) are gitignored. Committed numerical reports live in `docs/results/`.

## Setup

Python is pinned to 3.11.15 (`.python-version`). The GPU stack in `pyproject.toml` is `torch==2.9.0+cu128` and `warp-lang==1.10.0`. Use that CUDA 12.8 build. A cu118 torch wheel has no Blackwell kernels.

```bash
git clone --recurse-submodules https://github.com/arajshah/physics-aware-wm.git
cd physics-aware-wm
uv sync --extra dev
cp .env.example .env
```

`uv sync` creates `.venv`. Commands below use `.venv/bin/python`. `uv run python ...` is the same interpreter.

Headless MuJoCo rendering:

```bash
export MUJOCO_GL=egl
```

`GOOGLE_API_KEY` is required only for the vision-language propose / verify / regress loop. Replay, dataset export, and the representation experiments do not call that API.

PHINO is imported from `external/phino` on `sys.path`. The scripts resolve it in this order: an explicit `--phino-root`, then the submodule when `external/phino/train/data/mujoco_engine.py` exists, then `PHINO_ROOT`. Do not install a second PHINO environment for the commands in this file. The recorded runs used the SIMPACT environment.

```bash
git submodule update --init external/phino
```

The submodule is pinned to `3f1134d523559e52adce08bdac8e0b711c3c722c` (`eval/grouped-physics-probes-oct28`). That commit is the checkout named in the experiment provenance. Its contrastive loader matches `main` of the same fork.

Import `open3d` before `coacd`. `simpact/real2sim/convex_decomp.py` already does this. `frankx`, `pyrealsense2`, and `warp` are optional imports.

### Third-party weights

These are not in Git. Perception weights are not redistributed here.

| Piece | Role | How to obtain it |
|---|---|---|
| SAM 2.1 hiera large | masks | [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt). Set `SIMPACT_SAM2_CHECKPOINT`. |
| GroundingDINO tiny | open-vocabulary boxes | `IDEA-Research/grounding-dino-tiny`, downloaded by `transformers` on first use. |
| SAM 3D Objects | image to mesh | Clone [facebookresearch/sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) and download checkpoints from [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) after requesting access. Set `SIMPACT_SAM3D_DIR`. |
| FoundationPose | 6-DoF pose | Clone [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose). Weights: [public Drive folder](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) (`weights/2023-10-28-18-33-37` refiner, `weights/2024-01-11-20-02-45` scorer). Set `SIMPACT_FOUNDATIONPOSE_DIR`. |
| DINOv2-B/14 | encoder backbone | Downloaded at runtime by `torch.hub` from `facebookresearch/dinov2`. Not stored in this repository. |
| kaolin | SAM-3D mesh extraction | Built by `scripts/setup_rigid_env.sh`. |

A fresh FoundationPose clone needs its two-line torch 2.9 patch before the `mycuda` build (`-std=c++14` to `-std=c++17`, and `.type()` to `.scalar_type()`). See [docs/RIGID_ENV_SETUP.md](docs/RIGID_ENV_SETUP.md) and [docs/VALIDATION_rigid.md](docs/VALIDATION_rigid.md).

```bash
bash scripts/setup_rigid_env.sh
uv sync --extra rigid --extra dev
```

After that script, a bare `uv sync` removes the source-built rigid packages. Resync with `uv sync --extra rigid --extra dev`, then rerun the script.

### Hardware

Replay of the committed push example and the CPU test suite run without the perception weights. The rigid reconstruction was executed on an L40 (peak 18.74 GB, `RIGID_PIPELINE_OK` in the Brown log `results/rigid_0103.log`). Use a GPU with about 24 GB before rerunning `scripts/run_rigid_pipeline.py`. An 8 GB GPU can load MuJoCo and import the perception adapters. It cannot rerun SAM-3D. Contrastive training uses CUDA and downloads DINOv2-B/14.

## Reproduction

Lightweight checks are (1) and (7), and `--validate-only` in (3) once the grid is on disk. (2), (4), (5), and (6) need a GPU. (4)–(6) also need `data/phino_simpact_v3obs_v2`. None of these commands use an absolute machine path.

### 1. Replay the committed push example

No extra download. The scene is `examples/push_real2sim/0103_push_0/`.

```bash
export MUJOCO_GL=egl
.venv/bin/python scripts/replay_rollout.py --task push \
  --scene examples/push_real2sim/0103_push_0 \
  --plan examples/push_real2sim/0103_push_0/runs/refined_plan.json \
  --out_dir /tmp/push_replay \
  --check examples/push_real2sim/0103_push_0/runs/final_rollout/rollout_00.json
```

Expected tail:

```text
ok   blue_milk_carton: final-position delta 0.000 cm
ok   white_coconut_milk_carton: final-position delta 0.000 cm
replay matches the recorded rollout
```

This checks that the committed meshes, calibration, and recorded plan reproduce the committed final positions. It does not rerun reconstruction. The replay writes `/tmp/push_replay/` and does not modify `examples/`.

Upstream closed-loop reproduction for push, rope, dough, and sweep is `bash scripts/reproduce_all.sh`. It writes to a timestamped directory under `/tmp`. Do not pass `--overwrite-examples`.

### 2. Rigid reconstruction

Requires the weights in the table above and a GPU with about 24 GB. Not run as part of publishing this repository.

```bash
.venv/bin/python scripts/run_rigid_pipeline.py \
  --data_dir examples/push_real2sim/0103_push_0 \
  --objects "white coconut milk carton. blue milk carton." \
  --cam 1 \
  --K assets/calibration/0103/cam1_K.txt \
  --out_dir /tmp/rigid_demo \
  --xml /tmp/rigid_demo/scene.xml
```

Success is the line `RIGID_PIPELINE_OK` and a `scene.xml` MuJoCo can load. The push planner does not consume that XML directly. It rebuilds the robot-frame scene from `{name}_scaled.obj`, `{name}_6d_cam1.txt`, and the `0103` extrinsic (`simpact/executor/push_scene.py`). The committed `examples/push_real2sim/0103_push_0/sim/` files are what replay uses.

Documented pose and shape figures, from 2026-06-22 on an RTX 5090, are in [docs/VALIDATION_rigid.md](docs/VALIDATION_rigid.md). They were not remeasured for this repository. `scripts/validate_rigid_pipeline.py` is named in that note and is not in the tree.

### 3. PHINO-compatible data

Static-image interface export from the 1 m grid. This directory is `data/phino_simpact_v0`. Do not train on its similarity matrix: settle orientation contaminates some within-geometry pairs, and cross-geometry pairs mostly follow mesh body-frame conventions. Details: [docs/phino_experiment_log.md](docs/phino_experiment_log.md).

```bash
.venv/bin/python scripts/export_phino_dataset.py --validate-only \
  --grid-dir results/phino_mesh_grid_cutoff_1m_v1

MUJOCO_GL=egl .venv/bin/python scripts/export_phino_dataset.py \
  --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
  --phino-root external/phino \
  --output-dir data/phino_simpact_v0
```

The export refuses an existing output directory. Pass only `train/` to PHINO pretraining. `TrajectoryNormalizer(4, 50)` is fit on `train/` and copied into `heldout/`.

Action-conditioned clips used for the tables below:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/phino_v3_observation_dataset.py \
  --pilot-dir results/phino_v3_identifiability_pilot_v1 \
  --grid-dir results/phino_mesh_grid_cutoff_1m_v1 \
  --phino-root external/phino \
  --output-dir data/phino_simpact_v3obs_v2
```

The command replays saved Stage 3 traces. It does not resimulate the grid. The script exits if the output directory exists, and if the grid manifest hash differs from the pilot summary. Do not point training at `data/phino_simpact_v3obs_v1`.

### 4. Trajectory SoftInfoNCE

One leave-one-geometry-out run. The published pilot trained all five geometries at seeds 0 and 1 for 400 steps. Checkpoints are external; see [Data and checkpoints](#data-and-checkpoints).

```bash
.venv/bin/python scripts/phino_action_contrastive_train.py \
  --objective trajectory --mode clip --seed 0 \
  --test-geometry mesh_cube \
  --data-dir data/phino_simpact_v3obs_v2 \
  --phino-root external/phino \
  --output-dir /tmp/phino_traj_cube_s0
```

Fixed hyperparameters, also used by the baselines: batch of 24 clips, backbone learning rate `1e-5`, head learning rate `1e-3`, weight decay `0.05`, 40-step warmup, last four blocks trainable. The output directory must not already exist. The first run downloads DINOv2-B/14.

### 5. Standard InfoNCE and supervised regression

Same clips, budget, and augmentation. `--objective instance` is symmetric InfoNCE and does not read trajectory similarity. `--objective regression` predicts training-fold standardized log mass and log friction. The regression head is discarded before the probe.

```bash
.venv/bin/python scripts/phino_action_contrastive_train.py \
  --objective instance --mode clip --seed 0 \
  --test-geometry mesh_cube \
  --data-dir data/phino_simpact_v3obs_v2 \
  --phino-root external/phino \
  --output-dir /tmp/phino_infonce_cube_s0

.venv/bin/python scripts/phino_action_contrastive_train.py \
  --objective regression --mode clip --seed 0 \
  --test-geometry mesh_cube \
  --data-dir data/phino_simpact_v3obs_v2 \
  --phino-root external/phino \
  --output-dir /tmp/phino_regression_cube_s0
```

The five-GPU sweep that produced the published baseline checkpoints is `bash scripts/run_phino_action_baselines.sh`. It writes `results/phino_action_baselines_v1/` and does not retrain the trajectory-similarity checkpoints.

### 6. Linear probes

```bash
.venv/bin/python scripts/phino_action_probe.py extract \
  --name ft_clip_s0_mesh_cube \
  --checkpoint results/phino_action_learning_v2/runs/clip_s0_mesh_cube/checkpoint.pt \
  --data-dir data/phino_simpact_v3obs_v2 \
  --features-dir results/phino_action_learning_v2/features \
  --phino-root external/phino

.venv/bin/python scripts/phino_action_probe.py extract --grid \
  --name grid_ft_clip_s0_mesh_cube \
  --checkpoint results/phino_action_learning_v2/runs/clip_s0_mesh_cube/checkpoint.pt \
  --data-dir data/phino_simpact_v3obs_v2 \
  --features-dir results/phino_action_learning_v2/features \
  --phino-root external/phino

.venv/bin/python scripts/phino_action_probe.py evaluate \
  --data-dir data/phino_simpact_v3obs_v2 \
  --output-dir /tmp/phino_probe_clip_s0 \
  --baseline image_position \
  --encoder trajectory_s0=results/phino_action_learning_v2/features/ft_clip_s0_{fold}.npz
```

`--grid` is the 4×4 pooled patch readout. Omit `--checkpoint` on `extract` for frozen DINOv2 (`--name dinov2_frozen`). `{fold}` is replaced with the held-out geometry name (`mesh_cube`, `mesh_cow`, `mesh_nut`, `blue_milk_carton`, `white_coconut_milk_carton`). A path without `{fold}` is one feature bank for every fold. The image-position baseline is the segmentation centroid and area stored with the clips. It does not use the encoder. The published probe commands are in `scripts/run_phino_action_baselines.sh`.

### 7. Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Without `GOOGLE_API_KEY`, the live Gemini tests skip. With an invalid key they fail in the client. That failure is the credential, not a change in the scene or export tests. GPU smokes (`scripts/smoke_test_mpm.py`, `scripts/smoke_test_arap.py`, `scripts/smoke_rigid_coexist.py`) are separate and were not run for this publication.

A smaller subset:

```bash
.venv/bin/python -m pytest tests/test_real2sim_pipeline.py \
  tests/test_scene_schema.py tests/test_export_phino_dataset.py -q
```

The renderer test needs offscreen GL:

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest \
  tests/test_export_phino_dataset.py::test_phino_renderer_frames_isolated_object -q
```

## Experiments and results

Numbers below are copied from the committed reports. They were not recomputed while assembling this repository. Mass and friction are simulator assignments on a 3×3 grid. Normalized MAE (nMAE) is the probe error divided by the error of predicting the training mean. The primary split is leave-one-geometry-out on 292 configurations that have all four actions. Seeds are reported as seed 0 / seed 1 when two runs exist. Full tables: [docs/results/phino_action_baselines_v1/summary.md](docs/results/phino_action_baselines_v1/summary.md) and [docs/results/phino_action_learning_v2/scaling_summary.md](docs/results/phino_action_learning_v2/scaling_summary.md).

### Identifiability from action-conditioned state

Stage 3 asked which interventions make mass and friction identifiable before any representation learning. Gate: failed. Report: [docs/results/phino_v3_identifiability_pilot_v1/](docs/results/phino_v3_identifiability_pilot_v1/).

| Quantity | What the pilot measured |
|---|---|
| Mass | Vertical toss apex is shape-independent and friction-blind (sensitivity 0.000). Mass error is 0.07–0.12 of the baseline on ideal state and 0.26–0.29 at 224 px. |
| Friction | Needs time-resolved deceleration after the push. Ideal state can identify it. At 224 px and 20 Hz the ratio to baseline is about 0.50 (cow) and 0.52 (cube). |
| Two pushes, final position only | Sensitivity directions are 2.1–8.4° apart. |
| Pre-registered gate | At most 2 of 6 variants pass. Centre-of-mass pushes topple the cartons and flip the nut. The bunny rocks and the sphere rolls. |

An earlier trajectory-only ridge audit, before this pilot, is in [docs/results/phino_settle_audit_v1/](docs/results/phino_settle_audit_v1/). Within-geometry recovery on clean geometries holds there. Cow and bunny similarities in the static v0 export are contaminated by friction-dependent settling. Do not train on v0 similarities.

### Representations on `v3obs_v2`

Pooled CLS‖mean-patch features. Leave-one-geometry-out nMAE at all four interventions (`push_x_v2`, `push_lo_1.6N`, `push_hi_3.2N`, `toss_4.0N`). Entries are mass / friction.

| Method | nMAE (mass / friction) |
|---|---|
| Frozen DINOv2 | 0.76 / 0.86 |
| Trajectory SoftInfoNCE | 0.61 / 0.60 and 0.60 / 0.61 |
| Standard InfoNCE | 0.65 / 0.59 and 0.63 / 0.69 |
| Supervised regression | 0.58 / 0.51 and 0.61 / 0.59 |
| Image-position tracker | 0.04 / 0.12 |

Sample-weighted across 292 configurations. Learned methods are seed 0 / seed 1. Frozen DINOv2 and the tracker have one fit.

The same endpoint with a 4×4 spatial readout:

| Method | nMAE (mass / friction) |
|---|---|
| Frozen DINOv2 | 0.67 / 0.59 |
| Trajectory SoftInfoNCE | 0.55 / 0.51 and 0.50 / 0.48 |
| Standard InfoNCE | 0.57 / 0.53 and 0.50 / 0.53 |
| Supervised regression | 0.51 / 0.45 and 0.54 / 0.49 |
| Image-position tracker | 0.04 / 0.12 |

Clip fine-tuning is lower error than frozen DINOv2 on the leave-one-geometry-out pooled endpoint for both seeds (paired geometry difference about −0.18 nMAE for mass and −0.22 for friction; both seeds lower on 5/5 geometries). A static-image control does not improve on frozen features (pooled leave-one-geometry-out mass nMAE 0.86). Within a geometry, held-out physics is a smaller gap: pooled mass nMAE is 0.35 frozen and 0.34 / 0.33 for trajectory SoftInfoNCE.

Trajectory similarity does not have a consistent advantage over standard InfoNCE. At four interventions the geometry-level paired difference is about 0.01 nMAE or smaller (pooled mass −0.001, pooled friction −0.005, 4×4 mass −0.007, 4×4 friction −0.013). The two seeds disagree in sign on several geometries. Supervised regression is lower on some geometries and higher on others. The per-geometry tables are in the baseline summary linked above.

The image-position tracker, using the segmentation centroid and area, remains near 0.04 / 0.12. Pooled DINOv2 features do not preserve that image-space motion across geometries (position-decoding R² at most about 0.20). Bunny and sphere are excluded from this dataset (2 and 0 configurations passed the shared-start gate). The five evaluated geometries are the two cartons, cow, cube, and nut.

## Data and checkpoints

Datasets, checkpoints, and third-party weights are not in Git.

Google Drive folder for the files listed below:

`GOOGLE_DRIVE_FOLDER_URL`

Replace that token with the folder URL after upload. Per-file SHA-256 sums for the upload set are in [docs/artifact_checksums.sha256](docs/artifact_checksums.sha256). Paths below are relative to the repository root. Place the Drive tree so `data/` and `results/` sit next to `scripts/`.

The validated training set is `data/phino_simpact_v3obs_v2`. `data/phino_simpact_v3obs_v1` leaked friction through sub-0.1 mm start offsets. It is not part of the upload set and must not be used for training.

| Path | Size | What it is |
|---|---|---|
| `data/phino_simpact_v3obs_v2/` | 98 MB | Action-conditioned clips. One shared before-image per geometry and camera. `provenance.json` SHA-256 `8a89b94d292dcc0d3a8c1314286d04910c9109aa8d4dea61f6156167a3886fdb`. The committed copy of that record is `docs/results/phino_action_learning_v2/dataset_provenance.json`. |
| `results/phino_mesh_grid_cutoff_1m_v1/` | 3.9 MB | Seven-geometry grid, 1.0 m displacement cap. 40/63 complete fingerprints. `manifest.json` SHA-256 `69d208de7d05cef77b40fdc6624837b9781a9c283841e3544f3d4dc7eb9d66d2`. |
| `results/phino_v3_identifiability_pilot_v1/` | 40 MB | Stage 3 traces the observation script replays. `traces.npz` SHA-256 `37175ed40e9ece0a82625cc77f214a99eafaebd6d71e3a3445d86c0cf1defebc`. Summary SHA-256 `de484463ddaf6c712357946bee6cc4b63a6882882a3042109f311a9263be7d06`. |
| `results/phino_action_learning_v2/runs/` | 5.2 GB | 15 trajectory-similarity checkpoints (`clip` seeds 0 and 1, plus the static control), with `config.json` and `metrics.csv`. |
| `results/phino_action_baselines_v1/runs/` | 7.0 GB | InfoNCE and regression checkpoints, seeds 0 and 1, five geometries. |
| `results/phino_action_learning_v2/probes_all/` | 37 MB | Pooled probe outputs for the trajectory runs. |
| `results/phino_action_learning_v2/probes_grid_ft/` | 11 MB | 4×4 probe outputs for the fine-tuned trajectory runs. |
| `results/phino_action_learning_v2/probes_grid_frozen/` | 5.0 MB | 4×4 frozen readout. |
| `results/phino_action_baselines_v1/probes_pooled/` | 24 MB | Pooled baseline probe outputs. |
| `results/phino_action_baselines_v1/probes_grid/` | 13 MB | 4×4 baseline probe outputs. |

Training logs for those runs are in the matching `logs/` directories of the same upload. Feature banks (`results/phino_action_learning_v2/features/`, about 2.9 GB, and `results/phino_action_baselines_v1/features/`, about 3.6 GB) are omitted. `phino_action_probe.py extract` rebuilds them from the checkpoints and `data/phino_simpact_v3obs_v2/`.

PHINO commit recorded with the dataset and the Stage 3 pilot: `3f1134d523559e52adce08bdac8e0b711c3c722c`.

## Status and limitations

Implemented and recorded:

- Rigid reconstruction and MuJoCo rollouts, including the committed push example.
- Export of the 1 m grid into PHINO's dataloader format, and the action-conditioned clip set `v3obs_v2`.
- Trajectory SoftInfoNCE, standard InfoNCE, and supervised regression on that clip set, with leave-one-geometry-out linear probes and an image-position tracker.
- The Stage 3 identifiability pilot. The pre-registered gate failed.
- Separate AnyGoal / SMWM diagnostics under `docs/results/anygoal_*`. Those runs did not train the PHINO objective. The Insert-Peg ranking that cast normalized actions to uint8 is superseded. With float32 actions, which is the training dtype, 25-step opportunity regret is 0.12 [0.04, 0.22] and pair accuracy is 0.96 [0.93, 0.98]. The uint8 rescore reproduces the stored regret 0.64 [0.50, 0.76] and is the record of that bug: [docs/results/anygoal_training_contract_v1/report.md](docs/results/anygoal_training_contract_v1/report.md). The goal-pose readout remains at 2.9 cm median hand error and 5.2 cm median object error.

Requires weights or a large GPU, and was not rerun here:

- SAM-3D reconstruction and `scripts/smoke_rigid_coexist.py`.
- Retraining the contrastive encoder. The tables cite the existing checkpoints.

Not done:

- A world model that predicts future observations from these representations.
- Planning with the learned representation.
- A new rigid-body protocol. The identifiability gate failed, and the elastic-reciprocity study was a NO-GO ([docs/phino_reciprocity_feasibility.md](docs/phino_reciprocity_feasibility.md)).
- Recovery of unknown real-world mass or friction from pixels. The labels are assigned in simulation.

Mass and friction written into generated MuJoCo XML by the rigid scene builder are fixed defaults (`mass="0.3"`, friction `0.3 0.005 0.0001` in `simpact/real2sim/scene.py`). They are not identified properties. The learning grids set those coefficients explicitly and record the values in the manifest.

## References and attribution

SIMPACT is Apache-2.0. The license is [LICENSE](LICENSE). This tree keeps that license and the upstream history.

```bibtex
@InProceedings{liu2026cvpr,
    author    = {Liu, Haowen and Yao, Shaoxiong and Chen, Haonan and Gao, Jiawei and Mao, Jiayuan and Huang, Jia-Bin and Du, Yilun},
    title     = {SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {20790-20801}
}
```

- Paper: <https://arxiv.org/abs/2512.05955>
- Upstream code: <https://github.com/ShaoxiongYao/simpact>
- Project page: <https://simpact-bot.github.io/>

PHINO (Physical Intuition via Newtonian Observations) is included only as the submodule `external/phino`, URL <https://github.com/arajshah/phino>, pinned to `3f1134d523559e52adce08bdac8e0b711c3c722c`. Upstream repository: <https://github.com/Hongyang-Du/phino>. The pinned checkout contains no `LICENSE` file. That absence is not a license grant. `external/phino/stable-worldmodel` declares MIT in its `pyproject.toml` and is used by the AnyGoal diagnostics, not by the contrastive training commands above.

DINOv2 weights are fetched from `facebookresearch/dinov2` and are not vendored. SAM, SAM 3D Objects, FoundationPose, and GroundingDINO weights stay outside Git. Follow each project's license when downloading them.

The experiment record is [docs/phino_experiment_log.md](docs/phino_experiment_log.md). The static-image export contract is [docs/phino_pretraining_integration_contract.md](docs/phino_pretraining_integration_contract.md). `docs/phino_initial_handoff.md` is a historical note from 2026-11-05.
