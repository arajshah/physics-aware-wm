# Teammate handoff

How to use this fork from a fresh clone. Research conclusions stay in
[`phino_cursor_handoff.md`](phino_cursor_handoff.md) and
[`phino_experiment_log.md`](phino_experiment_log.md). Large files that are not
in Git are listed in [`BROWN_ARTIFACT_MANIFEST.md`](BROWN_ARTIFACT_MANIFEST.md).

Checked on 2026-11-16 against branch `phino-physics-experiments` at
`32cc8a42796e7caf0aeb98f7ae173603d45fba27`, on the Brown tree
`/nbu/people/ashah264/simpact` (the same directory as
`/cs/local/nbu/people/ashah264/simpact`). The machine used for this audit has
an NVIDIA GeForce RTX 5060 Ti with **8 GB** of VRAM. That is enough to load
MuJoCo and to import the perception adapters. It is not enough to rerun
SAM-3D. The full rigid reconstruction was not rerun.

Repository: <https://github.com/arajshah/simpact>  
Branch: `phino-physics-experiments`  
Upstream paper repo (unchanged by this handoff): <https://github.com/ShaoxiongYao/simpact>

## 1. Setup

Python is pinned to **3.11.15** (`.python-version`). The GPU stack in
`pyproject.toml` is `torch==2.9.0+cu128` and `warp-lang==1.10.0`. Use that
CUDA 12.8 build. A cu118 torch wheel has no Blackwell kernels.

```bash
git clone https://github.com/arajshah/simpact.git
cd simpact
git checkout phino-physics-experiments
uv sync --extra dev
cp .env.example .env
```

Commands below use `.venv/bin/python`, which is the interpreter `uv sync`
creates. `uv run python ...` is the same thing.

Headless MuJoCo rendering on this host needs:

```bash
export MUJOCO_GL=egl
```

`GOOGLE_API_KEY` is required only for VLM propose / verify / regress. The
replay in section 6 does not call a model API. The `.env` on the audit machine
does not contain a usable Gemini key, so closed-loop planning was not run.

After `bash scripts/setup_rigid_env.sh`, resync with
`uv sync --extra rigid --extra dev` and rerun the script. A bare `uv sync`
removes the source-built rigid packages. Details:
[`RIGID_ENV_SETUP.md`](RIGID_ENV_SETUP.md).

Import `open3d` before `coacd`. The convex-decomposition module already does
this. Wrap `frankx`, `pyrealsense2`, and `warp` imports so a machine without
those packages still imports the rest of the library.

## 2. External models and checkpoints

These are not in Git. `external/` is gitignored.

| Piece | Role | Where it lives on this Brown checkout | Where to obtain it |
|---|---|---|---|
| SAM 2.1 hiera large | Grounded-SAM-2 masks | `external/checkpoints/sam2.1_hiera_large.pt` (857 MB). `.env` points `SIMPACT_SAM2_CHECKPOINT` at that file. | <https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt> |
| GroundingDINO tiny | Open-vocab boxes | Hugging Face cache under `.cache/huggingface/` (gitignored) | `IDEA-Research/grounding-dino-tiny`, loaded by `transformers` on first use |
| SAM 3D Objects | Image to a complete mesh | `external/sam-3d-objects/` (12.7 GB, including `checkpoints/hf/slat_decoder_mesh.ckpt`) | Clone <https://github.com/facebookresearch/sam-3d-objects>. Request access and download checkpoints from <https://huggingface.co/facebook/sam-3d-objects> (`hf download`, see that repo's `doc/setup.md`). Weights are under the SAM license. |
| FoundationPose | 6-DoF pose | `external/FoundationPose/` (477 MB). Weights present: `weights/2023-10-28-18-33-37/` (refiner) and `weights/2024-01-11-20-02-45/` (scorer). | Clone <https://github.com/NVlabs/FoundationPose>. Weights: <https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing> |
| kaolin | SAM-3D mesh extraction | Built into `.venv` by `scripts/setup_rigid_env.sh` from `$KAOLIN_SRC` | <https://github.com/NVIDIAGameWorks/kaolin> |
| Camera calibration | World-frame poses for the committed example | `assets/calibration/0103/` (tracked) | Already in the clone |
| Franka gripper | Push rollouts | `assets/robot/` (tracked) | Already in the clone |

`SIMPACT_GROUNDED_SAM2_DIR` and `SIMPACT_HUNYUAN3D_DIR` are still placeholders
in `.env`. Grounded-SAM-2 does not need a repo clone. Hunyuan3D is an unused
alternate reconstructor.

A fresh FoundationPose clone needs the two-line torch 2.9 patch before the
`mycuda` build (`-std=c++14` to `-std=c++17`, and `.type()` to `.scalar_type()`).
The patch notes are in [`VALIDATION_rigid.md`](VALIDATION_rigid.md) and
[`RIGID_ENV_SETUP.md`](RIGID_ENV_SETUP.md).

## 3. Minimal input

The one example that is already in Git is
`examples/push_real2sim/0103_push_0/`. No extra download is required to load
it or to replay its recorded plan.

```text
examples/push_real2sim/0103_push_0/
  capture/camera1_rgb.png          # 640x480 RGB
  capture/camera1_depth.npy        # float metres, 640x480
  capture/initial_ee_pose.txt      # 4x4, robot frame
  sim/scene.yaml                   # camera profile "0103", cam 1, initial EE pose
  sim/blue milk carton_scaled.obj
  sim/blue milk carton_scaled_0.png
  sim/blue milk carton_6d_cam1.txt
  sim/white coconut milk carton_scaled.obj
  sim/white coconut milk carton_scaled_0.png
  sim/white coconut milk carton_6d_cam1.txt
  runs/refined_plan.json
  runs/final_rollout/rollout_00.json
```

Intrinsics and the camera-to-robot extrinsic are not copied into the trial.
`sim/scene.yaml` points at `assets/calibration/0103/` (`cam1_K.txt`,
`cam1_to_robot.txt`).

Object names for this trial, matching `simpact/tasks.py`:

```text
white coconut milk carton. blue milk carton.
```

## 4. Reconstruct a rigid scene

Driver: `scripts/run_rigid_pipeline.py`. Stages, in order:

1. Read RGB (`.npy` or `.png`) and depth in metres.
2. `GroundedSAM2Segmenter` — text prompt to masks.
3. Write a full mask and an RGBA crop.
4. `SAM3DReconstructor` — crop to a scale-free mesh.
5. Metric scale from the masked depth cloud (statistical outliers removed).
6. `FoundationPoseEstimator` — 6-DoF object-to-camera pose.
7. Fit a table plane from depth, settle the lowest vertex to z = 0, and write
   `scene.xml`. When calibration resolves, also write
   `{name}_6d_cam1.txt` and `{name}_mujoco_cam1.txt`.
8. `build_mujoco_scene`.

```bash
# NOT RE-RUN on 2026-11-16. Needs a GPU with about 24 GB.
# The 2026-11-04 log on an L40 peaked at 18.74 GB (results/rigid_0103.log).
# This audit host has 8 GB.
.venv/bin/python scripts/run_rigid_pipeline.py \
  --data_dir examples/push_real2sim/0103_push_0 \
  --objects "white coconut milk carton. blue milk carton." \
  --cam 1 \
  --K assets/calibration/0103/cam1_K.txt \
  --out_dir /tmp/rigid_demo \
  --xml /tmp/rigid_demo/scene.xml
```

Success of that command is the line `RIGID_PIPELINE_OK` plus a `scene.xml`
that MuJoCo can load. That line is in `results/rigid_0103.log` from the L40
run. It means the chain executed. It is not a new accuracy measurement.

The push planner does not consume that `scene.xml` directly. It rebuilds the
robot-frame scene from `{name}_scaled.obj`, `{name}_6d_cam1.txt`, and the
0103 extrinsic (`simpact/executor/push_scene.py`). The committed `sim/` files
are the assets that replay uses. The Nov 4 meshes in `results/rigid_0103/`
are a separate decimation (about 20k vertices, 40k faces). The committed
meshes are about 5.5k–6.4k vertices and 10k faces.

Documented accuracy, not remeasured here, is in
[`VALIDATION_rigid.md`](VALIDATION_rigid.md) (2026-06-22, RTX 5090). That note
includes a point-cloud fit on `0103_push_0` of 15.0 mm (blue carton) and
20.2 mm (white carton). `scripts/validate_rigid_pipeline.py` is described in
that document and is not in the repository.

Perception smokes, also not run on this 8 GB GPU:

```bash
.venv/bin/python scripts/smoke_gsam2.py
.venv/bin/python scripts/smoke_sam3d.py
.venv/bin/python scripts/smoke_rigid_coexist.py   # expects RIGID_COEXIST_OK
```

## 5. Load the committed scene in MuJoCo

Checked on 2026-11-16. Building the scene from the committed meshes and the
0103 calibration produced 4 bodies and 2 meshes. Fifty `mj_step` calls left
`qpos` finite. The same load-and-step check succeeded for
`results/rigid_0103/scene.xml` (Brown only).

Replay the recorded plan and compare final object positions to the committed
rollout. This was run on 2026-11-16. Both objects matched to 0.000 cm.

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

The replay writes `/tmp/push_replay/rollout_00.json` and `rollout_00.mp4`.
It does not modify `examples/`.

## 6. Tests

Relevant subset, 2026-11-16, without `MUJOCO_GL`:

```bash
.venv/bin/python -m pytest tests/test_real2sim_pipeline.py \
  tests/test_scene_schema.py tests/test_export_phino_dataset.py -q
```

Result: **66 passed, 2 skipped**. Skips: `test_golden_trial` (set
`SIMPACT_R2S_GOLDEN_DIR` to opt in) and
`test_phino_renderer_frames_isolated_object` (offscreen GL was unset).

The renderer test with EGL, same day: **1 passed**.

```bash
MUJOCO_GL=egl .venv/bin/python -m pytest \
  tests/test_export_phino_dataset.py::test_phino_renderer_frames_isolated_object -q
```

Full suite, run on 2026-11-16 in this environment:

```bash
.venv/bin/python -m pytest tests/ -q
```

Result: **336 passed, 2 skipped, 3 failed**. The three failures are live
Gemini calls rejected with `API_KEY_INVALID`:

- `tests/test_dough_auto_build.py::test_dough_auto_build_matches_reference`
- `tests/test_propose.py::test_real_gemini_proposal`
- `tests/test_regress.py::test_real_gemini_regress`

They need a valid `GOOGLE_API_KEY`. The rest of the suite passed, including
the scene-schema, real2sim geometry, and PHINO export tests. GPU smokes
(`scripts/smoke_test_mpm.py`, `scripts/smoke_test_arap.py`,
`scripts/smoke_rigid_coexist.py`) were not run.

`bash scripts/reproduce_all.sh --quick` rebuilds planning outputs and, when
the perception stack and a Gemini key are present, rebuilds scenes. It was
not run. It writes to a timestamped directory under `/tmp` unless
`--overwrite-examples` is passed. Do not pass `--overwrite-examples`.

## 7. What “success” means

| Check | What it shows | Status on 2026-11-16 |
|---|---|---|
| `RIGID_PIPELINE_OK` in `results/rigid_0103.log` | The eight stages finished once, on an L40, for this example | Historical log. Not rerun. |
| MuJoCo load of `results/rigid_0103/scene.xml` | That XML is simulatable | Rechecked. 4 bodies, 2 meshes, finite state. |
| Replay `--check` on `examples/push_real2sim/0103_push_0` | The committed reconstruction plus the recorded plan reproduce the committed final positions | Rechecked. 0.000 cm on both objects. |
| `docs/VALIDATION_rigid.md` millimetre / degree figures | Pose and shape agreement from 2026-06-22 | Not remeasured. |
| PHINO dataloader and learning numbers in `docs/results/` | Interface and learning pilots already written up | Summaries are in Git. Weights and grids are on Brown. Those numbers were not recomputed. |

## 8. Known gaps

- This audit GPU is an 8 GB RTX 5060 Ti. The documented reconstruction peak is
  18.7 GB. Use a GPU with at least ~24 GB before rerunning
  `run_rigid_pipeline.py` or `smoke_rigid_coexist.py`.
- `scripts/validate_rigid_pipeline.py` is documented and missing.
- Scale in the rigid driver is a depth-bbox ratio after outlier removal, not
  a measured CAD dimension. [`VALIDATION_rigid.md`](VALIDATION_rigid.md)
  records the earlier 64 cm failure and the correction to about 26 cm.
- `run_rigid_pipeline.py` settles objects onto a depth-fitted plane.
  `PushSceneRollout` places them with the 0103 camera-to-robot extrinsic and
  snaps them to the rig table. Those are two frames. Replay uses the second.
- Mass and friction in the generated XML are fixed defaults (`mass="0.3"`,
  friction `0.3 0.005 0.0001` in `simpact/real2sim/scene.py`). They are not
  identified physical properties.
- SAM-3D output used by the XML writer is vertex color, averaged to one RGBA
  per object. The committed example additionally has UV textures
  (`*_scaled_0.png`), and the replay uses those textures.
- Closed-loop VLM planning needs each person’s own `GOOGLE_API_KEY`.
- Static images with assigned mass and friction do not identify those
  values. Read the warning at the top of
  [`phino_cursor_handoff.md`](phino_cursor_handoff.md) before citing a
  training result.
- `data/phino_simpact_v3obs_v1/` leaked friction through sub-0.1 mm start
  offsets. Use `data/phino_simpact_v3obs_v2/` for the action-conditioned
  learning pilot. Do not train on v0 similarity matrices
  ([`phino_cursor_handoff.md`](phino_cursor_handoff.md)).

## 9. Larger artifacts

Nothing under `results/`, `data/`, or `external/` is in Git. `/results/` is
gitignored. Summaries and figures that are safe to clone are under
`docs/results/` (59 tracked files, about 3.7 MB).

The Brown inventory and the proposed copy method are in
[`BROWN_ARTIFACT_MANIFEST.md`](BROWN_ARTIFACT_MANIFEST.md). Do not upload
those trees to an external host until the research owner approves the
destination. Third-party checkpoints have their own licenses.

## 10. PHINO code

Training code is a separate checkout, `arajshah/phino`, on this machine at
`/nbu/people/ashah264/phino`. It is not a submodule. Export commands that need
it take `--phino-root`. See
[`phino_pretraining_integration_contract.md`](phino_pretraining_integration_contract.md).
