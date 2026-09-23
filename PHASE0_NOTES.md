# Phase 0 Notes — GeoFF3D / SLRF verification (SIH26158)

Source: `https://github.com/yanxian-ll/GeoFF3D` cloned at commit HEAD as of 2026-09-24
(shallow clone, `--depth 1`). All claims below cite exact files/lines in that clone.
This is **read-only source investigation** — no pip installs, no model downloads, no
Kaggle session were exercised. Everything under "must verify on Kaggle" is unverified.

## 1. Checkpoints

**No trained GeoFF3D (or Pi3X-finetuned, VGGT-finetuned, VGGT-Omega) checkpoint is
downloadable from this repo or referenced anywhere as a public artifact.**

- `bash_scripts/run_slrf/geoff3d.sh:15` defaults `checkpoint` to
  `$ROOT_DIR/checkpoints/geoff3d/checkpoint-best.pth` — a path that only exists after
  running `bash_scripts/train/geoff3d_stage1.sh` + `geoff3d_stage2.sh` yourself.
- `bash_scripts/run_slrf/pi3x.sh:15` defaults to
  `experiments/uav_training/pi3x_finetuning_8v_4d_8ipg_2g/checkpoint-best.pth` — same
  story, only produced by `bash_scripts/train/pi3x_finetuning.sh`.
- Same pattern for `vggt.sh` and `vggt_omega.sh`.
- `geoff3d/utils/hf_utils/hf_helpers.py:49` (`initialize_geoff3d_model`) has
  three-tier HF `from_pretrained`/`hf_hub_download` fallback logic keyed on
  `high_level_config["hf_model_name"]`, but **this function is never called anywhere
  else in the repo** (`grep -n "initialize_geoff3d_model"` finds only its own
  definition). It's dead/unused infra, not evidence of a published GeoFF3D checkpoint.
- README.md:51 says: *"Place pretrained weights under `checkpoints/`. The default
  Pi3X path is `checkpoints/pi3x`; override it with `PI3X_BASE_MODEL` if needed."*
  This refers only to the **Pi3X base model** (used as the frozen/finetune-start
  backbone for GeoFF3D training, see `configs/model/geoff3d.yaml:12` and
  `bash_scripts/train/geoff3d_stage1.sh:21`), not a finished GeoFF3D checkpoint.

**What IS downloadable:** the raw Pi3X and Pi3 base models from Hugging Face —
`geoff3d/models/external/pi3x/__init__.py:52,74-83` calls
`Pi3X.from_pretrained("yyfz233/Pi3X", ...)` directly via `huggingface_hub`/`safetensors`.
Similarly `geoff3d/models/external/pi3/__init__.py:22,36-43` for `yyfz233/Pi3`, and
`geoff3d/models/external/vggt/__init__.py:47-56` for VGGT-1B (`facebook/VGGT-1B`,
confirmed by URL in `benchmarking/third_party/vggt-slam2.0/vggt_slam/local_weights.py:42`).

**Conclusion:** Backbone **A (GeoFF3D+SLRF)** is not viable — there is no way to get a
trained GeoFF3D checkpoint without running the two-stage training pipeline ourselves,
which is infeasible inside a Kaggle session. Only the **raw pretrained backbones**
(Pi3X, Pi3, VGGT-1B) are obtainable, all via Hugging Face.

## 2. SLRF entry point & I/O format

Entry point: `scripts/run_slrf.py`, invoked through per-backbone launchers in
`bash_scripts/run_slrf/{geoff3d,pi3x,vggt,vggt_omega}.sh`. All four launchers wrap the
same `run_slrf.py` with different `model=`, `checkpoint=`, `align=`, and prior-prior
defaults (see the four scripts, each ~25 lines).

**Input folder format** (`scripts/run_slrf.py:187-238`, `geoff3d/slrf/scene_io.py:149-191`):

```
<scene_dir>/
  images/   <stem>.{jpg,jpeg,png,bmp,tif,tiff}     (required; run_slrf.py:192-202)
  cams/     <stem>.txt                              (required; run_slrf.py:203-206)
  depth/    <stem>.exr                               (optional; only required when
                                                       footprint_estimation=prior or
                                                       depth_prior=input)
```

`cams/<stem>.txt` is a COLMAP/MVSNet-style camera file (`parse_cam_txt`,
`geoff3d/slrf/scene_io.py:149-191`):
```
extrinsic
<4x4 world-to-camera matrix T_w2c, one row per line>
intrinsic
<3x3 K matrix, one row per line>
[optional "h w fov"/"h w hfov" header line + values]
```
Note the *translation/rotation priors* the task spec mentions aren't a separate file —
they ARE this same `cams/*.txt` (the extrinsic block *is* the pose prior fed to the
model; `translation_prior=input`/`rotation_prior=input` in the launcher scripts tell
the model to actually use the translation/rotation components of this matrix as priors
rather than ignoring them).

**Output**: `<output_path>/result.rrd` (Rerun recording, primary output — see
`save_spatial_rrd`, `geoff3d/slrf/rrd_writer.py`), `<output_path>/processing_time.json`,
`<output_path>/eval/metrics.json` + `metrics_summary.csv`, chunk-partition visualization
PNGs, and optionally `<output_path>/mesh/` (TSDF, only if `export_tsdf_mesh=true`) and
`<output_path>/bundle_adjustment/` (only if `bundle_adjustment=true`). **There is no
direct LAS/OBJ/GeoTIFF export in SLRF** — those are Rerun `.rrd` + a TSDF mesh at best;
we will need to write our own exporters reading points/cameras back out of the chunk
records or the `.rrd`, which the task's `export.py` module already plans for.

## 3. `scale_yaw_translation` / `FOOTPRINT_ESTIMATION=sequential` / `max_chunk_size`

- **`align`** (`scripts/run_slrf.py:156-169`, `geoff3d/slrf/geometry_align.py` /
  `ALIGN_MODES`): GeoFF3D supports `{none, scale, translation, scale_translation,
  scale_yaw_translation, yaw_translation, sim3}` (listed in `configs/slrf.yaml`
  comments, lines ~46-48). **Every other model (Pi3X, Pi3, VGGT, VGGT-Omega) only
  supports `sim3`** — enforced by `run_slrf.py:164-168` (`raise ValueError` otherwise).
  This is the collinear-flight fix the task spec references: `scale_yaw_translation`
  solves scale + yaw (Z rotation) + XYZ translation only, never fitting roll/pitch to
  noisy GPS — but **this alignment mode is only available when `model=geoff3d`**, i.e.
  it is architecturally tied to GeoFF3D's translation/rotation-prior training, not a
  generic post-processing step usable with Pi3X/VGGT. Confirmed at `bash_scripts/run_slrf/pi3x.sh:19`
  (`align=sim3`, hardcoded, not overridable via the same env-var pattern as `geoff3d.sh`).
  → **This is a real architectural risk for Backbone B/C**: our own 4-DoF
  gravity-fixed alignment (task requirement §4) will have to be implemented in our own
  `align.py`, independent of GeoFF3D's `scale_yaw_translation`, if we're not running the
  actual GeoFF3D model.
- **`FOOTPRINT_ESTIMATION=sequential`** (`scripts/run_slrf.py:369-410`,
  `geoff3d/slrf/footprint_estimation.py`): runs a **full extra forward pass** of the
  same backbone over the whole scene, chunked into non-overlapping sequential chunks
  (tail <8 images merged into previous chunk — this matches README.md:107-108 exactly),
  to estimate per-image spatial footprints without needing GT depth. Its output feeds
  `build_spatial_chunks(..., footprint_source="sequential")`. This roughly **doubles**
  backbone inference cost (one pass for footprints, one for the real spatial chunks) —
  important for the 15-minute budget; `footprint_estimation="prior"` (needs `depth/*.exr`
  per frame) skips this but requires metric depth we won't have without RTK/LiDAR.
- **`max_chunk_size`** (`configs/slrf.yaml`, default `32`; launcher scripts default
  `30`): passed straight into `build_spatial_chunks` (`geoff3d/slrf/chunking.py`) as the
  max images per spatial chunk; `min_chunk_size` (default 8) is the merge-back
  threshold. Overridable per the README example
  (`CHECKPOINT=... bash bash_scripts/run_slrf/geoff3d.sh scene out max_chunk_size=24`).
  This is the OOM lever the task's robustness rules ask for (halve on OOM and retry).

## 4. Install requirements & Kaggle risk

`pyproject.toml` hard-pins:
```
torch==2.5.0
torchvision==0.20.0
xformers==0.0.28.post2
```
plus `gsplat`, `open3d>=0.18`, `pycolmap>=3.10,<3.12`, `OpenEXR`, `rerun-sdk~=0.24.1`,
`uniception==0.1.6`, `hydra-core`, `omegaconf>=2.3,<2.4`. README.md:31-37 additionally
requires `torch-scatter==2.1.2` and `torch-cluster` from `data.pyg.org` wheels matched
to `torch-2.5.0+cu121`.

**Risk**: `pip install -e .` on this repo will try to force-reinstall torch/torchvision/
xformers at those exact pins, directly violating the task's "don't reinstall torch,
try Kaggle's preinstalled torch first" rule, and OpenEXR/gsplat/pycolmap/open3d are all
C-extension packages with real odds of wheel mismatches on Kaggle's image. `gsplat`
compiles CUDA kernels at install time (needs nvcc + matching torch ABI) — highest risk
of a hard install failure, and we don't need Gaussian-splat refinement
(`gsplat_refine=false` is already the SLRF default) so it should be skip-installable.

**Recommended approach (must verify in an actual Kaggle session)**:
1. Do NOT `pip install -e .` the whole repo. Vendor only the needed subpackages
   (`geoff3d/slrf/`, `geoff3d/models/`, `geoff3d/utils/`, `geoff3d/datasets/base` if
   imported transitively) into our own `sih3d/vendor/geoff3d/` and install their
   non-torch dependencies individually with `--no-deps`, skipping `gsplat` entirely
   (nothing in the SLRF main path requires it unless `gsplat_refine=true`, which is off
   by default — confirmed `configs/slrf.yaml` bottom section).
2. Check Kaggle's preinstalled torch version/CUDA first; only install
   `torch-scatter`/`torch-cluster`/`xformers` wheels matched to whatever's actually
   there (2× T4 = sm_75, no bf16) — **this needs an actual Kaggle session to confirm
   wheel availability**, not verifiable from this clone.
3. `OpenEXR` python bindings need the system `openexr`/`ilmbase` libs — verify via
   `apt list --installed` on Kaggle or fall back to `imageio`'s EXR plugin if apt
   install isn't permitted without sudo.

## 5. Package layout (relevant to backbone interface)

```
geoff3d/
  models/external/{pi3, pi3x, vggt, vggt_omega, geoff3d, dinov2}/   # per-backbone wrappers + weights loaders
  models/__init__.py                                                 # init_model(model_str, model_config, ...) factory
  slrf/
    scene_io.py            # images/cams/depth loading, parse_cam_txt
    chunking.py             # build_spatial_chunks / order_spatial_chunks
    footprint_estimation.py # prior vs sequential footprint modes
    model_runner.py         # init_model_from_hydra, load_checkpoint, prior-policy resolution
    geometry_align.py       # ALIGN_MODES, apply_chunk_pose_alignment (per-chunk)
    chunk_post_align.py     # deferred cross-chunk alignment
    pipeline_postprocess.py # apply_final_global_pose_alignment
    tsdf_mesh.py            # optional TSDF export
    rrd_writer.py           # Rerun .rrd output
  utils/{geometry,colmap,image,inference}.py
scripts/run_slrf.py          # the one entry point (see §2)
configs/{model,slrf.yaml,...} # Hydra configs
```
`init_model(model_str, model_config, ...)` in `geoff3d/models/__init__.py` is the clean
factory boundary — our own `sih3d/backbone.py` config-switch (A/B/C) should mirror this
signature rather than reimplement chunk orchestration, since `slrf/chunking.py` +
`slrf/model_runner.py` are reusable independent of which backbone we ultimately run.

## 6. Backbone A/B/C feasibility (with evidence)

- **A — GeoFF3D+SLRF**: **not viable.** No downloadable trained checkpoint (§1). Would
  require running `geoff3d_stage1.sh` + `geoff3d_stage2.sh` training ourselves on
  UAV datasets we don't have, for many GPU-hours — completely out of budget.
- **B — Pi3X through GeoFF3D's SLRF framework**: **viable, but with a concrete gotcha
  that must be verified in a real environment.** `run_slrf.py:498` unconditionally sets
  `model.model_config.load_pretrained_weights=false` before calling
  `init_model_from_hydra` (same at the "sequential footprint" stage, line ~387) — this
  means **`Pi3XWrapper` never calls `Pi3X.from_pretrained()` when driven through
  `run_slrf.py`**, regardless of `pretrained_model_name_or_path`
  (`geoff3d/models/external/pi3x/__init__.py:71-88`: `from_pretrained` is only called
  `if self.load_pretrained_weights`). Weights only ever enter via
  `load_checkpoint()` (`geoff3d/slrf/model_runner.py:410-424`), a raw
  `torch.load(checkpoint); model.load_state_dict(state, strict=False)` against the
  **wrapper's** state dict (keys prefixed `model.*`, since `Pi3XWrapper.model = Pi3X(...)`).
  A plain `Pi3X.from_pretrained("yyfz233/Pi3X").state_dict()` saved as-is would silently
  fail to load (key-prefix mismatch, masked by `strict=False`) and run on **random
  weights** with no error. **Workaround (untested, logic-verified only)**: download
  `Pi3X.from_pretrained("yyfz233/Pi3X")` once, save
  `{"model": {"model."+k: v for k,v in model.state_dict().items()}}` to a local
  `.pth`, and pass that as `CHECKPOINT=` to `bash_scripts/run_slrf/pi3x.sh`. **Must be
  tested on an actual machine with torch before relying on it** — this is inferred from
  reading `load_state_dict`/`Pi3XWrapper` source only, not run.
  Also: Pi3X only supports `align=sim3` (§3), so our own gravity-fixed 4-DoF alignment
  (task §4) must be implemented in `sih3d/align.py`, not delegated to SLRF's `align=`.
- **C — MapAnything Apache variant, own chunking**: **viable and the fallback with
  fewest unknowns.** `facebook/map-anything-apache` wasn't found referenced inside
  `GeoFF3D/geoff3d/` proper (only in `benchmarking/third_party/vggt-long/` third-party
  vendor code, e.g. `configs/map_long_config.yaml:6`, unrelated to our path), so this is
  a standalone integration — no dependency on GeoFF3D's SLRF machinery, checkpoint
  problems, or the forced-random-weights footgun above, at the cost of writing our own
  chunk-overlap propagation instead of reusing `geoff3d/slrf/chunking.py`.

**Recommendation**: implement backbones in priority **C first (least risk, ships a
working QUICK-mode pipeline fastest), then B (higher potential quality if the
checkpoint-prefix workaround holds up, using GeoFF3D's SLRF chunking/footprint/align
machinery which is genuinely more sophisticated than anything we'd write from
scratch), A left as a stub that raises a clear "no published checkpoint" error** so the
config switch exists per the task's requirement #5 but doesn't pretend to work.
Confirm this ordering after Phase B's checkpoint workaround is actually tested.

## 7. Open questions / must-verify-on-Kaggle

1. **Torch/CUDA compatibility**: Kaggle's preinstalled torch version/CUDA build vs.
   `torch-scatter`/`torch-cluster`/`xformers` wheel availability for that exact combo —
   cannot be checked without a live Kaggle session.
2. **Pi3X checkpoint-prefix workaround (§6, Backbone B)**: whether
   `{"model."+k: v}`-prefixed state dict actually loads cleanly into `Pi3XWrapper`
   (`load_state_dict` should report zero missing/unexpected keys) — needs an actual
   torch run to confirm the exact submodule name is `model` and not something else
   picked up by a decorator/mixin.
3. **`geoff3d/slrf/` importability in isolation**: whether `geoff3d/slrf/*.py` can be
   imported without pulling in the full `geoff3d.datasets`/`geoff3d.train` tree (which
   has heavier deps like `tensorboard`, `lpips`) — only skimmed file layout, didn't
   trace every import.
4. **gsplat/OpenEXR/pycolmap on Kaggle**: whether these install cleanly at all if we
   ever need `export_tsdf_mesh`/`gsplat_refine`/`render_dom` (all off by default, so
   only matters if we later want SLRF's built-in mesh/DOM path instead of our own
   `mesh.py`/`fusion.py`).
5. **No LAS/OBJ/GeoTIFF export exists in SLRF** (§2) — confirmed by absence, not a
   missing search; our `export.py` must build these from chunk point/camera records or
   by reading `result.rrd`, which needs its own investigation into the `.rrd` schema
   (`geoff3d/slrf/rrd_writer.py`) if we want to reuse SLRF's aggregated output rather
   than re-deriving points from `chunk_records` ourselves in the pipeline loop.

---

## 8. MapAnything Apache (Backbone C) — direct API

Source: `https://github.com/facebookresearch/map-anything` cloned at commit HEAD as of
2026-09-24 (shallow clone). README.md is unusually precise (a public-facing API doc, not
just marketing) so most of this section cites it directly plus source confirmation of
the exact signatures.

**Model loading**: `mapanything/models/mapanything/model.py:92`:
`class MapAnything(nn.Module, PyTorchModelHubMixin)`. `.from_pretrained(...)` therefore
comes from the standard `huggingface_hub.PyTorchModelHubMixin` mixin (not a custom
loader, unlike GeoFF3D's `Pi3XWrapper` — no checkpoint-prefix footgun here). Confirmed
both variants are real, distinct HF repos (README.md:808-825): `facebook/map-anything`
(CC-BY-NC 4.0) vs. **`facebook/map-anything-apache`** (Apache 2.0) — same API, different
training-data licensing, exactly as the task spec assumes. Call:
```python
from mapanything.models import MapAnything
model = MapAnything.from_pretrained("facebook/map-anything-apache").to(device)
```
There's also a V1-release pair (`facebook/map-anything-apache-v1`, README.md:817-818) —
not needed, the non-versioned repo id is the current release.

**Input format** — `MapAnything.infer()`, `mapanything/models/mapanything/model.py:2028-2117`
(full docstring read directly from source, not just the README):
```python
def infer(
    self, views: List[Dict[str, Any]],
    memory_efficient_inference: bool = True, minibatch_size: int = None,
    use_amp: bool = True, amp_dtype: str = "bf16",
    apply_mask: bool = True, mask_edges: bool = True,
    edge_normal_threshold: float = 5.0, edge_depth_threshold: float = 0.03,
    apply_confidence_mask: bool = False, confidence_percentile: float = 10,
    ignore_calibration_inputs=False, ignore_depth_inputs=False, ignore_pose_inputs=False,
    ignore_depth_scale_inputs=False, ignore_pose_scale_inputs=False,
    use_multiview_confidence=False,
    multiview_conf_depth_abs_thresh=0.02, multiview_conf_depth_rel_thresh=0.02,
) -> List[Dict[str, torch.Tensor]]
```
Each view dict: required `img` (B,3,H,W) normalized + `data_norm_type` (must be `"dinov2"`
for MapAnything per README.md:534). Optional, any combination: `intrinsics` (B,3,3)
**or** `ray_directions` (B,H,W,3) (mutually exclusive — enforced, model.py docstring
line 2095), `depth_z` (B,H,W,1) (requires intrinsics/ray_directions), `camera_poses`
(B,4,4) OpenCV cam2world, or `(quats (B,4), trans (B,3))` tuple, `is_metric_scale`
(bool or (B,)-tensor, defaults True if omitted). **Constraint**: if any view has
`camera_poses`, view 0 must also have them (model.py:2097). This directly maps to our
task's "feed the backbone every prior available: intrinsics, GPS translations, attitude
rotations" requirement — GPS-derived translation + attitude become `camera_poses`,
camera intrinsics (from metadata or estimated) become `intrinsics`, and
`is_metric_scale=True` should be set since GPS gives us real metric translations, which
is exactly the metric-anchoring mechanism this model wants.

**Output format** (model.py:2099-2116, matches README.md:169-195 exactly): per-view
dict with `pts3d` (world frame, B,H,W,3), `pts3d_cam` (camera frame), `depth_z`,
`depth_along_ray`, `ray_directions`, `intrinsics` (recovered), `cam_trans`/`cam_quats`/
`camera_poses` (cam2world, world frame), `metric_scaling_factor` (B,) — the actual
metric scale factor the model applied (useful to sanity-check against our GPS-derived
expected scale — a large deviation would flag bad GPS sync or a degenerate chunk),
`conf` (B,H,W), `mask`/`non_ambiguous_mask`/`non_ambiguous_mask_logits`,
`img_no_norm`. World frame is **not** automatically GPS-referenced — even with
`camera_poses` priors, the model still needs our own alignment step to guarantee the
world frame matches ENU/UTM exactly (the `camera_poses` prior likely anchors it well in
practice, but this should be verified empirically, not assumed).

**Chunking / view limits**: no hard limit found; the README's profiling claims *"up to
2000 views on 140 GB"* with `memory_efficient_inference=True, minibatch_size=1`
(README.md:158, 639). T4 = 16 GB, ~9x less memory, so this scales down to roughly
low-hundreds of views per chunk at best *(rough linear extrapolation, not confirmed by
reading the actual memory-efficient-inference implementation — must be verified
empirically on real T4 hardware, since attention/activation memory doesn't always scale
linearly with view count)*. No existing large-scene spatial-chunking utility was found
in `mapanything/` proper (nothing equivalent to GeoFF3D's `slrf/chunking.py`) — we will
need our own chunking exactly as the task assumes for Backbone C ("using our own
chunking"). `scripts/demo_inference_on_colmap_outputs.py` (read header,
lines 1-40) is the closest existing reference: loads COLMAP calibration+poses as
external priors and feeds them into MapAnything the same way we'd feed GPS priors —
good pattern to mirror for `sih3d/backbone.py`'s Option-C prior-injection code, though
it processes one scene as a single batch (no chunking loop to copy).

**Install risk — significantly lower than GeoFF3D (Backbone A/B)**: `pyproject.toml`
(read in full) has **no version pin on torch/torchvision/CUDA at all** — the repo
explicitly says *"we don't pin a specific version of PyTorch or CUDA... install PyTorch
based on your specific system"* (README.md:128). Base `dependencies` list (pyproject.toml
lines 12-27) has **no torch-scatter, no torch-cluster, no xformers, no gsplat** — just
`huggingface_hub`, `hydra-core`, `opencv-python-headless`, `rerun-sdk`, `safetensors`,
`trimesh`, `uniception==0.1.7`, etc. The `mapanything` base model entry in the model
table is explicitly marked **"Install Extra: (base)"** (README.md:403) — i.e. no extra
install group needed, confirmed by the extras table (pyproject.toml lines 29-52) only
gating VGGT-Omega/DUSt3R/MASt3R/MUSt3R/Pow3R/AnyCalib/DA3/Pi3-X-via-external-repo, none
of which we need since we call `MapAnything` directly, not through
`model_factory("pi3x", ...)`. **This makes Backbone C by far the safest install on
Kaggle** — `pip install -e .` without extras should not touch the preinstalled torch at
all. `uniception==0.1.7` (a Meta-adjacent helper lib, README.md:846) is the one
dependency whose own transitive requirements weren't traced — **must verify it doesn't
itself pin torch** on an actual Kaggle session before trusting this conclusion fully.

**Reference scripts to pattern our code on**:
- `scripts/demo_images_only_inference.py` — simplest end-to-end image-only path.
- `scripts/demo_inference_on_colmap_outputs.py` — external calibration+pose priors fed
  into `model.infer()`, with `--ignore_calibration_inputs`/`--ignore_pose_inputs` flags
  showing how to toggle which priors are actually used, and `--save_glb`/`--save_colmap`
  export options we can reference for our own `export.py`.
- `scripts/profile_memory_runtime.py` — has the actual memory-efficient-inference
  profiling harness; useful if we need to empirically determine our own T4 chunk-size
  budget rather than guessing from the 2000-views/140GB number.

**Top risks/unknowns for Backbone C** (needs a live GPU/Kaggle session):
1. Actual per-chunk view-count budget on a 16 GB T4 with `memory_efficient_inference=True`
   — the 2000-views/140GB figure doesn't linearly imply a safe T4 number.
2. Whether `camera_poses` + `is_metric_scale=True` priors alone are sufficient to anchor
   the output `pts3d` world frame to true ENU/UTM without extra alignment, or whether
   our own `align.py` post-alignment step (task §4) is still required regardless (the
   task spec assumes the latter — "Never use plain Sim(3) for chunk-to-world
   alignment" — so this is probably moot, but worth confirming the model's own gauge
   doesn't already do something incompatible with that assumption).
3. `uniception==0.1.7`'s own dependency footprint (possible transitive torch pin) —
   untraced.
