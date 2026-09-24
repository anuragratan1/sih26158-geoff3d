# Running `sih26158_geoff3d.ipynb` on Kaggle

## 1. Upload the notebook

1. On [kaggle.com](https://www.kaggle.com), go to **Code** → **New Notebook**.
2. **File → Import Notebook** → upload `sih26158_geoff3d.ipynb` (regenerate it
   first with `python3 build_notebook.py` if you've changed anything under
   `sih3d/` — the notebook is generated from those files, never hand-edited).

## 2. Attach your video dataset

1. In the notebook editor's right sidebar, **Add Input** → **Upload Dataset**.
2. Upload a folder containing:
   - your drone video (`.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`, or `.ts`)
   - its telemetry, if you have it as a separate file: a DJI `.SRT`, or
     `.csv` / `.gpx` / `.kml` / `.json` / MAVLink `.tlog`/`.bin`. If your
     video has telemetry muxed in as a subtitle/data stream instead, you
     don't need a separate file — the notebook extracts it automatically.
   - (optional) a camera intrinsics file (`.json`/`.yaml` with `fx/fy/cx/cy`
     or a camera matrix) if you have one. Otherwise intrinsics are
     estimated from image dimensions with a typical-FOV assumption — fine
     for a first run, but real intrinsics improve accuracy.
3. No naming convention is required for the video/telemetry pair beyond
   matching by filename stem (`flight1.mp4` + `flight1.SRT`) or just being
   in the same folder — `io_detect.py` handles both. If a folder has
   multiple videos, the longest one is used automatically and the rest are
   listed as ignored in the notebook output.

## 3. (Optional) Attach the fine-tuned checkpoint dataset

The UAVFF3D authors (github.com/yanxian-ll/UAVFF3D) publish drone-domain
fine-tuned checkpoints for Pi3, Pi3X, MapAnything, and VGGT, distributed via
Baidu Netdisk. If you've downloaded and republished one as a Kaggle dataset:

1. **Add Input** → select that dataset.
2. No configuration needed — `io_detect.find_checkpoints()` scans
   `/kaggle/input` for files matching backbone-name hints (`mapanything`/
   `mapa`, `pi3x`, `pi3`, `vggt`) and prefers a filename containing "best".
3. The notebook loads it with **strict key matching** — if the checkpoint's
   state dict doesn't match the model architecture exactly (even after
   trying the one known `model.`-prefix variant), it **fails loudly** with
   the exact missing/unexpected keys rather than silently running on
   partially-random weights. If that happens, the checkpoint format likely
   differs from what's documented in `PHASE0_NOTES.md` §10 — the run falls
   back to the stock pretrained backbone and logs why.

## 4. (Optional) Attach a cache dataset, after your first run

See the "Publishing a cache dataset" cell in the notebook itself for the
exact steps — in short: after a successful run, `cache/` in the notebook's
output contains downloaded wheels/checkpoints; publish that folder as a
dataset and attach it on future runs to skip re-downloading.

## 5. Required settings

In the notebook editor's right sidebar, under **Notebook Options**:

- **Internet**: **On** (required — model weights and pip installs need it;
  the first cell checks this and fails fast with a clear message if it's
  off, rather than failing confusingly deep in the pipeline).
- **Accelerator**: **GPU T4 x2** if available (the pipeline uses a second
  GPU for masking+fusion when present), otherwise a single GPU. The
  pipeline runs on CPU as a last resort if no GPU is attached at all, but
  QUICK mode's <5-minute target will not be met on CPU.

## 6. Configure and run

At the top of the notebook, the **Config** cell has:

```python
MODE = "QUICK"          # or "FULL" for the whole video
BACKBONE = "C"           # "C" (MapAnything, recommended) or "B" (Pi3X).
                          # "A" always raises — no trained checkpoint exists
                          # for it (see PHASE0_NOTES.md); it's kept in the
                          # config switch only so all three can be selected,
                          # per the task's own requirement, not because it
                          # works.
PRIOR_MODE = "AUTO"      # or force "RGB"/"C"/"P"/"CP" to compare settings
USE_FINETUNED_CHECKPOINT = True
```

Leave the defaults for a first run. **Run All**. QUICK mode processes the
first ~90 seconds (or ~120 keyframes, whichever comes first) and should
finish in well under 5 minutes on 2x T4, excluding first-time package
installs (those add a few extra minutes the first time only — subsequent
runs with a cache dataset attached skip them).

## 7. What you should see

The notebook runs the pipeline in a background thread and renders a live
dashboard in the main thread as it goes:

- **Header**: detected video/telemetry summary, mode, GPUs, backbone,
  overall progress bar, elapsed time.
- **Pipeline strip**: 7 cards (Drone Video input, then the 6 processing
  stages) — grey = pending, blue = running, green = done, amber = a
  fallback was used for that stage (check `report.json`'s
  `fallbacks_triggered` for what and why), red = error.
- **GPU panel**: live utilization/memory sparklines per GPU.
- **Frames panel**: the frame currently being decoded + its sharpness
  score, and a scrolling filmstrip of keyframes (rejected ones greyed out).
- **Trajectory panel**: GPS track vs. the estimated camera trajectory, once
  alignment has run.
- **Geometry panel**: depth + confidence maps of the latest processed chunk.
- **Live 3D panel**: the point cloud growing chunk by chunk (downsampled to
  150k points for the live preview — the actual output files aren't
  downsampled), with an RGB/height/confidence color toggle.
- **Log panel**: the last ~30 log lines.
- **Results card**: a table of every output file with size/status, and the
  embedded `viewer.html` once the mesh stage finishes.

If a stage's card turns amber or red, that's expected robustness behavior,
not necessarily a broken run — every stage has a documented fallback path
(see each module's docstring in `sih3d/`) and the pipeline keeps going with
degraded output rather than stopping. Check `report.json`'s
`fallbacks_triggered` list for exactly what happened.

**Kaggle's "Save Version" runs the notebook headless** — nobody is watching
the live widgets during that run. The pipeline still completes normally;
`report.html` (not the live dashboard) is where you see the final state
afterward, including a static snapshot of the GPU history and stage table.

## 8. Where outputs go

Everything lands in `/kaggle/working/outputs/`:

| File | What it is |
|---|---|
| `pointcloud.las` / `.ply` | Georeferenced (if GPS was available) dense point cloud, RGB + confidence + view-count as extra dimensions |
| `mesh.obj` + `.mtl` + texture PNG, `mesh.glb` | Vertex-colored mesh, texture-baked if it fit the time budget |
| `mesh.fbx` | Best-effort via `assimp`; may be absent — check `report.json` for why, this is allowed to fail per the task spec |
| `dsm.tif`, `orthomosaic.tif` | GeoTIFF rasters, CRS-tagged if georeferenced |
| `coverage.tif` | View-count/confidence map — marks unobserved areas |
| `trajectory.geojson` / `.kml` | GPS track vs. estimated camera track, for GIS tools |
| `viewer.html` | Standalone three.js viewer — self-contained single file (point cloud, mesh, and trajectories embedded inline), works even if downloaded on its own |
| `report.json` / `report.html` | Per-stage timing, per-stage GPU utilization, point/face counts, alignment RMSE vs. GPS, collinearity index, every fallback triggered |

Kaggle's **Output** tab lets you download individual files or the whole
`outputs/` folder as a zip.

## 9. If something goes wrong

Run the notebook's **Debug** cell (right after Launch). It prints one block
covering: the pipeline's success/failure state and full traceback if it
failed, the last 100 log lines, GPU/environment info (`nvidia-smi` output,
torch/CUDA versions), and everything auto-detected about your inputs. Copy
that whole block when asking for help — it's designed to be pasted as-is.

## 10. Known things still needing on-Kaggle verification

Documented honestly rather than glossed over — see `PHASE0_NOTES.md` for
the full detail on each:

- Exact `torch-scatter`/`torch-cluster`/`xformers` wheel availability for
  whatever torch/CUDA build Kaggle's image currently ships (only matters if
  you ever need those directly — the default Backbone C path doesn't).
- Whether the real UAVFF3D-published checkpoint files (once available) use
  exactly the `{"model": state_dict}` format confirmed from the authors'
  own benchmark-conversion script, or whether the actual training-loop
  checkpoint format differs — if it differs, `strict_load_checkpoint()`
  will fail loudly with the exact key mismatch rather than silently
  running on partially-random weights, which is the correct behavior
  either way, not a bug to work around.
- Actual per-chunk view-count budget for MapAnything on a 16 GB T4 — start
  with the defaults, watch `report.json`'s GPU utilization and any OOM
  fallback-triggered halving, and tune `chunk_size` down if needed.
