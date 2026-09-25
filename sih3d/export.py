"""Write every output in outputs/, per the task's format list. Every writer
takes `georeferenced: bool` — when False (no-GPS mode), CRS is skipped on
GeoTIFF/LAS and a local coordinate frame is used instead, matching the
task's no-GPS-mode requirement. Every writer logs and returns a status
instead of raising, except for the point cloud writers (LAS/PLY), which are
never allowed to silently fail — if those break, the whole run has nothing
to show.
"""

from __future__ import annotations

import json
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .events import EventBus
from .fusion import FusedPointCloud
from .mesh import _MIN_USABLE_MESH_FACES, MeshResult, TextureBakeResult
from .telemetry import enu_to_geodetic


@dataclass
class ExportStatus:
    name: str
    path: Path | None
    ok: bool
    skipped_reason: str | None = None
    size_bytes: int = 0
    timing_s: float = 0.0


def _mesh_export_status(name: str, path: Path, n_faces: int, timing_s: float) -> ExportStatus:
    """Shared success/failure call for mesh.obj/glb/ply: a file that was
    technically written but has too few faces to be a real reconstruction
    (e.g. a 1.7 KB mesh.ply from a 44-triangle degenerate mesh) must not be
    reported as OK — mesh.py's own TSDF-vs-Poisson fallback should already
    prevent this at the source, but this is the export-side check so a
    degenerate mesh is caught here too regardless of which meshing path
    produced it."""
    size = path.stat().st_size
    if n_faces < _MIN_USABLE_MESH_FACES:
        return ExportStatus(
            name=name, path=None, ok=False, size_bytes=size, timing_s=timing_s,
            skipped_reason=f"only {n_faces} faces (min usable: {_MIN_USABLE_MESH_FACES}) — not a real reconstruction",
        )
    return ExportStatus(name=name, path=path, ok=True, size_bytes=size, timing_s=timing_s)


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------

def export_pointcloud_ply(cloud: FusedPointCloud, out_path: Path, bus: EventBus) -> ExportStatus:
    """Manual binary_little_endian PLY writer — no dependency beyond numpy,
    so this never fails just because Open3D isn't installed."""
    t0 = time.time()
    n = len(cloud.points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float confidence\nproperty ushort view_count\n"
        "end_header\n"
    ).encode("ascii")

    colors_u8 = np.clip(cloud.colors, 0, 255).astype(np.uint8)
    with open(out_path, "wb") as f:
        f.write(header)
        for i in range(n):
            f.write(struct.pack(
                "<fffBBBfH",
                float(cloud.points[i, 0]), float(cloud.points[i, 1]), float(cloud.points[i, 2]),
                int(colors_u8[i, 0]), int(colors_u8[i, 1]), int(colors_u8[i, 2]),
                float(cloud.confidence[i]), int(min(cloud.view_count[i], 65535)),
            ))

    size = out_path.stat().st_size
    bus.log(f"Wrote {out_path.name}: {n} points ({size / 1e6:.1f} MB)")
    return ExportStatus(name="pointcloud.ply", path=out_path, ok=True, size_bytes=size, timing_s=time.time() - t0)


def export_pointcloud_las(
    cloud: FusedPointCloud, out_path: Path, bus: EventBus, epsg: int | None, georeferenced: bool,
) -> ExportStatus:
    t0 = time.time()
    try:
        import laspy
    except Exception as e:
        bus.log(f"laspy not available ({e}) — pointcloud.las skipped, pointcloud.ply still produced", level="warn")
        return ExportStatus(name="pointcloud.las", path=None, ok=False, skipped_reason=str(e))

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="confidence", type=np.float32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="view_count", type=np.uint16))

    if georeferenced and epsg:
        try:
            import pyproj

            header.add_crs(pyproj.CRS.from_epsg(epsg))
        except Exception as e:
            bus.log(f"Could not attach CRS (EPSG:{epsg}) to LAS header ({e}); writing without CRS", level="warn")

    las = laspy.LasData(header)
    las.x = cloud.points[:, 0]
    las.y = cloud.points[:, 1]
    las.z = cloud.points[:, 2]
    colors_u16 = (np.clip(cloud.colors, 0, 255) * 257).astype(np.uint16)  # 0-255 -> 0-65535 exactly
    las.red = colors_u16[:, 0]
    las.green = colors_u16[:, 1]
    las.blue = colors_u16[:, 2]
    las.confidence = cloud.confidence.astype(np.float32)
    las.view_count = np.clip(cloud.view_count, 0, 65535).astype(np.uint16)

    las.write(str(out_path))
    size = out_path.stat().st_size
    crs_note = f"EPSG:{epsg}" if (georeferenced and epsg) else "no CRS (not georeferenced)"
    bus.log(f"Wrote {out_path.name}: {len(cloud.points)} points, {crs_note} ({size / 1e6:.1f} MB)")
    return ExportStatus(name="pointcloud.las", path=out_path, ok=True, size_bytes=size, timing_s=time.time() - t0)


# ---------------------------------------------------------------------------
# Mesh: OBJ+MTL(+PNG), GLB, FBX (best effort)
# ---------------------------------------------------------------------------

def export_mesh_obj(mesh_result: MeshResult, bake: TextureBakeResult | None, out_dir: Path, bus: EventBus) -> ExportStatus:
    t0 = time.time()
    if mesh_result.mesh is None:
        bus.log("No mesh produced — mesh.obj skipped", level="warn")
        return ExportStatus(name="mesh.obj", path=None, ok=False, skipped_reason="no mesh")

    mesh = mesh_result.mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    obj_path = out_dir / "mesh.obj"

    if bake is not None and bake.textured:
        mtl_path = out_dir / "mesh.mtl"
        png_path = out_dir / "mesh_texture.png"
        from PIL import Image

        Image.fromarray(bake.texture_rgb).save(png_path)
        mtl_path.write_text(
            "newmtl material0\nKa 1.0 1.0 1.0\nKd 1.0 1.0 1.0\nKs 0.0 0.0 0.0\n"
            f"map_Kd {png_path.name}\n"
        )
        uv = bake.uv  # (n_faces*3, 2), per-face-corner, matches xatlas `indices` face order
        with open(obj_path, "w") as f:
            f.write(f"mtllib {mtl_path.name}\nusemtl material0\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for uv_pair in uv:
                f.write(f"vt {uv_pair[0]:.6f} {1.0 - uv_pair[1]:.6f}\n")
            # Faces reference original mesh vertex indices for position, but
            # sequential 1-based indices for UV (one vt per face-corner).
            for fi, tri in enumerate(triangles):
                a, b, c = tri + 1
                ta, tb, tc = fi * 3 + 1, fi * 3 + 2, fi * 3 + 3
                f.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
    else:
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        with open(obj_path, "w") as f:
            for i, v in enumerate(vertices):
                if colors is not None:
                    c = colors[i]
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
                else:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for tri in triangles:
                a, b, c = tri + 1
                f.write(f"f {a} {b} {c}\n")

    bus.log(f"Wrote {obj_path.name}: {len(vertices)} verts, {len(triangles)} faces (textured={bake.textured if bake else False})")
    return _mesh_export_status("mesh.obj", obj_path, len(triangles), time.time() - t0)


def export_mesh_glb(mesh_result: MeshResult, bake: TextureBakeResult | None, out_path: Path, bus: EventBus) -> ExportStatus:
    t0 = time.time()
    if mesh_result.mesh is None:
        return ExportStatus(name="mesh.glb", path=None, ok=False, skipped_reason="no mesh")

    try:
        import trimesh
    except Exception as e:
        bus.log(f"trimesh not available ({e}) — mesh.glb skipped", level="warn")
        return ExportStatus(name="mesh.glb", path=None, ok=False, skipped_reason=str(e))

    mesh = mesh_result.mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if bake is not None and bake.textured:
        from PIL import Image

        uv = bake.uv.reshape(-1, 3, 2)
        # trimesh needs one UV per (position) vertex; since xatlas may have
        # duplicated vertices at seams, rebuild a duplicated-vertex mesh so
        # each face-corner UV lines up 1:1 with its own vertex position.
        flat_vertices = vertices[triangles].reshape(-1, 3)
        flat_uv = uv.reshape(-1, 2)
        flat_faces = np.arange(len(flat_vertices)).reshape(-1, 3)
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(bake.texture_rgb))
        visual = trimesh.visual.TextureVisuals(uv=flat_uv, material=material)
        tm = trimesh.Trimesh(vertices=flat_vertices, faces=flat_faces, visual=visual, process=False)
    else:
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        vertex_colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8) if colors is not None else None
        tm = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=vertex_colors, process=False)

    tm.export(str(out_path))
    bus.log(f"Wrote {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return _mesh_export_status("mesh.glb", out_path, len(triangles), time.time() - t0)


def export_mesh_ply(mesh_result: MeshResult, bake: TextureBakeResult | None, out_path: Path, bus: EventBus) -> ExportStatus:
    """Vertex-colored PLY — opens directly in MeshLab/CloudCompare/Blender
    with no plugins, useful for a quick local preview off Kaggle without
    needing a glTF-aware viewer. PLY has no UV-texture-image concept the
    way glb/obj do, so a successful bake's real photo texture is baked
    into per-vertex colors first (to_color()) rather than silently
    reverting to the pre-bake flat vertex-colored mesh."""
    t0 = time.time()
    if mesh_result.mesh is None:
        return ExportStatus(name="mesh.ply", path=None, ok=False, skipped_reason="no mesh")

    try:
        import trimesh
    except Exception as e:
        bus.log(f"trimesh not available ({e}) — mesh.ply skipped", level="warn")
        return ExportStatus(name="mesh.ply", path=None, ok=False, skipped_reason=str(e))

    mesh = mesh_result.mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    if bake is not None and bake.textured:
        from PIL import Image

        uv = bake.uv.reshape(-1, 3, 2)
        flat_vertices = vertices[triangles].reshape(-1, 3)
        flat_uv = uv.reshape(-1, 2)
        flat_faces = np.arange(len(flat_vertices)).reshape(-1, 3)
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(bake.texture_rgb))
        visual = trimesh.visual.TextureVisuals(uv=flat_uv, material=material)
        tm = trimesh.Trimesh(vertices=flat_vertices, faces=flat_faces, visual=visual, process=False)
        try:
            tm.visual = tm.visual.to_color()
        except Exception as e:
            bus.log(f"Could not bake texture into vertex colors for mesh.ply ({e}); exporting untextured", level="warn")
    else:
        colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
        vertex_colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8) if colors is not None else None
        tm = trimesh.Trimesh(vertices=vertices, faces=triangles, vertex_colors=vertex_colors, process=False)

    tm.export(str(out_path))
    bus.log(f"Wrote {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return _mesh_export_status("mesh.ply", out_path, len(triangles), time.time() - t0)


def export_mesh_fbx(obj_path: Path | None, out_path: Path, bus: EventBus) -> ExportStatus:
    """Best-effort only, per the task spec — assimp via apt/CLI. Logs
    clearly and never raises if assimp isn't installed or the conversion
    fails for any reason."""
    t0 = time.time()
    if obj_path is None or not obj_path.exists():
        return ExportStatus(name="mesh.fbx", path=None, ok=False, skipped_reason="no source mesh.obj")

    try:
        result = subprocess.run(
            ["assimp", "export", str(obj_path), str(out_path)],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        bus.log("assimp CLI not found — mesh.fbx skipped (best-effort only, install via 'apt-get install assimp-utils')", level="warn")
        return ExportStatus(name="mesh.fbx", path=None, ok=False, skipped_reason="assimp not installed")
    except Exception as e:
        bus.log(f"assimp FBX export raised {type(e).__name__}: {e} — mesh.fbx skipped", level="warn")
        return ExportStatus(name="mesh.fbx", path=None, ok=False, skipped_reason=str(e))

    if result.returncode != 0 or not out_path.exists():
        bus.log(f"assimp FBX export failed (exit {result.returncode}): {result.stderr[-300:]} — mesh.fbx skipped", level="warn")
        return ExportStatus(name="mesh.fbx", path=None, ok=False, skipped_reason=result.stderr[-300:])

    size = out_path.stat().st_size
    bus.log(f"Wrote {out_path.name} via assimp ({size / 1e6:.1f} MB)")
    return ExportStatus(name="mesh.fbx", path=out_path, ok=True, size_bytes=size, timing_s=time.time() - t0)


# ---------------------------------------------------------------------------
# DSM / orthomosaic / coverage rasters
# ---------------------------------------------------------------------------

def _rasterize_grid(
    points_xy: np.ndarray, values: dict[str, np.ndarray], cell_size_m: float, reduce: dict[str, str],
    device: str = "cpu",
) -> tuple[dict[str, np.ndarray], tuple[float, float], int, int]:
    """Rasterizes scattered (x,y)-indexed values onto a regular grid via
    torch.scatter_reduce, as the task spec explicitly asks for. Returns
    (arrays_by_name, (min_x,min_y) grid origin, width, height). Cells with
    no points get NaN (caller decides the nodata value)."""
    import torch

    min_x, min_y = points_xy[:, 0].min(), points_xy[:, 1].min()
    max_x, max_y = points_xy[:, 0].max(), points_xy[:, 1].max()
    width = max(int(np.ceil((max_x - min_x) / cell_size_m)) + 1, 1)
    height = max(int(np.ceil((max_y - min_y) / cell_size_m)) + 1, 1)

    col = ((points_xy[:, 0] - min_x) / cell_size_m).astype(np.int64).clip(0, width - 1)
    row = ((max_y - points_xy[:, 1]) / cell_size_m).astype(np.int64).clip(0, height - 1)  # flip y: raster row 0 = north
    flat_idx = torch.from_numpy(row * width + col).to(device)

    out = {}
    counts = torch.zeros(width * height, device=device)
    counts.scatter_add_(0, flat_idx, torch.ones(len(points_xy), device=device))

    for name, vals in values.items():
        vals_t = torch.from_numpy(np.ascontiguousarray(vals)).float().to(device)
        grid = torch.full((width * height,), float("nan"), device=device)
        mode = reduce.get(name, "mean")
        if mode == "amax":
            grid = torch.full((width * height,), float("-inf"), device=device)
            grid.scatter_reduce_(0, flat_idx, vals_t, reduce="amax", include_self=True)
            grid[counts == 0] = float("nan")
        else:  # mean
            summed = torch.zeros(width * height, device=device)
            summed.scatter_add_(0, flat_idx, vals_t)
            grid = torch.where(counts > 0, summed / counts.clamp_min(1), torch.full_like(summed, float("nan")))
        out[name] = grid.reshape(height, width).cpu().numpy()

    out["_count"] = counts.reshape(height, width).cpu().numpy()
    return out, (float(min_x), float(max_y)), width, height


def _write_geotiff(path: Path, bands: dict[str, np.ndarray], origin_topleft: tuple[float, float], cell_size_m: float, epsg: int | None, bus: EventBus) -> bool:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except Exception as e:
        bus.log(f"rasterio not available ({e}) — {path.name} skipped", level="warn")
        return False

    band_names = list(bands.keys())
    height, width = next(iter(bands.values())).shape
    transform = from_origin(origin_topleft[0], origin_topleft[1], cell_size_m, cell_size_m)
    crs = None
    if epsg:
        try:
            crs = rasterio.crs.CRS.from_epsg(epsg)
        except Exception as e:
            bus.log(f"Could not build CRS EPSG:{epsg} for {path.name} ({e}); writing without CRS", level="warn")

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=len(band_names),
        dtype="float32", crs=crs, transform=transform, nodata=np.nan,
    ) as dst:
        for i, name in enumerate(band_names, start=1):
            dst.write(bands[name].astype(np.float32), i)
            dst.set_band_description(i, name)
    return True


def export_dsm_orthomosaic(
    cloud: FusedPointCloud, out_dir: Path, bus: EventBus, epsg: int | None, georeferenced: bool, cell_size_m: float = 0.2,
) -> tuple[ExportStatus, ExportStatus]:
    t0 = time.time()
    if len(cloud.points) < 10:
        skip = ExportStatus(name="dsm.tif/orthomosaic.tif", path=None, ok=False, skipped_reason="too few points")
        return skip, skip

    grids, origin, width, height = _rasterize_grid(
        cloud.points[:, :2],
        {"height": cloud.points[:, 2], "r": cloud.colors[:, 0], "g": cloud.colors[:, 1], "b": cloud.colors[:, 2]},
        cell_size_m, reduce={"height": "amax", "r": "mean", "g": "mean", "b": "mean"},
    )
    dsm_path = out_dir / "dsm.tif"
    ortho_path = out_dir / "orthomosaic.tif"
    epsg_used = epsg if georeferenced else None

    dsm_ok = _write_geotiff(dsm_path, {"height": grids["height"]}, origin, cell_size_m, epsg_used, bus)
    ortho_ok = _write_geotiff(ortho_path, {"r": grids["r"], "g": grids["g"], "b": grids["b"]}, origin, cell_size_m, epsg_used, bus)

    if dsm_ok:
        bus.log(f"Wrote {dsm_path.name}: {width}x{height} cells @ {cell_size_m}m" + ("" if georeferenced else " (no CRS — not georeferenced)"))
    if ortho_ok:
        bus.log(f"Wrote {ortho_path.name}: {width}x{height} cells @ {cell_size_m}m")

    elapsed = time.time() - t0
    # ok=True with size_bytes defaulting to 0 (the dataclass default) made
    # every successful run of this function LOOK identical to a broken one
    # in report.json — both showed "ok": true, "size_bytes": 0. These files
    # were never actually empty; size_bytes was just never populated.
    dsm_size = dsm_path.stat().st_size if dsm_ok and dsm_path.exists() else 0
    ortho_size = ortho_path.stat().st_size if ortho_ok and ortho_path.exists() else 0
    dsm_status = ExportStatus(name="dsm.tif", path=dsm_path if dsm_ok else None, ok=dsm_ok and dsm_size > 0, size_bytes=dsm_size, timing_s=elapsed)
    ortho_status = ExportStatus(name="orthomosaic.tif", path=ortho_path if ortho_ok else None, ok=ortho_ok and ortho_size > 0, size_bytes=ortho_size, timing_s=elapsed)
    return dsm_status, ortho_status


def export_coverage(
    cloud: FusedPointCloud, out_dir: Path, bus: EventBus, epsg: int | None, georeferenced: bool, cell_size_m: float = 0.2,
) -> ExportStatus:
    t0 = time.time()
    if len(cloud.points) < 10:
        return ExportStatus(name="coverage.tif", path=None, ok=False, skipped_reason="too few points")

    grids, origin, width, height = _rasterize_grid(
        cloud.points[:, :2],
        {"view_count": cloud.view_count.astype(np.float32), "confidence": cloud.confidence},
        cell_size_m, reduce={"view_count": "mean", "confidence": "mean"},
    )
    path = out_dir / "coverage.tif"
    ok = _write_geotiff(path, {"view_count": grids["view_count"], "confidence": grids["confidence"]}, origin, cell_size_m, epsg if georeferenced else None, bus)
    if ok:
        unobserved_frac = float(np.isnan(grids["view_count"]).mean())
        bus.log(f"Wrote {path.name}: {unobserved_frac * 100:.1f}% of the bounding area unobserved")
    size = path.stat().st_size if ok and path.exists() else 0
    return ExportStatus(name="coverage.tif", path=path if ok else None, ok=ok and size > 0, size_bytes=size, timing_s=time.time() - t0)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def export_trajectories(
    gps_track_enu: list[tuple[float, float, float]],
    camera_track_enu: list[tuple[float, float, float]] | None,
    origin_latlonalt: tuple[float, float, float],
    out_dir: Path, bus: EventBus,
) -> tuple[ExportStatus, ExportStatus]:
    t0 = time.time()
    lat0, lon0, alt0 = origin_latlonalt

    def to_lonlat(track):
        return [(lon, lat) for lat, lon, _alt in (enu_to_geodetic(e, n, u, lat0, lon0, alt0) for e, n, u in track)]

    features = []
    if gps_track_enu:
        features.append({
            "type": "Feature", "properties": {"name": "gps_track"},
            "geometry": {"type": "LineString", "coordinates": to_lonlat(gps_track_enu)},
        })
    if camera_track_enu:
        features.append({
            "type": "Feature", "properties": {"name": "estimated_camera_track"},
            "geometry": {"type": "LineString", "coordinates": to_lonlat(camera_track_enu)},
        })

    geojson_path = out_dir / "trajectory.geojson"
    geojson_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))

    kml_placemarks = []
    for feat in features:
        coords_str = " ".join(f"{lon},{lat},0" for lon, lat in feat["geometry"]["coordinates"])
        kml_placemarks.append(
            f"<Placemark><name>{feat['properties']['name']}</name>"
            f"<LineString><coordinates>{coords_str}</coordinates></LineString></Placemark>"
        )
    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>\n'
        + "\n".join(kml_placemarks) + "\n</Document></kml>\n"
    )
    kml_path = out_dir / "trajectory.kml"
    kml_path.write_text(kml)

    bus.log(f"Wrote trajectory.geojson/.kml: {len(features)} track(s)")
    elapsed = time.time() - t0
    return (
        ExportStatus(name="trajectory.geojson", path=geojson_path, ok=True, timing_s=elapsed),
        ExportStatus(name="trajectory.kml", path=kml_path, ok=True, timing_s=elapsed),
    )
