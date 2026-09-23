# Vendored code notice

`sih3d/vendor/pi3/` and `sih3d/vendor/dinov2/` are copied from
[GeoFF3D](https://github.com/yanxian-ll/GeoFF3D) (Apache License 2.0), which
itself vendors Meta's Pi3/Pi3X model code and a DINOv2 backbone. Only import
paths were rewritten (`geoff3d.models.external.*` -> `sih3d.vendor.*`); no
model logic was changed. See each file's own header for original copyright.

We vendor rather than depend on GeoFF3D's SLRF orchestration (`scripts/run_slrf.py`)
because SLRF forces `load_pretrained_weights=false` and only loads weights via
a raw `state_dict` load keyed to `Pi3XWrapper`'s `model.*`-prefixed submodule,
which is a real footgun (see PHASE0_NOTES.md). We call `Pi3X.from_pretrained(...)`
directly and do our own chunking, exactly like Backbone C (MapAnything).
