from __future__ import annotations

from .common import SmokeContext, log, fail, iter_manifest_files


def run(ctx: SmokeContext) -> None:
    manifest_dir = ctx.bundle_root / "manifest"
    if not manifest_dir.exists():
        log("[smoke] manifest dir not found; skipping manifest parse")
        return

    try:
        import yaml  # type: ignore
    except Exception as e:
        fail(f"Missing PyYAML for manifest parsing: {e}")

    for p in iter_manifest_files(manifest_dir):
        try:
            obj = yaml.safe_load(p.read_text(encoding="utf-8"))
            if obj is None:
                fail(f"Manifest is empty: {p}")
        except Exception as e:
            fail(f"Failed to parse manifest {p}: {e}")

    log("[smoke] manifest parse ok")
