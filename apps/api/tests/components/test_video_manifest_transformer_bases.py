from pathlib import Path

import yaml


def test_video_manifest_transformer_bases():
    api_root = Path(__file__).resolve().parents[2]
    manifest_dir = api_root / "manifest" / "video"
    assert manifest_dir.is_dir(), f"Missing manifest dir: {manifest_dir}"

    allowed = {
        "ltx2.base",
        "hunyuanvideo15.base",
        "wan.base",
        "wan.vace",
        "wan.fun",
        "wan.fun_vace",
        "wan.animate",
        "wan.multitalk",
        "wan.scail",
        "wan.humo",
        "wan.s2v",
        "wan.ovi",
        "wan.lynx",
        "wan.lynx_lite",
    }

    for path in sorted(manifest_dir.glob("*.yml")):
        doc = yaml.safe_load(path.read_text())
        components = doc.get("spec", {}).get("components", [])
        for comp in components:
            if comp.get("type") != "transformer":
                continue
            base = comp.get("base")
            assert base in allowed, f"{path.name}: unexpected transformer base {base}"
