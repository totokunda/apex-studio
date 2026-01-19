"""
Quick sanity check for the NumPy+OpenCV software renderer that replaced `opendr`.

Run (from apps/api):
  .\\.venv\\Scripts\\python.exe .\\scripts\\dev\\demo_numpy_opencv_renderer.py

NOTE:
We load the module by file path to avoid importing `src.preprocess.mesh_graphormer`,
whose `__init__.py` currently has heavy side effects (model downloads).
"""

from __future__ import annotations

from pathlib import Path
import importlib.util

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]  # apps/api/
_RENDERER_PY = (
    _ROOT
    / "src"
    / "preprocess"
    / "mesh_graphormer"
    / "custom_mesh_graphormer"
    / "utils"
    / "renderer.py"
)

_spec = importlib.util.spec_from_file_location("mesh_graphormer_renderer", _RENDERER_PY)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to load renderer module spec from: {_RENDERER_PY}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Renderer = _mod.Renderer


def _make_cube() -> tuple[np.ndarray, np.ndarray]:
    # Cube centered at origin, side length 2 (z will be shifted by camera_t).
    v = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float32,
    )

    # 12 triangles (two per face)
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return v, f


def main() -> None:
    out_path = _ROOT / ".tmp" / "renderer_demo.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = 512, 512
    bg = np.ones((H, W, 3), dtype=np.float32) * np.array(
        [0.05, 0.05, 0.05], dtype=np.float32
    )

    vertices, faces = _make_cube()
    r = Renderer(width=W, height=H, faces=faces)

    img = r.render(
        vertices,
        faces=faces,
        img=bg,
        use_bg=True,
        camera_rot=np.zeros(3, dtype=np.float32),
        camera_t=np.array([0.0, 0.0, 6.0], dtype=np.float32),
        focal_length=800.0,
        camera_center=np.array([W * 0.5, H * 0.5], dtype=np.float32),
        body_color="light_blue",
    )

    # Save with OpenCV (expects BGR uint8)
    bgr = np.clip(img[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), bgr)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
