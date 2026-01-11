from __future__ import annotations

import importlib

from .common import SmokeContext, log, fail


def run(ctx: SmokeContext) -> None:
    log("[smoke] preprocessors import")
    try:
        from src.api.preprocessor_registry import (  # type: ignore
            list_preprocessors,
            get_preprocessor_details,
        )
    except Exception as e:
        fail(f"Failed to import preprocessor registry: {e}")

    try:
        preprocessors = list_preprocessors(check_downloaded=False)
    except Exception as e:
        fail(f"Failed to list preprocessors: {e}")

    bad: list[tuple[str, str]] = []
    for info in preprocessors:
        pid = info.get("id") or info.get("name") or "<unknown>"
        try:
            det = get_preprocessor_details(pid)
            mod = det.get("module")
            cls_name = det.get("class")
            if not mod or not cls_name:
                raise RuntimeError(f"missing module/class in manifest for {pid}")
            m = importlib.import_module(mod)
            getattr(m, cls_name)
        except Exception as e:
            bad.append((str(pid), str(e)))

    if bad:
        msgs = "\n".join([f" - {pid}: {err}" for pid, err in bad])
        fail("Preprocessor import failures:\n" + msgs)

    log(f"[smoke] preprocessors import ok: {len(preprocessors)}")


