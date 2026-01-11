from __future__ import annotations

from .common import SmokeContext, log, fail


def run(ctx: SmokeContext) -> None:
    log("[smoke] api import + openapi")
    try:
        import src.api.main as api_main  # type: ignore

        app = getattr(api_main, "app", None)
        if app is None:
            fail("src.api.main imported but `app` is missing")
        app.openapi()
        log("[smoke] api openapi ok")
    except Exception as e:
        fail(f"API import/openapi failed: {e}")


