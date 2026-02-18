from __future__ import annotations

import threading
import time

from src.api import ray_app


class _FakeRay:
    def __init__(self) -> None:
        self.initialized = False

    def is_initialized(self) -> bool:
        return bool(self.initialized)

    def shutdown(self) -> None:
        self.initialized = False


def _reset_ray_init_state() -> None:
    with ray_app._ray_init_cond:
        ray_app._ray_init_state = "not_started"
        ray_app._ray_init_error = None
        ray_app._ray_init_cond.notify_all()


def test_get_ray_app_waits_while_init_is_in_flight(monkeypatch):
    fake_ray = _FakeRay()
    init_started = threading.Event()
    allow_init_finish = threading.Event()
    init_calls = 0

    def fake_init_once() -> None:
        nonlocal init_calls
        init_calls += 1
        # Simulate Ray flipping to initialized before full init completes.
        fake_ray.initialized = True
        init_started.set()
        assert allow_init_finish.wait(timeout=2.0)

    monkeypatch.setattr(ray_app, "ray", fake_ray)
    monkeypatch.setattr(ray_app, "_init_ray_once", fake_init_once)
    _reset_ray_init_state()

    errors: list[Exception] = []
    second_done = threading.Event()

    def _runner(done_event: threading.Event | None = None) -> None:
        try:
            ray_app.get_ray_app()
        except Exception as e:  # pragma: no cover - should not happen
            errors.append(e)
        finally:
            if done_event is not None:
                done_event.set()

    t1 = threading.Thread(target=_runner, daemon=True)
    t1.start()
    assert init_started.wait(timeout=1.0), "first initializer never started"

    t2 = threading.Thread(target=_runner, args=(second_done,), daemon=True)
    t2.start()
    time.sleep(0.15)
    assert not second_done.is_set(), "second caller should block until init completes"

    allow_init_finish.set()
    t1.join(timeout=1.5)
    t2.join(timeout=1.5)

    assert not errors
    assert init_calls == 1
    assert ray_app.is_ray_ready()


def test_get_ray_app_accepts_external_init_without_reinit(monkeypatch):
    fake_ray = _FakeRay()
    fake_ray.initialized = True
    init_called = False

    def fake_init_once() -> None:
        nonlocal init_called
        init_called = True

    monkeypatch.setattr(ray_app, "ray", fake_ray)
    monkeypatch.setattr(ray_app, "_init_ray_once", fake_init_once)
    _reset_ray_init_state()

    got = ray_app.get_ray_app()

    assert got is fake_ray
    assert not init_called
    assert ray_app.is_ray_ready()
