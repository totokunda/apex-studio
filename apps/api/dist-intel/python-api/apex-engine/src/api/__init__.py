"""
API module for Apex Engine
Now powered by Ray for distributed task processing
"""

"""
Important: keep this package `__init__` free of import-time side effects.

Importing modules like `ray_tasks` here can trigger Ray's auto-init hooks (directly or
indirectly) during *module import*, which is unsafe under uvicorn and can lead to
intermittent "core worker already initialized" crashes when the app later initializes Ray
explicitly during startup.
"""

__all__: list[str] = []
