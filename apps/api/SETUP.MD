1. Create a new YAML file in the `manifest/engine` directory, for the given `manifest/engine_old`
 - This YAML file should have the same setup as the file `/home/tosin_coverquick_co/apex/manifest/engine/qwenimage/qwenimage-edit-1.0.0.v1.yml`, where we now have an inputs section, and a robust metadata section. We instead want to use the associated parameters as specified in the associated python file for the given engine. We additionally want our model_path to be a list of paths with a variant, precision and type, as shown in the qwenimage-edit-1.0.0.v1.yml file.
2. Go to the associated python file for the given engine, and run type, and add the a safe_emit_progress function to each part of the engine run function. Use `/home/tosin_coverquick_co/apex/src/engine/qwenimage/edit.py` as a reference. Additionally you need to add a denoise_progress_callback to the engine run function. 
3. Go the associated denoise method file, and add the `denoise_progress_callback` to the denoise method. Use `/home/tosin_coverquick_co/apex/src/denoise/qwenimage_denoise.py` as a reference, for how it should be added. 
4. Please go the the `base.py` file for a given engine and add the following if not already present: 
    ```python
    def __getattr__(self, name: str):  # noqa: D401
        """Delegate attribute access to the composed BaseEngine when not found here."""
        try:
            return getattr(self.main_engine, name)
        except AttributeError as exc:
            raise AttributeError(f"{self.__class__.__name__!s} has no attribute '{name}'") from exc

    # Improve editor introspection (e.g., autocomplete) by exposing attributes of main_engine
    def __dir__(self):
        return sorted(set[str](list[str](super().__dir__()) + dir(self.main_engine)))
``` 
- Afterwards please delete all methods that would be found in the `base_engine.py` file for the given engine, as the getattr method is now handling all attribute access.
5. (Optional): You may need to add a custom _render_step function to the `base.py` file for the given engine. If already present, ignore this step, otherwise you need the _render_step function to have the exact same logic as what is used to decode and return the image from the engine. Please check if the process differs from the base_engine.py file implementation. 
- If you do need to add a custom _render_step function, please add the to the following `__init__.py` file for the given engine: ```python
    def _render_step(self, latents, render_on_step_callback):
        impl = getattr(self, "implementation_engine", None)
        if impl is not None and hasattr(impl, "_render_step"):
            try:
                return impl._render_step(latents, render_on_step_callback)
            except Exception:
                pass
        # Fallback to BaseEngine behavior
        return super()._render_step(latents, render_on_step_callback)
```
