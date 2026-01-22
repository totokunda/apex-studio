from src.engine.wan.i2v import WanI2VEngine


class SkyReelsI2VEngine(WanI2VEngine):
    """SkyReels Image-to-Video Engine Implementation"""

    def run(self, **kwargs):
        """Image-to-video generation for SkyReels model"""
        # Override with fps=24 as per the original implementation
        kwargs["fps"] = kwargs.get("fps", 16)
        return super().run(**kwargs)
