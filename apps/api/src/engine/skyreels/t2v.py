from src.engine.wan.t2v import WanT2VEngine


class SkyReelsT2VEngine(WanT2VEngine):
    """SkyReels Text-to-Video Engine Implementation"""

    def run(self, **kwargs):
        """Text-to-video generation for SkyReels model"""
        # Override with fps=24 as per the original implementation
        kwargs["fps"] = kwargs.get("fps", 16)
        return super().run(**kwargs)
