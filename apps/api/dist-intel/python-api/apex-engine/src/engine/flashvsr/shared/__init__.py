from diffusers.video_processor import VideoProcessor
from src.engine.flashvsr.shared.color_corrector import TorchColorCorrectorWavelet
from src.engine.wan.shared import WanShared


class FlashVSRShared(WanShared):
    """Base class for FlashVSR engine implementations containing common functionality"""

    def __init__(self, yaml_path: str, **kwargs):
        super().__init__(yaml_path, **kwargs)
        self.color_corrector = TorchColorCorrectorWavelet()
