from src.preprocess.aux_cache import AuxillaryCache
from src.preprocess.dwpose import DwposeDetector
from diffusers.utils.export_utils import export_to_video
input_path = "/home/tosin_coverquick_co/apex-studio/apps/api/apex-test-suite/outputs/wan-2.2-fun-a14b-control-1.0.0.v1.mp4"
cache = AuxillaryCache(
        input_path,
        "dwpose"
    )

progress_callback = lambda x, y: print(f"Processing frame {x} of {y}")
dwpose = DwposeDetector.from_pretrained()
frame_range = cache._get_video_frame_range()
total_frames = len([f for f in frame_range if f not in cache.cached_frames])
frames = cache.video_frames(batch_size=1)
print(f"Total frames: {total_frames}")
result = dwpose(
    frames,
    job_id="test",
    progress_callback=progress_callback,
    total_frames=total_frames,
)

cache.save_result(result)
print(cache.get_result_path())