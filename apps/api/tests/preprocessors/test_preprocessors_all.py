import sys
import os
import torch
from pathlib import Path
import traceback
import argparse


# Add apps/api to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all preprocessors
from src.preprocess.uniformer import UniformerSegmentor
from src.preprocess.teed import TEDDetector
from src.preprocess.ptlflow import PTLFlowDetector
from src.preprocess.open_pose import OpenposeDetector
from src.preprocess.dwpose import DwposeDetector, AnimalPoseDetector
from src.preprocess.densepose import DenseposeDetector
from src.preprocess.depth_anything.transformers import DepthAnythingDetector
from src.preprocess.lineart import LineartDetector
from src.preprocess.pidi import PidiNetDetector
from src.preprocess.scribble import ScribbleDetector, ScribbleXDogDetector
from src.preprocess.hed import HEDdetector
from src.preprocess.canny import CannyDetector
from src.preprocess.zoe.transformers import ZoeDetector, ZoeDepthAnythingDetector
from src.preprocess.tile import TileDetector
from src.preprocess.unimatch import UnimatchDetector
from src.preprocess.scribble_anime import ScribbleAnimeDetector
from src.preprocess.rembg import RembgDetector
from src.preprocess.recolor import Recolorizer
from src.preprocess.pyracanny import PyraCannyDetector
from src.preprocess.pose2d import Pose2dDetector
from src.preprocess.shuffle import ContentShuffleDetector
from src.preprocess.oneformer.transformers import OneformerSegmentor
from src.preprocess.normalbae import NormalBaeDetector
from src.preprocess.mlsd import MLSDdetector
from src.preprocess.midas.transformers import MidasDetector
from src.preprocess.metric3d import Metric3DDetector
from src.preprocess.manga_line import LineartMangaDetector
from src.preprocess.lineart_standard import LineartStandardDetector
from src.preprocess.mediapipe_face import MediapipeFaceDetector
from src.preprocess.mesh_graphormer import MeshGraphormerDetector
from src.preprocess.lineart_anime import LineartAnimeDetector
from src.preprocess.leres import LeresDetector
from src.preprocess.dwpose_nlf import DwposeNlfDetector
from src.preprocess.dsine import DsineDetector
from src.preprocess.depth_anything_v2 import DepthAnythingV2Detector
from src.preprocess.diffusion_edge import DiffusionEdgeDetector
from src.preprocess.color import ColorDetector
from src.preprocess.anime_face_segment import AnimeFaceSegmentor
from src.preprocess.binary import BinaryDetector
from src.preprocess.face2d import Face2dDetector

# List of preprocessors to test
PREPROCESSORS = [
    UniformerSegmentor,
    TEDDetector,
    PTLFlowDetector,
    OpenposeDetector,
    DwposeDetector,
    AnimalPoseDetector,
    DenseposeDetector,
    DepthAnythingDetector,
    LineartDetector,
    PidiNetDetector,
    ScribbleDetector,
    ScribbleXDogDetector,
    HEDdetector,
    CannyDetector,
    ZoeDetector,
    ZoeDepthAnythingDetector,
    TileDetector,
    UnimatchDetector,
    ScribbleAnimeDetector,
    RembgDetector,
    Recolorizer,
    PyraCannyDetector,
    Pose2dDetector,
    ContentShuffleDetector,
    OneformerSegmentor,
    NormalBaeDetector,
    MLSDdetector,
    MidasDetector,
    Metric3DDetector,
    LineartMangaDetector,
    LineartStandardDetector,
    MediapipeFaceDetector,
    MeshGraphormerDetector,
    LineartAnimeDetector,
    LeresDetector,
    DwposeNlfDetector,
    DsineDetector,
    DepthAnythingV2Detector,
    DiffusionEdgeDetector,
    ColorDetector,
    AnimeFaceSegmentor,
    BinaryDetector,
    Face2dDetector,
]

def main():
    parser = argparse.ArgumentParser(description="Test preprocessors.")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to video file to use for testing.")
    parser.add_argument("--model", "-m", type=str, help="Filter to run only models containing this string (case-insensitive).")
    args = parser.parse_args()

    # Verify custom file exists
    video_path = Path(args.file)
    if not video_path.exists():
        print(f"Error: File {video_path} not found.")
        return
    print(f"Using test file: {video_path}")

    # Filter preprocessors if requested
    models_to_test = PREPROCESSORS
    if args.model:
        models_to_test = [
            cls for cls in PREPROCESSORS 
            if args.model.lower() in cls.__name__.lower()
        ]
        if not models_to_test:
            print(f"No models found matching '{args.model}'")
            return

    success_count = 0
    fail_count = 0
    failed_models = []

    for PreprocessorClass in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {PreprocessorClass.__name__} on {video_path}...")
        
        try:
            print(f"  Loading model...")
            # Instantiate using from_pretrained
            model = PreprocessorClass.from_pretrained()
            
            print(f"  Processing video...")
            # Run on a few frames to test
            # process_video yields frames
            
            # Create a generator by calling the model instance with the video path
            # This triggers __call__ -> process_video
            generator = model(str(video_path))
            
            # Consume a few frames to ensure it's working
            count = 0
            for frame in generator:
                count += 1
                if count >= 3: # Test 3 frames
                    break
            
            print(f"  SUCCESS: Processed {count} frames.")
            success_count += 1
            
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            fail_count += 1
            failed_models.append(PreprocessorClass.__name__)
        
        # Clean up to save memory
        if 'model' in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"Total tested: {len(models_to_test)}")
    print(f"Passed: {success_count}")
    print(f"Failed: {fail_count}")
    
    if fail_count > 0:
        print("\nFailed models:")
        for name in failed_models:
            print(f"- {name}")

if __name__ == "__main__":
    main()
