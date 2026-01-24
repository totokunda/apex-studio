"""
This file contains a Processor that can be used to process images with controlnet aux processors
"""

import io
import logging
from typing import Dict, Optional, Union

from PIL import Image

LOGGER = logging.getLogger(__name__)


MODEL_PARAMS = {
    "scribble_hed": {"scribble": True},
    "softedge_hed": {"scribble": False},
    "scribble_hedsafe": {"scribble": True, "safe": True},
    "softedge_hedsafe": {"scribble": False, "safe": True},
    "depth_midas": {},
    "mlsd": {},
    "openpose": {"include_body": True, "include_hand": False, "include_face": False},
    "openpose_face": {
        "include_body": True,
        "include_hand": False,
        "include_face": True,
    },
    "openpose_faceonly": {
        "include_body": False,
        "include_hand": False,
        "include_face": True,
    },
    "openpose_full": {"include_body": True, "include_hand": True, "include_face": True},
    "openpose_hand": {
        "include_body": False,
        "include_hand": True,
        "include_face": False,
    },
    "scribble_pidinet": {"safe": False, "scribble": True},
    "softedge_pidinet": {"safe": False, "scribble": False},
    "scribble_pidsafe": {"safe": True, "scribble": True},
    "softedge_pidsafe": {"safe": True, "scribble": False},
    "normal_bae": {},
    "lineart_realistic": {"coarse": False},
    "lineart_coarse": {"coarse": True},
    "lineart_anime": {},
    "canny": {},
    "shuffle": {},
    "depth_zoe": {},
    "depth_leres": {"boost": False},
    "depth_leres++": {"boost": True},
    "mediapipe_face": {},
    "tile": {},
}
