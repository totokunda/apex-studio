from src.converters.base_converter import BaseConverter
from src.converters.utils import update_state_dict_
from typing import Dict, Any, List


class WhisperConverter(BaseConverter):
    def __init__(self):
        super().__init__()
    
    def convert(self, state_dict: Dict[str, Any], model_keys: List[str] = None):
        # strip model. prefix in place
        for key in list(state_dict.keys()):
            if key.startswith("model."):
                update_state_dict_(state_dict, key, key[len("model.") :])
        return state_dict
    

class Wav2Vec2ModelMultitalkConverter(BaseConverter):
    def __init__(self):
        super().__init__()
    
    def convert(self, state_dict: Dict[str, Any], model_keys: List[str] = None):
        # strip model. prefix in place
        for key in list(state_dict.keys()):
            if key.startswith("wav2vec2."):
                update_state_dict_(state_dict, key, key[len("wav2vec2.") :])
        return state_dict