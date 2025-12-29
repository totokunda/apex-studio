from typing import Tuple, Any
import numpy as np
from loguru import logger
import numpy as np
import PIL.Image
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, TensorDataset
from transformers import SiglipModel, SiglipProcessor
import cv2
from src.postprocess.base import (
    BasePostprocessor,
    PostprocessorCategory,
    postprocessor_registry,
)

from retinaface.data import cfg_re50
from retinaface.layers.functions.prior_box import PriorBox
from retinaface.models.retinaface import RetinaFace
import pathlib
import os
import json
from typing import Iterable, Union
from dataclasses import dataclass
import re
from src.utils.defaults import DEFAULT_POSTPROCESSOR_SAVE_PATH

from retinaface.utils.nms.py_cpu_nms import py_cpu_nms


def read_keyword_list_from_dir(folder_path: str) -> list[str]:
    """Read keyword list from all files in a folder."""
    output_list = []
    file_list = []
    # Get list of files in the folder
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_list.append(file)

    # Process each file
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        try:
            with open(file_path, "r") as f:
                output_list.extend([line.strip() for line in f.readlines()])
        except Exception as e:
            logger.error(f"Error reading file {file}: {str(e)}")

    return output_list


def to_ascii(prompt: str) -> str:
    """Convert prompt to ASCII."""
    return re.sub(r"[^\x00-\x7F]+", " ", prompt)


def pixelate_face(face_img: np.ndarray, blocks: int = 5) -> np.ndarray:
    """
    Pixelate a face region by reducing resolution and then upscaling.

    Args:
        face_img: Face region to pixelate
        blocks: Number of blocks to divide the face into (in each dimension)

    Returns:
        Pixelated face region
    """
    h, w = face_img.shape[:2]
    # Shrink the image and scale back up to create pixelation effect
    temp = cv2.resize(face_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def filter_detected_boxes(
    boxes, scores, confidence_threshold, nms_threshold, top_k, keep_top_k
):
    """Filter boxes based on confidence score and remove overlapping boxes using NMS."""
    # Keep detections with confidence above threshold
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # Sort by confidence and keep top K detections
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    scores = scores[order]

    # Run non-maximum-suppression (NMS) to remove overlapping boxes
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    dets = dets[:keep_top_k, :]
    boxes = dets[:, :-1]
    return boxes


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/utils/box_utils.py to handle batched inputs
def decode_batch(loc, priors, variances):
    """Decode batched locations from predictions using priors and variances.

    Args:
        loc (tensor): Batched location predictions for loc layers.
            Shape: [batch_size, num_priors, 4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4]
        variances: (list[float]): Variances of prior boxes.

    Return:
        Decoded batched bounding box predictions
            Shape: [batch_size, num_priors, 4]
    """
    batch_size = loc.size(0)
    priors = priors.unsqueeze(0).expand(batch_size, -1, -1)

    boxes = torch.cat(
        (
            priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
        ),
        dim=2,
    )

    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def _check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    logger.debug("Missing keys:{}".format(len(missing_keys)))
    logger.debug("Unused checkpoint keys:{}".format(len(unused_pretrained_keys)))
    logger.debug("Used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def _remove_prefix(state_dict, prefix):
    """Old version of the model is stored with all names of parameters sharing common prefix 'module.'"""
    logger.debug("Removing prefix '{}'".format(prefix))

    def f(x):
        return x.split(prefix, 1)[-1] if x.startswith(prefix) else x

    return {f(key): value for key, value in state_dict.items()}


# Adapted from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
def load_model(model, pretrained_path):
    logger.debug("Loading pretrained model from {}".format(pretrained_path))
    pretrained_dict = torch.load(pretrained_path, weights_only=True)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = _remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = _remove_prefix(pretrained_dict, "module.")
    _check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


VIOLENCE = {
    "category": "violence",
    "blocked_reason": """This prompt is blocked due to the presence of violent content.""",
    "instruction": """O1: Violence. Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Contain unpleasant vivid violent imagery or violent description that arouses discomfort in the reader.
Can
- Provide information on violence.
- Discuss the topics of violence, and can discuss historical events involving violence.
""",
}

SEXUAL = {
    "category": "sexual",
    "blocked_reason": """This prompt is blocked due to the presence of sexual content.""",
    "instruction": """O2: Sexual. Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual
  activities.
- Engage or encourage sexual activity that is considered illegal.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
""",
}

CRIMINAL_PLANNING = {
    "category": "criminal_planning",
    "blocked_reason": """This prompt is blocked due to the presence of criminal planning/confessions content.""",
    "instruction": """O3: Criminal Planning/Confessions. Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the
  person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
- Have a confession of crime that was committed or will be committed in future.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used
  to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide
  detailed information on how to defeat those systems.
""",
}

WEAPONS = {
    "category": "weapons",
    "blocked_reason": """This prompt is blocked due to the presence of weapons content.""",
    "instruction": """O4: Guns and Illegal Weapons. Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that
  they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
- Assist or encourage people to execute or own other illegal weapons for example biological agents, or chemical
  weapons, including instructions on how to create or use them.
Can
- Discuss firearms and the arguments for and against firearm ownership.
""",
}

SUBSTANCE_ABUSE = {
    "category": "substance_abuse",
    "blocked_reason": """This prompt is blocked due to the presence of substance abuse content.""",
    "instruction": """O5: Controlled/Regulated Substances. Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including
  but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex:
  what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to
  provide step-by-step instructions for their creation.
""",
}

SUICIDE = {
    "category": "suicide",
    "blocked_reason": "This prompt is blocked due to the presence of suicide and self harm content.",
    "instruction": """O6: Suicide and Self Harm. Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who
  express or imply that they wish to harm themselves.
- Contain confessions or intent of self harm
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by
  those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm
  themselves.
""",
}

CSAM = {
    "category": "csam",
    "blocked_reason": "This prompt contains material related to Child Sexual Abuse",
    "instruction": """O7: Sexual (minor). Should not
- Assist, engage or encourage intention to engage in sexual activity with an underage/minor.
- Discuss or plan sexual acts, sharing of media of sexual nature, personal information or intention to meet in person
  for sexual activity with a minor.
- Should not include discussion where private information is shared with underage minors for purposes of acting on them
  and taking actions that are sexual in nature including meeting or sharing media.
- Should not discuss child abuse or engage, encourage, assist in sexual grooming/trafficking of children.
Can
- Provide information about sex education for use in schools.
""",
}

HATRED = {
    "category": "hatred",
    "blocked_reason": "This prompt is blocked due to the presence of hate speech content.",
    "instruction": """O8: Hate/Identity Hate. Should not
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like
  their
race, color, ethnicity, religion, national origin, sexual orientation, gender, gender identity, or disability or caste.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults
  based on those characteristics.
- Should not discuss prejudice towards a group classified as protected according to the law of Germany, UK, USA or any
  other country.
- Should not contain slurs.
- Should not contain text that is distasteful, offensive, or hateful that causes others to feel grief, anger, or fear.
Can
- Provide information on discrimination.
- Discuss the topics of discrimination, and can discuss historical events involving discrimination.
- Discuss neutral statements referring to group identity and contextualized hate speech (that talks about hate but does
  not incite hate).
""",
}

HARASSMENT = {
    "category": "harassment",
    "blocked_reason": "This prompt is blocked due to the presence of harassment content.",
    "instruction": """O10: Harassment. Should not
- Have content that may be used to torment or annoy individuals in real life, or make harassment more likely to occur.
- Engage in harassment, misconduct and unlawful discrimination
- Single out an individual or marginalization and/or retaliation based on protected characteristics like Race, Color,
  Gender, Sex,
Sexual orientation, Gender identity, and gender expression, National origin, Ethnicity, Disability (including being
regarded as disabled), Religion, Age (40+), Pregnancy (including pregnancy, childbirth or related medical conditions),
Genetic information, Military or veteran status, Citizenship status, Political activity, or affiliation,
- Taking or requesting statutorily protected leave Body characteristics.
- Promise to give a benefit, or a threat to retaliate or take an adverse action based on a response to harassment and
  misconduct.
- Bullying.
- Being derogatory and humiliating toward an individual but not necessarily protected characteristics under law.
- Making rude or insulting comments, demeaning, and objectifying terms toward an individual.
Can
- Discuss negative impacts of harassment and bullying based on protected characteristics like race, color, gender, sex,
  nationality, within a contextualized setting.
- Provide information on protected characteristics or refers, cites material that describes what protected
  characteristics are.
- Provide information or refer or cite material that informs what harassment is.
""",
}

THREAT = {
    "category": "threat",
    "blocked_reason": "This prompt is blocked due to the presence of threat content.",
    "instruction": """O11: Threat. Should not
- Discuss a communication of intent to inflict harm or loss on another person.
- Should not contain discussion of intention of intimidation which is a tactic used between conflicting parties to make
  the other timid or psychologically insecure for coercion or control.
- Should not contain intention of knowingly putting another person in fear of bodily injury.
- Should not assist or plan injury, physical or mental damage, act or instance of injury, or material or any kind of
  loss to a person.
- Should not contain threats that are forbidden by law as intent to obtain a monetary advantage or to compel a person
  to act against their will.
Can
- Provide information about threat is and the meaning of threat.
""",
}

PROFANITY = {
    "category": "profanity",
    "blocked_reason": "This prompt is blocked due to the presence of profanity content.",
    "instruction": """O12: Profanity. Should not
- Contain swear words, curse words, or other obscene or profane language
- This includes offensive words used without any intention to act on them or without any explicit targeted recipient
""",
}

UNSAFE_CATEGORIES = [
    VIOLENCE,
    SEXUAL,
    CRIMINAL_PLANNING,
    WEAPONS,
    SUBSTANCE_ABUSE,
    SUICIDE,
    CSAM,
    HATRED,
    HARASSMENT,
    THREAT,
    PROFANITY,
]

CLASS_IDX_TO_NAME = {
    0: "Safe",
    1: "Sexual_Content",
    2: "Violence",
    3: "Drugs",
    4: "Child_Abuse",
    5: "Hate_and_Harassment",
    6: "Self-Harm",
}

# RetinaFace model constants from https://github.com/biubug6/Pytorch_Retinaface/blob/master/detect.py
TOP_K = 5_000
KEEP_TOP_K = 750
NMS_THRESHOLD = 0.4


CENSOR = "*"
COSMOS_GUARDRAIL_CHECKPOINT = "nvidia/Cosmos-1.0-Guardrail"


class ContentSafetyGuardrail:
    def is_safe(self, **kwargs) -> Tuple[bool, str]:
        raise NotImplementedError(
            "ContentSafetyGuardrail::is_safe method must be implemented by child classes"
        )


class PostprocessingGuardrail:
    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "PostprocessingGuardrail::postprocess method must be implemented by child classes"
        )


class GuardrailRunner(torch.nn.Module):
    def __init__(
        self,
        safety_models: list[ContentSafetyGuardrail] | None = None,
        generic_block_msg: str = "",
        generic_safe_msg: str = "",
        postprocessors: list[PostprocessingGuardrail] | None = None,
    ):
        super().__init__()
        self.safety_models = safety_models
        self.generic_block_msg = generic_block_msg
        self.generic_safe_msg = (
            generic_safe_msg if generic_safe_msg else "Prompt is safe"
        )
        self.postprocessors = postprocessors

    def run_safety_check(self, input: Any) -> Tuple[bool, str]:
        """Run the safety check on the input."""
        if not self.safety_models:
            logger.warning("No safety models found, returning safe")
            return True, self.generic_safe_msg

        for guardrail in self.safety_models:
            guardrail_name = str(guardrail.__class__.__name__).upper()
            logger.debug(f"Running guardrail: {guardrail_name}")
            safe, message = guardrail.is_safe(input)
            if not safe:
                reasoning = (
                    self.generic_block_msg
                    if self.generic_block_msg
                    else f"{guardrail_name}: {message}"
                )
                return False, reasoning

        return True, self.generic_safe_msg

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Run the postprocessing on the video frames."""
        if not self.postprocessors:
            logger.warning("No postprocessors found, returning original frames")
            return frames

        for guardrail in self.postprocessors:
            guardrail_name = str(guardrail.__class__.__name__).upper()
            logger.debug(f"Running guardrail: {guardrail_name}")
            frames = guardrail.postprocess(frames)

        return frames


@dataclass
class ModelConfig:
    input_size: int = 1152
    num_classes: int = 7


class SafetyClassifier(torch.nn.Module):
    def __init__(self, input_size: int = 1024, num_classes: int = 2):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_classes),
            # Note: No activation function here; CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        return self.layers(x)


class VideoSafetyModel(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.network = SafetyClassifier(
            input_size=config.input_size, num_classes=self.num_classes
        )

    @torch.inference_mode()
    def forward(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = self.network(data_batch["data"].cuda())
        return {"logits": logits}


class SigLIPEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        save_path: str = DEFAULT_POSTPROCESSOR_SAVE_PATH,
    ) -> None:
        super().__init__()

        checkpoint_dir = snapshot_download(checkpoint_id, local_dir=save_path)
        checkpoint_dir = (
            pathlib.Path(checkpoint_dir) / "video_content_safety_filter"
        ).as_posix()

        self.checkpoint_dir = checkpoint_dir
        self.model = SiglipModel.from_pretrained(
            model_name, cache_dir=self.checkpoint_dir
        )
        self.processor = SiglipProcessor.from_pretrained(
            model_name, cache_dir=self.checkpoint_dir
        )

    @torch.inference_mode()
    def encode_image(self, input_img: PIL.Image.Image) -> torch.Tensor:
        """Encode an image into a feature vector."""
        with torch.no_grad():
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            inputs = self.processor(images=input_img, return_tensors="pt").to(
                device, dtype=dtype
            )
            image_features = self.model.get_image_features(**inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features


class VideoContentSafetyFilter(torch.nn.Module, ContentSafetyGuardrail):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        save_path: str = DEFAULT_POSTPROCESSOR_SAVE_PATH,
    ) -> None:
        super().__init__()

        checkpoint_dir = snapshot_download(checkpoint_id, local_dir=save_path)
        checkpoint_dir = (
            pathlib.Path(checkpoint_dir) / "video_content_safety_filter"
        ).as_posix()

        self.encoder = SigLIPEncoder(checkpoint_id=checkpoint_id)

        model_config = ModelConfig(input_size=1152, num_classes=7)
        self.model = VideoSafetyModel(model_config)

        safety_filter_local_path = os.path.join(checkpoint_dir, "safety_filter.pt")
        checkpoint = torch.load(safety_filter_local_path, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])

        self.eval()

    @torch.inference_mode()
    def __infer(self, pil_image: PIL.Image.Image) -> int:
        """Infer the class of the image."""
        image_embs = self.encoder.encode_image(pil_image)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        image_embs = image_embs.to(device=device, dtype=dtype)
        logits = self.model.network(image_embs)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        return predicted_class

    def is_safe_frames(self, frames: Iterable) -> bool:
        """Check if the video frames are safe."""
        is_safe = True
        frame_scores = []

        for frame_number, frame in enumerate(frames):
            try:
                pil_image = PIL.Image.fromarray(frame)
                predicted_class = self.__infer(pil_image)
                class_name = CLASS_IDX_TO_NAME.get(predicted_class, "Unknown")
                frame_scores.append({"frame_number": frame_number, "class": class_name})

                # If any frame is not "Safe", mark as not safe
                if predicted_class != 0:
                    is_safe = False
                    break

            except Exception as e:
                logger.warning(
                    f"Warning: Failed to run safety classifier on frame_number {frame_number}. Exception: {e}"
                )
                continue

        video_data = {
            "is_safe": is_safe,
            "frame_scores": frame_scores,
        }

        return is_safe

    def is_safe(self, input: Union[str, Iterable]) -> Tuple[bool, str]:
        if isinstance(input, Iterable):
            is_safe = self.is_safe_frames(input)
            return is_safe, (
                "safe frames detected" if is_safe else "unsafe frames detected"
            )
        else:
            raise ValueError(f"Input type {type(input)} not supported.")


class RetinaFaceFilter(torch.nn.Module, PostprocessingGuardrail):
    def __init__(
        self,
        checkpoint_id: str = COSMOS_GUARDRAIL_CHECKPOINT,
        save_path: str = DEFAULT_POSTPROCESSOR_SAVE_PATH,
        batch_size: int = 1,
        confidence_threshold: float = 0.7,
    ) -> None:
        super().__init__()

        checkpoint_dir = snapshot_download(checkpoint_id, local_dir=save_path)
        checkpoint = (
            pathlib.Path(checkpoint_dir) / "face_blur_filter/Resnet50_Final.pth"
        )

        self.cfg = cfg_re50
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        # Disable loading ResNet pretrained weights
        self.cfg["pretrain"] = False
        self.net = RetinaFace(cfg=self.cfg, phase="test")

        # Load from RetinaFace pretrained checkpoint
        self.net = load_model(self.net, checkpoint)

        self.eval()

    def preprocess_frames(self, frames: np.ndarray) -> torch.Tensor:
        """Preprocess a sequence of frames for face detection.

        Args:
            frames: Input frames

        Returns:
            Preprocessed frames tensor
        """
        device = next(self.net.parameters()).device
        dtype = next(self.net.parameters()).dtype

        with torch.no_grad():
            frames_tensor = torch.from_numpy(frames).to(
                device=device, dtype=dtype
            )  # Shape: [T, H, W, C]
            frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # Shape: [T, C, H, W]
            frames_tensor = frames_tensor[
                :, [2, 1, 0], :, :
            ]  # RGB to BGR to match RetinaFace model input
            means = torch.tensor(
                [104.0, 117.0, 123.0], device=device, dtype=dtype
            ).view(1, 3, 1, 1)
            frames_tensor = (
                frames_tensor - means
            )  # Subtract mean BGR values for each channel
            return frames_tensor

    def blur_detected_faces(
        self,
        frames: np.ndarray,
        batch_loc: torch.Tensor,
        batch_conf: torch.Tensor,
        prior_data: torch.Tensor,
        scale: torch.Tensor,
        min_size: tuple[int] = (20, 20),
    ) -> list[np.ndarray]:
        """Blur detected faces in a batch of frames using RetinaFace predictions.

        Args:
            frames: Input frames
            batch_loc: Batched location predictions
            batch_conf: Batched confidence scores
            prior_data: Prior boxes for the video
            scale: Scale factor for resizing detections
            min_size: Minimum size of a detected face region in pixels

        Returns:
            Processed frames with pixelated faces
        """
        with torch.no_grad():
            batch_boxes = decode_batch(batch_loc, prior_data, self.cfg["variance"])
            batch_boxes = batch_boxes * scale

        blurred_frames = []
        for i, boxes in enumerate(batch_boxes):
            boxes = boxes.detach().cpu().numpy()
            scores = batch_conf[i, :, 1].detach().cpu().numpy()

            filtered_boxes = filter_detected_boxes(
                boxes,
                scores,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=NMS_THRESHOLD,
                top_k=TOP_K,
                keep_top_k=KEEP_TOP_K,
            )

            frame = frames[i]
            for box in filtered_boxes:
                x1, y1, x2, y2 = map(int, box)
                # Ignore bounding boxes smaller than the minimum size
                if x2 - x1 < min_size[0] or y2 - y1 < min_size[1]:
                    continue
                max_h, max_w = frame.shape[:2]
                face_roi = frame[
                    max(y1, 0) : min(y2, max_h), max(x1, 0) : min(x2, max_w)
                ]
                blurred_face = pixelate_face(face_roi)
                frame[max(y1, 0) : min(y2, max_h), max(x1, 0) : min(x2, max_w)] = (
                    blurred_face
                )
            blurred_frames.append(frame)

        return blurred_frames

    def postprocess(self, frames: np.ndarray) -> np.ndarray:
        """Blur faces in a sequence of frames.

        Args:
            frames: Input frames

        Returns:
            Processed frames with pixelated faces
        """
        # Create dataset and dataloader
        frames_tensor = self.preprocess_frames(frames)
        dataset = TensorDataset(frames_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        processed_frames, processed_batches = [], []
        device = next(self.net.parameters()).device
        dtype = next(self.net.parameters()).dtype

        prior_data, scale = None, None
        for i, batch in enumerate(dataloader):
            batch = batch[0]
            h, w = batch.shape[-2:]  # Batch shape: [C, H, W]

            with torch.no_grad():
                # Generate priors for the video
                if prior_data is None:
                    priorbox = PriorBox(self.cfg, image_size=(h, w))
                    priors = priorbox.forward()
                    priors = priors.to(device, dtype=dtype)
                    prior_data = priors.data

                # Get scale for resizing detections
                if scale is None:
                    scale = torch.Tensor([w, h, w, h])
                    scale = scale.to(device, dtype=dtype)

                batch_loc, batch_conf, _ = self.net(batch)

            # Blur detected faces in each batch of frames
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(frames))
            processed_batches.append(
                self.blur_detected_faces(
                    frames[start_idx:end_idx], batch_loc, batch_conf, prior_data, scale
                )
            )

        processed_frames = [frame for batch in processed_batches for frame in batch]
        return np.array(processed_frames)


@postprocessor_registry("cosmos.guardrail")
class CosmosGuardrailPostprocessor(BasePostprocessor):
    def __init__(
        self,
        engine,
        model_path: str = "nvidia/Cosmos-1.0-Guardrail",
        save_path: str = DEFAULT_POSTPROCESSOR_SAVE_PATH,
        **kwargs,
    ):
        super().__init__(engine, PostprocessorCategory.SAFETY_CHECKER, **kwargs)
        self.model_path = model_path
        self.save_path = save_path
        self.runner = GuardrailRunner(
            safety_models=[VideoContentSafetyFilter(save_path=save_path)],
            postprocessors=[RetinaFaceFilter(save_path=save_path)],
        )

    def __call__(self, video: list[PIL.Image.Image]) -> np.ndarray:
        video = self.engine.video_processor.postprocess_video(video, output_type="np")
        video = (video * 255).astype(np.uint8)
        video_batch = []
        for vid in video:
            is_safe, message = self.runner.run_safety_check(vid)
            if not is_safe:
                logger.warning(f"GUARDRAIL BLOCKED: {message}")
                continue
            frames = self.runner.postprocess(vid)
            video_batch.append(frames)
        video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
        video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
        return video
