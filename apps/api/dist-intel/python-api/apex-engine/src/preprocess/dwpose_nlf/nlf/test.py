import pt.backbones.builder as backbone_builder
import pt.models.field as pt_field
import pt.models.nlf_model as pt_nlf_model
from pt.multiperson import multiperson_model, person_detector
from safetensors.torch import load_file
import yaml
import torch
from PIL import Image
import numpy as np

with open("model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

backbone, normalizer, out_channels = backbone_builder.build_backbone(config)
weight_field = pt_field.build_field(config)
model_pytorch = pt_nlf_model.NLFModel(
    config, backbone, weight_field, normalizer, out_channels
)
onnx_path = "/home/tosin_coverquick_co/apex/ts/yolov8x.onnx"
detector = person_detector.PersonDetector(onnx_path)

multimodel = multiperson_model.MultipersonNLF(
    model_pytorch,
    detector,
    pad_white_pixels=False,
    smpl_config=config["smpl_config"],
    smplx_config=config["smplx_config"],
    fitter_smpl_config=config["fitter_smpl_config"],
    fitter_smplx_config=config["fitter_smplx_config"],
)

model_path = "/home/tosin_coverquick_co/apex/ts/nlf_l_multi_0.3.2_no_detector_cano_all.safetensors"
model_state_dict = load_file(model_path)
# drop all keys
multimodel.load_state_dict(model_state_dict)
multimodel = multimodel.cuda()
multimodel.crop_model.backbone.half()
multimodel.crop_model.heatmap_head.layer.half()

jit_model = torch.jit.load(
    "/home/tosin_coverquick_co/apex/ts/nlf_l_multi_0.3.2.torchscript"
)

image_path = "/home/tosin_coverquick_co/apex/assets/demo_images/chroma.jpg"
image = Image.open(image_path)
image = image.convert("RGB")
image = image.resize((512, 512))
image = np.array(image)
image = image.transpose(2, 0, 1)
image = image.astype(np.float32)
image = image / 255.0
image = torch.tensor(image)
image = image.unsqueeze(0)
image = image.cuda()

with torch.inference_mode():
    pred_jit = jit_model.detect_smpl_batched(image)

with torch.inference_mode():
    pred_multimodel = multimodel.detect_smpl_batched(image)

print(pred_multimodel)
print(pred_jit)
