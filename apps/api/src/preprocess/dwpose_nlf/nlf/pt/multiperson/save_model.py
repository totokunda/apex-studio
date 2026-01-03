import argparse

import simplepyutils as spu
import torch
from simplepyutils import FLAGS

import nlf.pt.backbones.builder as backbone_builder
import nlf.pt.init as init
import nlf.pt.models.field as pt_field
import nlf.pt.models.nlf_model as pt_nlf_model
from nlf.paths import DATA_ROOT
from nlf.pt.multiperson import multiperson_model, person_detector
import simplepyutils.argparse as spu_argparse
import florch.layers.lora


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--input-model-path', type=str)
    parser.add_argument('--output-model-path', type=str)
    parser.add_argument('--pad-white-pixels', action=spu_argparse.BoolAction)
    init.initialize(parent_parser=parser)

    backbone, normalizer, out_channels = backbone_builder.build_backbone()
    weight_field = pt_field.build_field()
    model_pytorch = pt_nlf_model.NLFModel(backbone, weight_field, normalizer, out_channels)
    state_dict = torch.load(FLAGS.input_model_path, weights_only=False)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    missing, unexpected = model_pytorch.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        raise RuntimeError(f"Missing keys: {missing}")

    if FLAGS.lora_rank > 0:
        florch.layers.lora.remove_lora(model_pytorch.backbone, merge=True)

    model_pytorch = model_pytorch.cuda().eval()
    model_pytorch.backbone.half()
    model_pytorch.heatmap_head.layer.half()

    detector = person_detector.PersonDetector(f'{DATA_ROOT}/yolov8x.torchscript')

    skeleton_infos = spu.load_pickle(f"{DATA_ROOT}/skeleton_conversion/skeleton_types_huge8.pkl")
    multimodel = multiperson_model.MultipersonNLF(
        model_pytorch, detector, skeleton_infos, pad_white_pixels=FLAGS.pad_white_pixels
    )
    multimodel = torch.jit.script(multimodel.cuda().eval())
    torch.jit.save(multimodel, FLAGS.output_model_path)


if __name__ == '__main__':
    main()
