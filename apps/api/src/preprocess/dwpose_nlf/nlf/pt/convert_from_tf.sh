#!/usr/bin/env bash

set -euo pipefail

# Save the original default ImageNet-1K-pretrained EfficientNetV2 weights
# In PyTorch, EffNetV2-S uses entirely different weights from the TensorFlow ones (trained with a different recipe)
# while EffNetV2-L uses the ImageNet-21K+ImageNet-1K-finetune weights
# We originally used the original TensorFlow-based, ImageNet-1K-pretrained weights for both models
python -m nlf.pt.convert_ckpt_from_tf --output-model-path=models/effnetv2_s_default_init.pt --config-file=nlf/pt/config/nlf_s.args --from-default-pretrained --save-backbone-only --no-batch-renorm
python -m nlf.pt.convert_ckpt_from_tf --output-model-path=models/effnetv2_l_default_init.pt --config-file=nlf/pt/config/nlf_l.args --from-default-pretrained --save-backbone-only --no-batch-renorm

# Next are the models from the MeTRAbs-ACAE paper from WACV 2023 (https://arxiv.org/abs/2212.14474). These were the starting points for NLF training.
python -m nlf.pt.convert_ckpt_from_tf --input-model-path=wacv23_models/effv2s_bn16ss_huge8_3e-4_sep_1gpu_simplerand_1600k_finetune_48lchirnozero_regul_duallr_aegt_16000k/model_fp32 --output-model-path=models/effnetv2_s_init.pt --config-file=nlf/pt/config/nlf_s.args --from-saved-model --save-backbone-only
python -m nlf.pt.convert_ckpt_from_tf --input-model-path=wacv23_models/effv2l_bn16ss_huge8_3e-4_sep_3gpu_simplerand_384_cont800k_finetune_48lchirnozero_regul_duallr_aegt_ga/model_fp32_ --output-model-path=models/effnetv2_l_init.pt --config-file=nlf/pt/config/nlf_l.args --from-saved-model --save-backbone-only

# Now convert the NLF models from TF to PyTorch
python -m nlf.pt.convert_ckpt_from_tf --input-model-path=models/nlf_l_crop_tf --output-model-path=models/nlf_l_crop.pt --config-file=nlf/pt/config/nlf_l.args
python -m nlf.pt.convert_ckpt_from_tf --input-model-path=models/nlf_s_crop_tf --output-model-path=models/nlf_s_crop.pt --config-file=nlf/pt/config/nlf_s.args

# Now generate multiperson versions
python -m nlf.pt.multiperson.save_model --input-model-path=models/nlf_l_crop.pt --output-model-path=models/nlf_l_multi.torchscript --config-file=nlf/pt/config/nlf_l.args
python -m nlf.pt.multiperson.save_model --input-model-path=models/nlf_s_crop.pt --output-model-path=models/nlf_s_multi.torchscript --config-file=nlf/pt/config/nlf_s.args