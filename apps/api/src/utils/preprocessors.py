MODEL_WEIGHTS = {
    "salient": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/salient/u2net.pt",
    "sam": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/sam/sam_vit_b_01ec64.pth",
    "sam2": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/sam2/sam2.1_hiera_large.pt",
    "gdino": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/gdino/groundingdino_swint_ogc.pth",
    "pose_detection": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/pose/yolox_l.onnx",
    "pose": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/pose/dw-ll_ucoco_384.onnx",
    "depth.midas": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/depth/dpt_hybrid-midas-501f0c75.pt",
    "depth.anything_v2": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/depth/depth_anything_v2_vitl.pth",
    "flow": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/flow/raft-things.pth",
    "ram": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/ram/ram_plus_swin_large_14m.pth",
    "scribble": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/scribble/anime_style/netG_A_latest.pth",
}


MODEL_CONFIGS = {
    "sam2": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
    "gdino": "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/gdino/GroundingDINO_SwinT_OGC.py",
}

RAM_TAG_COLOR_PATH = "https://huggingface.co/ali-vilab/VACE-Annotators/resolve/main/layout/ram_tag_color_list.txt"
