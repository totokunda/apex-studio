import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"
