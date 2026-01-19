import sys
import os
from pathlib import Path

try:
    import diffusers

    print(f"Diffusers location: {diffusers.__file__}")
    peft_path = Path(diffusers.__file__).parent / "loaders" / "peft.py"
    print(f"PEFT path: {peft_path}")
    if peft_path.exists():
        content = peft_path.read_text()
        print(f"PEFT content length: {len(content)}")
        if "except KeyError" in content:
            print("PEFT: 'except KeyError' FOUND (Likely Patched - Fallback Active)")
        else:
            print(
                "PEFT: 'except KeyError' NOT FOUND (Likely Unpatched/Missing Fallback)"
            )

        if "_SET_ADAPTER_SCALE_FN_MAPPING[self.__class__.__name__]" in content:
            print("PEFT: Assignment line FOUND")
        else:
            print("PEFT: Assignment line NOT FOUND")
    else:
        print("PEFT file does not exist")
except ImportError:
    print("Diffusers not installed")

print("-" * 20)

try:
    import xformers
    import importlib.util

    print(f"Xformers location: {xformers.__file__}")
    spec = importlib.util.find_spec("xformers.ops.fmha.flash3")
    if spec and spec.origin:
        flash3_path = Path(spec.origin)
        print(f"Flash3 path: {flash3_path}")
        if flash3_path.exists():
            content = flash3_path.read_text()
            if "FLASH3_HAS_PAGED_ATTENTION = True" in content:
                print("Flash3: Patched assignment (FLASH3_HAS_PAGED_ATTENTION) FOUND")
            elif "_C_flashattention3 = torch.ops.flash_attn_3" in content:
                print("Flash3: Old assignment found, but missing new patch markers")
            else:
                print("Flash3: Patched assignment NOT FOUND")
    else:
        print("Flash3 module not found")
except ImportError:
    print("Xformers not installed")
