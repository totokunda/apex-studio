from abc import ABC, abstractmethod
from pathlib import Path
import yaml
from src.utils.yaml import LoaderWithInclude
from src.manifest.loader import validate_and_normalize
from src.mixins.download_mixin import DownloadMixin
from loguru import logger
import os
import shutil
import subprocess
from src.quantize.quants import QuantType


class BaseQuantizer(ABC, DownloadMixin):
    @abstractmethod
    def quantize(self, output_path: str = None, **kwargs):
        pass

    def _load_yaml(self, file_path: str | Path):
        file_path = Path(file_path)
        text = file_path.read_text()

        # --- PASS 1: extract `shared:` (legacy) or `spec.shared` (v1) ---
        prelim = yaml.load(text, Loader=yaml.FullLoader)
        shared_entries = []
        if isinstance(prelim, dict):
            shared_entries.extend(prelim.get("shared", []) or [])
            spec = prelim.get("spec", {}) or {}
            if isinstance(spec, dict):
                shared_entries.extend(spec.get("shared", []) or [])

        # build alias → manifest Path
        shared_manifests = {}
        for entry in shared_entries:
            p = (file_path.parent / entry).resolve()
            try:
                alias = p.stem.split("_", 1)[1]
            except Exception:
                alias = p.stem
            shared_manifests[alias] = p

        # attach it to our custom loader
        LoaderWithInclude.shared_manifests = shared_manifests

        # --- PASS 2: real load with !include expansion ---
        loaded = yaml.load(text, Loader=LoaderWithInclude)

        # Validate and normalize if this is a v1 manifest
        try:
            loaded = validate_and_normalize(loaded)
        except Exception:
            raise

        return loaded

    def _fix_output_path(self, output_path: str, quantization_str: str):
        file_ending = f".gguf"
        if output_path.endswith(file_ending) and file_ending.endswith(
            f".{quantization_str}.gguf"
        ):
            return output_path
        elif output_path.endswith(file_ending):
            return output_path.replace(file_ending, f".{quantization_str}.gguf")
        else:
            return output_path + file_ending

    def _llama_cpp_quant(
        self,
        fp16_quant_path: str,
        output_path: str,
        quantization: QuantType = QuantType.F16,
        llama_quantize_path: str = "llama-quantize",
    ):
        logger.info(f"Quantizing model with quantization type {quantization}")

        # Resolve llama-quantize binary location with robust fallbacks:
        # 1) explicit arg if it exists
        # 2) env var APEX_LLAMA_QUANTIZE_BIN
        # 3) submodule build path: thirdparty/llama.cpp/build/bin/llama-quantize
        # 4) repo bin path: bin/llama-quantize
        # 5) PATH lookup
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates: list[str] = []

        if llama_quantize_path and os.path.isabs(llama_quantize_path):
            candidates.append(llama_quantize_path)

        env_bin = os.getenv("APEX_LLAMA_QUANTIZE_BIN")
        if env_bin:
            candidates.append(env_bin)

        submodule_bin = os.path.join(
            repo_root, "thirdparty", "llama.cpp", "build", "bin", "llama-quantize"
        )
        candidates.append(submodule_bin)

        repo_bin = os.path.join(repo_root, "bin", "llama-quantize")
        candidates.append(repo_bin)

        path_bin = shutil.which("llama-quantize")
        if path_bin:
            candidates.append(path_bin)

        resolved = next((p for p in candidates if p and os.path.exists(p)), None)
        if not resolved:
            search_list = "\n".join(candidates)
            raise FileNotFoundError(
                "Could not locate 'llama-quantize'. Tried:\n"
                + search_list
                + "\nBuild the submodule and/or set APEX_LLAMA_QUANTIZE_BIN to the binary."
            )

        quantization_str = quantization.value
        output_path = self._fix_output_path(output_path, quantization_str)

        cmd = [resolved, fp16_quant_path, output_path, quantization_str]
        logger.info(f"Running: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if proc.stdout:
                logger.debug(proc.stdout)
            if proc.stderr:
                logger.debug(proc.stderr)
        except subprocess.CalledProcessError as exc:
            logger.error(exc.stdout or "")
            logger.error(exc.stderr or "")
            raise RuntimeError(
                f"Failed to quantize model with exit code {exc.returncode}. Command: {' '.join(cmd)}"
            ) from exc

        return output_path

    def _requires_llama_cpp_quant(self, quantization: QuantType):
        print(f"quantization: {quantization}")
        return quantization not in [QuantType.F16, QuantType.F32]
