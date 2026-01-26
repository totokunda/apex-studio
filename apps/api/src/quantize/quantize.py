from src.quantize.text_encoder import (
    get_model_architecture,
    ModelBase,
    ModelType,
)
from src.quantize.base import BaseQuantizer
from tempfile import NamedTemporaryFile
from src.utils.defaults import DEFAULT_COMPONENTS_PATH
from pathlib import Path
import torch
from typing import List
from src.quantize.transformer import (
    ModelArchitecture as TransformerModelArchitecture,
    convert_model as convert_transformer_model,
)

from src.quantize.quants import QuantType, qconfig_map


class TextEncoderQuantizer(BaseQuantizer):
    model_path: str = None
    tokenizer_path: str = None
    model_type: ModelType = None
    kwargs: dict = {}
    quantization: QuantType = QuantType.F16

    def __init__(
        self,
        output_path: str,
        model_path: str = None,
        tokenizer_path: str = None,
        model_type: ModelType = ModelType.TEXT,
        quantization: QuantType | str = QuantType.F16,
        **kwargs,
    ):
        self.output_path = output_path
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        self.model_type = model_type
        self.kwargs = kwargs

        if isinstance(quantization, str):
            quantization = QuantType(quantization)

        self.quantization = quantization

        self.model_path = self._download(self.model_path, DEFAULT_COMPONENTS_PATH)
        self.tokenizer_path = self._download(
            self.tokenizer_path, DEFAULT_COMPONENTS_PATH
        )

    @torch.inference_mode()
    def quantize(
        self,
        output_path: str = None,
        quantization: QuantType | str = None,
        split_max_tensors: int = 0,
        split_max_size: int = 0,
        dry_run: bool = False,
        small_first_shard: bool = False,
        remote_hf_model_id: str = None,
        hf_repo_id: str = None,
        use_temp_file: bool = False,
        bigendian: bool = False,
        llama_quantize_path: str = "llama-quantize",
        **kwargs,
    ):
        if isinstance(quantization, str):
            quantization = QuantType(quantization)

        if quantization is None:
            quantization = self.quantization

        if output_path is None:
            output_path = self.output_path

        requires_llama_cpp_quant = self._requires_llama_cpp_quant(quantization)

        hparams = ModelBase.load_hparams(Path(self.model_path), False)
        model_architecture = get_model_architecture(hparams, self.model_type)

        model_class = ModelBase.from_model_architecture(
            model_architecture, model_type=self.model_type
        )

        if requires_llama_cpp_quant:
            with NamedTemporaryFile(delete=True) as temp_file:
                quant_path = temp_file.name
        else:
            temp_file = None
            quant_path = self._fix_output_path(output_path, quantization.value)

        model_instance = model_class(
            Path(self.model_path),
            Path(self.tokenizer_path) if self.tokenizer_path is not None else None,
            qconfig_map["F16"].ftype,
            Path(quant_path),
            is_big_endian=bigendian,
            use_temp_file=use_temp_file,
            eager=self.kwargs.get("no_lazy", False),
            metadata_override=self.kwargs.get("metadata", None),
            model_name=self.kwargs.get("model_name", None),
            split_max_tensors=split_max_tensors,
            split_max_size=split_max_size,
            dry_run=dry_run,
            small_first_shard=small_first_shard,
            remote_hf_model_id=hf_repo_id,
        )

        model_instance.write()

        if requires_llama_cpp_quant:
            save_path = self._llama_cpp_quant(
                quant_path, output_path, quantization, llama_quantize_path
            )
        else:
            save_path = quant_path

        if temp_file is not None:
            temp_file.close()

        return save_path


class TransformerQuantizer(BaseQuantizer):
    model_path: str = None
    tokenizer_path: str = None
    model_type: ModelType = None
    kwargs: dict = {}
    quantization: QuantType = QuantType.F16

    def __init__(
        self,
        output_path: str,
        model_path: str = None,
        architecture: (
            str | TransformerModelArchitecture
        ) = TransformerModelArchitecture.WAN,
        quantization: QuantType | str = QuantType.F16,
        **kwargs,
    ):
        self.output_path = output_path
        self.model_path = model_path
        self.kwargs = kwargs

        if isinstance(quantization, str):
            quantization = QuantType(quantization)

        if isinstance(architecture, str):
            architecture = TransformerModelArchitecture(architecture)

        if architecture not in TransformerModelArchitecture:
            raise ValueError(f"Invalid architecture: {architecture}")

        self.architecture = architecture
        self.quantization = quantization
        self.model_path = self._download(self.model_path, DEFAULT_COMPONENTS_PATH)

    @torch.inference_mode()
    def quantize(
        self,
        model_path: str = None,
        output_path: str = None,
        quantization: QuantType | str = None,
        architecture: str | TransformerModelArchitecture = None,
        split_max_tensors: int = 0,
        split_max_size: int = 0,
        dry_run: bool = False,
        small_first_shard: bool = False,
        bigendian: bool = False,
        keys_to_exclude: List[str] = None,
        num_workers: int = None,
    ):

        if model_path is None:
            model_path = self.model_path

        if quantization is None:
            quantization = self.quantization

        if output_path is None:
            output_path = self.output_path

        if architecture is None:
            architecture = self.architecture

        if isinstance(architecture, str):
            architecture = TransformerModelArchitecture(architecture)

        quant_path = self._fix_output_path(output_path, quantization.value)

        convert_transformer_model(
            model_path=model_path,
            output_path=quant_path,
            model_architecture=architecture,
            split_max_tensors=split_max_tensors,
            split_max_size=split_max_size,
            dry_run=dry_run,
            small_first_shard=small_first_shard,
            bigendian=bigendian,
            qconfig=qconfig_map[quantization.value],
            keys_to_exclude=keys_to_exclude,
            num_workers=num_workers,
        )

        return quant_path
