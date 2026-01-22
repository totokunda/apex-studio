from functools import partial
from torch.nn import functional as F, init
import florch.layers
import florch.layers.lora
import torch
import torch.nn as nn
import pt.backbones.dinov2.hub.backbones as dinov2_backbones
import pt.backbones.efficientnet as effnet


def build_backbone(config):
    build_fn = get_build_fn(config)
    normalizer = get_normalizer(config)
    backbone, mean, std, out_channels = build_fn(normalizer, config)

    if config["lora_rank"] > 0:
        florch.layers.lora.apply_lora(backbone, lora_r=config["lora_rank"])

    preproc_layer = PreprocLayer(mean, std)
    result = torch.nn.Sequential(preproc_layer, backbone)
    return result, normalizer, out_channels


def get_build_fn(config):
    prefix_to_build_fn = dict(
        efficientnetv2=build_effnetv2,
        resnet=build_resnet,
        mobilenet=build_mobilenet,
        dinov2=build_dinov2,
        # siglip2=build_siglip2,
    )
    for prefix, build_fn in prefix_to_build_fn.items():
        if config["backbone"].startswith(prefix):
            return build_fn

    raise Exception(f'No backbone builder found for {config["backbone"]}.')


def build_resnet(bn):
    raise NotImplementedError()


def build_effnetv2(bn, config):
    effnet_size = config["backbone"].rpartition("-")[2]
    weights = getattr(effnet, f"EfficientNet_V2_{effnet_size.upper()}_Weights").DEFAULT
    backbone_raw = getattr(effnet, f"efficientnet_v2_{effnet_size}")(
        config=config, norm_layer=bn, weights=weights
    )
    backbone = backbone_raw.features
    return backbone, 0.5, 0.5, 1280


def build_mobilenet(bn):
    raise NotImplementedError()


def get_normalizer(config):
    if config["ghost_bn"]:
        raise NotImplementedError()
    elif config["batch_renorm"]:
        bn = partial(florch.layers.BatchRenorm2d, eps=1e-3)
    elif config["group_norm"]:
        bn = partial(GroupNormSameDtype, 32, eps=1e-3)
    elif config["transition_bn"]:
        bn = partial(florch.layers.TransitionBatchNorm2d, 32, eps=1e-3)
    else:
        bn = partial(nn.BatchNorm2d, eps=1e-3)

    if not config["group_norm"]:

        def result(*args, momentum=None, **kwargs):
            if momentum is None:
                momentum = 0.01

            momentum = 1.0 - (1.0 - momentum) ** (1.0 / config["grad_accum_steps"])
            return bn(*args, momentum=momentum, **kwargs)

    else:
        result = bn

    return result


class DinoV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = getattr(dinov2_backbones, config["backbone"].replace("-", "_"))()
        self.num_features = self.model.num_features
        self.feat_side = config["proc_side"] // self.model.patch_size

    def forward(self, x):
        raw_feat: torch.Tensor = self.model.forward_features(x)["x_norm_patchtokens"]
        return (
            raw_feat.unflatten(1, (self.feat_side, self.feat_side))
            .permute(0, 3, 1, 2)
            .to(x.dtype)
        )


#
# class SigLIP2(nn.Module):
#     def __init__(self):
#         super().__init__()
#         from transformers import AutoModel, AutoImageProcessor
#
#         ckpt = "google/siglip2-base-patch32-256"
#         processor = AutoImageProcessor.from_pretrained(ckpt)
#         self.image_mean = processor.image_mean
#         self.image_std = processor.image_std
#         self.model = AutoModel.from_pretrained(ckpt)
#         self.model.vision_model.requires_grad_(True)
#         del self.model.text_model
#         torch.cuda.empty_cache()
#         self.num_features = 768
#         self.feat_side = FLAGS.proc_side // 32
#
#     def forward(self, x):
#         return (
#             self.model.vision_model(x)['last_hidden_state']
#             .unflatten(1, (self.feat_side, self.feat_side))
#             .permute(0, 3, 1, 2)
#             .to(x.dtype)
#         )
#
#
# def build_siglip2(bn):
#     bbone = SigLIP2()
#     return (
#         bbone,
#         bbone.image_mean,
#         bbone.image_std,
#         bbone.num_features,
#     )


def build_dinov2(bn, config):
    bbone = DinoV2(config)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return bbone, mean, std, bbone.num_features


def torch_preproc(x, mean_rgb=(0.485, 0.456, 0.406), stdev_rgb=(0.229, 0.224, 0.225)):
    mean_rgb = torch.tensor(mean_rgb, dtype=x.dtype, device=x.device)[
        ..., torch.newaxis, torch.newaxis
    ]
    stdev_rgb = torch.tensor(stdev_rgb, dtype=x.dtype, device=x.device)[
        ..., torch.newaxis, torch.newaxis
    ]
    return (x - mean_rgb) / stdev_rgb


def caffe_preproc(x):
    mean_rgb = torch.tensor([103.939, 116.779, 123.68], dtype=x.dtype, device=x.device)[
        ..., torch.newaxis, torch.newaxis
    ]
    _255 = torch.tensor(255, dtype=x.dtype, device=x.device)
    return _255 * x - mean_rgb


def tf_preproc(x):
    _2 = torch.tensor(2, dtype=x.dtype, device=x.device)
    _1 = torch.tensor(1, dtype=x.dtype, device=x.device)
    return _2 * x - _1


def mobilenet_preproc(x):
    _255 = torch.tensor(255, dtype=x.dtype, device=x.device)
    return _255 * x


# class GroupNormSameDtype(nn.GroupNorm):
#     def forward(self, x):
#         return super().forward(x).to(x.dtype)


class GroupNormSameDtype(nn.Module):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps
        ).to(input.dtype)

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )


class PreprocLayer(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        # These are part of the backbone checkpoint (e.g. "backbone.0.mean/std")
        # and must be persistent so strict state_dict loading works.
        self.mean = nn.Buffer(to_tensor(mean), persistent=True)
        self.std = nn.Buffer(to_tensor(std), persistent=True)

    def forward(self, inp):
        return (inp - self.mean.to(inp.dtype)) / self.std.to(inp.dtype)


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    elif isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.float32)[..., torch.newaxis, torch.newaxis]
    else:
        return torch.tensor(x, dtype=torch.float32)
