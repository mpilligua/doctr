# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from typing import Any, List

from doctr.file_utils import is_tf_available, is_torch_available

from .. import detection
from ..detection.fast import reparameterize
from ..preprocessor import PreProcessor
from ..denoiser import Denoiser

__all__ = ["denoiser_predictor"]

ARCHS: List[str]


if is_tf_available():
    ARCHS = [
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
        "linknet_resnet34",
        "linknet_resnet50",
        "fast_tiny",
        "fast_small",
        "fast_base",
    ]
elif is_torch_available():
    ARCHS = [
        "db_resnet34",
        "db_resnet50",
        "db_mobilenet_v3_large",
        "linknet_resnet18",
        "linknet_resnet34",
        "linknet_resnet50",
        "fast_tiny",
        "fast_small",
        "fast_base",
        "mirnet"
    ]


def _predictor(arch: Any, pretrained: bool, assume_straight_pages: bool = True, **kwargs: Any) -> DenoiserPredictor:
    if isinstance(arch, str):
        if arch not in ARCHS:
            raise ValueError(f"unknown architecture '{arch}'")

        _model = detection.__dict__[arch](
            pretrained=pretrained,
            pretrained_backbone=kwargs.get("pretrained_backbone", True),
            assume_straight_pages=assume_straight_pages,
        )
        # Reparameterize FAST models by default to lower inference latency and memory usage
        if isinstance(_model, detection.FAST):
            _model = reparameterize(_model)
    else:
        if not isinstance(arch, (denoiser.MIRNet)):
            raise ValueError(f"unknown architecture: {type(arch)}")

        _model = arch
        _model.assume_straight_pages = assume_straight_pages

    kwargs.pop("pretrained_backbone", None)

    kwargs["mean"] = kwargs.get("mean", _model.cfg["mean"])
    kwargs["std"] = kwargs.get("std", _model.cfg["std"])
    kwargs["batch_size"] = kwargs.get("batch_size", 2)
    predictor = DenoiserPredictor(
        PreProcessor(_model.cfg["input_shape"][:-1] if is_tf_available() else _model.cfg["input_shape"][1:], **kwargs),
        _model,
    )
    return predictor


def denoiser_predictor(
    arch: Any = "mirnet",
    pretrained: bool = False,
    assume_straight_pages: bool = True,
    **kwargs: Any,
) -> DenoiserPredictor:
    """
    Args:
    ----
        arch: name of the architecture or model itself to use (e.g. 'db_resnet50')
        pretrained: If True, returns a model pre-trained on our text detection dataset
        assume_straight_pages: If True, fit straight boxes to the page
        **kwargs: optional keyword arguments passed to the architecture

    Returns:
    -------
        Detection predictor
    """
    return _predictor(arch, pretrained, assume_straight_pages, **kwargs)
