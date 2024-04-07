# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

# Credits: post-processing adapted from https://github.com/xuannianz/DifferentiableBinarization

from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from doctr.models.core import BaseModel

__all__ = ["mirnet"]


class mirnet(BaseModel):
