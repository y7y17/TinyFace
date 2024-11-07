from collections import namedtuple
from typing import Any, Tuple

import numpy
from numpy.typing import NDArray

Detection = NDArray[Any]
Prediction = NDArray[Any]

VisionFrame = NDArray[Any]
Mask = NDArray[Any]
Points = NDArray[Any]
Distance = NDArray[Any]
Matrix = NDArray[Any]
Anchors = NDArray[Any]

Resolution = Tuple[int, int]
Padding = Tuple[int, int, int, int]

BoundingBox = NDArray[Any]
FaceLandmark5 = NDArray[Any]
FaceLandmark68 = NDArray[Any]
Embedding = NDArray[numpy.float64]
FaceWarpTemplate = NDArray[Any]

Face = namedtuple(
    "Face",
    [
        "bounding_box",
        "score",
        "landmark_5",
        "embedding",
        "normed_embedding",
    ],
)

FacePair = namedtuple(
    "FacePair",
    [
        "reference",
        "destination",
    ],
)
