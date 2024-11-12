from functools import lru_cache
from typing import List, Sequence, Tuple

import cv2
import numpy
from cv2.typing import Size

from .typing import (
    Anchors,
    BoundingBox,
    Distance,
    FaceLandmark5,
    FaceWarpTemplate,
    Mask,
    Matrix,
    Padding,
    Points,
    Resolution,
    VisionFrame,
)


def unpack_resolution(resolution: str) -> Resolution:
    width, height = map(int, resolution.split("x"))
    return width, height


def resize_frame_resolution(
    vision_frame: VisionFrame, max_resolution: Resolution
) -> VisionFrame:
    height, width = vision_frame.shape[:2]
    max_width, max_height = max_resolution

    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(vision_frame, (new_width, new_height))
    return vision_frame


def apply_nms(
    bounding_boxes: List[BoundingBox],
    face_scores: List[float],
    score_threshold: float,
    nms_threshold: float,
) -> Sequence[int]:
    normed_bounding_boxes = [
        (x1, y1, x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bounding_boxes
    ]
    keep_indices = cv2.dnn.NMSBoxes(
        normed_bounding_boxes,
        face_scores,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )
    return keep_indices


@lru_cache(maxsize=None)
def create_static_box_mask(
    crop_size: Size, face_mask_blur: float, face_mask_padding: Padding
) -> Mask:
    blur_amount = int(crop_size[0] * 0.5 * face_mask_blur)
    blur_area = max(blur_amount // 2, 1)
    box_mask: Mask = numpy.ones(crop_size).astype(numpy.float32)
    box_mask[: max(blur_area, int(crop_size[1] * face_mask_padding[0] / 100)), :] = 0
    box_mask[-max(blur_area, int(crop_size[1] * face_mask_padding[2] / 100)) :, :] = 0
    box_mask[:, : max(blur_area, int(crop_size[0] * face_mask_padding[3] / 100))] = 0
    box_mask[:, -max(blur_area, int(crop_size[0] * face_mask_padding[1] / 100)) :] = 0
    if blur_amount > 0:
        box_mask = cv2.GaussianBlur(box_mask, (0, 0), blur_amount * 0.25)
    return box_mask


def estimate_matrix_by_face_landmark_5(
    face_landmark_5: FaceLandmark5, warp_template: FaceWarpTemplate, crop_size: Size
) -> Matrix:
    normed_warp_template = warp_template * crop_size
    affine_matrix = cv2.estimateAffinePartial2D(
        face_landmark_5,
        normed_warp_template,
        method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )[0]
    return affine_matrix


def warp_face_by_face_landmark_5(
    temp_vision_frame: VisionFrame,
    face_landmark_5: FaceLandmark5,
    warp_template: FaceWarpTemplate,
    crop_size: Size,
) -> Tuple[VisionFrame, Matrix]:
    affine_matrix = estimate_matrix_by_face_landmark_5(
        face_landmark_5, warp_template, crop_size
    )
    crop_vision_frame = cv2.warpAffine(
        temp_vision_frame,
        affine_matrix,
        crop_size,
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_AREA,
    )
    return crop_vision_frame, affine_matrix


def paste_back(
    temp_vision_frame: VisionFrame,
    crop_vision_frame: VisionFrame,
    crop_mask: Mask,
    affine_matrix: Matrix,
) -> VisionFrame:
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    temp_size = temp_vision_frame.shape[:2][::-1]
    inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
    inverse_vision_frame = cv2.warpAffine(
        crop_vision_frame, inverse_matrix, temp_size, borderMode=cv2.BORDER_REPLICATE
    )
    paste_vision_frame = temp_vision_frame.copy()
    paste_vision_frame[:, :, 0] = (
        inverse_mask * inverse_vision_frame[:, :, 0]
        + (1 - inverse_mask) * temp_vision_frame[:, :, 0]
    )
    paste_vision_frame[:, :, 1] = (
        inverse_mask * inverse_vision_frame[:, :, 1]
        + (1 - inverse_mask) * temp_vision_frame[:, :, 1]
    )
    paste_vision_frame[:, :, 2] = (
        inverse_mask * inverse_vision_frame[:, :, 2]
        + (1 - inverse_mask) * temp_vision_frame[:, :, 2]
    )
    return paste_vision_frame


def normalize_bounding_box(bounding_box: BoundingBox) -> BoundingBox:
    x1, y1, x2, y2 = bounding_box
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return numpy.array([x1, y1, x2, y2])


def distance_to_bounding_box(points: Points, distance: Distance) -> BoundingBox:
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    bounding_box = numpy.column_stack([x1, y1, x2, y2])
    return bounding_box


def distance_to_face_landmark_5(points: Points, distance: Distance) -> FaceLandmark5:
    x = points[:, 0::2] + distance[:, 0::2]
    y = points[:, 1::2] + distance[:, 1::2]
    face_landmark_5 = numpy.stack((x, y), axis=-1)
    return face_landmark_5


@lru_cache(maxsize=None)
def create_static_anchors(
    feature_stride: int, anchor_total: int, stride_height: int, stride_width: int
) -> Anchors:
    y, x = numpy.mgrid[:stride_height, :stride_width][::-1]
    anchors = numpy.stack((y, x), axis=-1)
    anchors = (anchors * feature_stride).reshape((-1, 2))
    anchors = numpy.stack([anchors] * anchor_total, axis=1).reshape((-1, 2))
    return anchors
