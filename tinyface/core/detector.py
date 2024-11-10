from typing import List, Optional, Tuple

import numpy
from onnxruntime import InferenceSession

from ..config import global_config
from ..download import download
from ..inference import create_inference_session
from ..typing import BoundingBox, Detection, FaceLandmark5, VisionFrame
from ..utils import (
    apply_nms,
    create_static_anchors,
    distance_to_bounding_box,
    distance_to_face_landmark_5,
    normalize_bounding_box,
    resize_frame_resolution,
    unpack_resolution,
)


class FaceDetector:
    def __init__(self) -> None:
        self._session: Optional[InferenceSession] = None
        self._model_path: Optional[str] = None

    def prepare(self):
        self._model_path = global_config().face_detector_model
        if not self._model_path:
            self._model_path = download(
                url="https://github.com/idootop/TinyFace/releases/download/models-1.0.0/scrfd_2.5g.onnx",
                known_hash="2c07342347cef21a619c49dd5664fb8c09570ae9eda5bff3e385c11eafc45ada",
            )
        if not self._session:
            self._session = create_inference_session(self._model_path)

    def _forward(self, detect_vision_frame: VisionFrame) -> Detection:
        self.prepare()
        assert self._session is not None
        return self._session.run(None, {"input": detect_vision_frame})

    def detect_faces(
        self,
        vision_frame: VisionFrame,
    ) -> Tuple[List[BoundingBox], List[float], List[FaceLandmark5]]:
        bounding_boxes, face_scores, face_landmarks_5 = self._detect_with_scrfd(
            vision_frame
        )
        bounding_boxes = [normalize_bounding_box(bbox) for bbox in bounding_boxes]
        keep_indices = apply_nms(
            bounding_boxes,
            face_scores,
            global_config().face_detector_score,
            nms_threshold=0.1,
        )
        filtered_boxes = [bounding_boxes[i] for i in keep_indices]
        filtered_scores = [face_scores[i] for i in keep_indices]
        filtered_landmarks = [face_landmarks_5[i] for i in keep_indices]
        return filtered_boxes, filtered_scores, filtered_landmarks

    def _detect_with_scrfd(
        self, vision_frame: VisionFrame
    ) -> Tuple[List[BoundingBox], List[float], List[FaceLandmark5]]:
        bounding_boxes = []
        face_scores = []
        face_landmarks_5 = []
        feature_strides = [8, 16, 32]
        feature_map_channel = 3
        anchor_total = 2
        face_detector_width, face_detector_height = unpack_resolution(
            global_config().face_detector_size
        )
        temp_vision_frame = resize_frame_resolution(
            vision_frame, (face_detector_width, face_detector_height)
        )
        ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
        ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
        detect_vision_frame = self._prepare_detect_frame(temp_vision_frame)
        detection = self._forward(detect_vision_frame)

        for index, feature_stride in enumerate(feature_strides):
            keep_indices = numpy.where(
                detection[index] >= global_config().face_detector_score
            )[0]

            if numpy.any(keep_indices):
                stride_height = face_detector_height // feature_stride
                stride_width = face_detector_width // feature_stride
                anchors = create_static_anchors(
                    feature_stride, anchor_total, stride_height, stride_width
                )
                bounding_box_raw = (
                    detection[index + feature_map_channel] * feature_stride
                )
                face_landmark_5_raw = (
                    detection[index + feature_map_channel * 2] * feature_stride
                )

                for bounding_box in distance_to_bounding_box(anchors, bounding_box_raw)[
                    keep_indices
                ]:
                    bounding_boxes.append(
                        numpy.array(
                            [
                                bounding_box[0] * ratio_width,
                                bounding_box[1] * ratio_height,
                                bounding_box[2] * ratio_width,
                                bounding_box[3] * ratio_height,
                            ]
                        )
                    )

                for score in detection[index][keep_indices]:
                    face_scores.append(score[0])

                for face_landmark_5 in distance_to_face_landmark_5(
                    anchors, face_landmark_5_raw
                )[keep_indices]:
                    face_landmarks_5.append(
                        face_landmark_5 * [ratio_width, ratio_height]
                    )

        return bounding_boxes, face_scores, face_landmarks_5

    def _prepare_detect_frame(self, temp_vision_frame: VisionFrame) -> VisionFrame:
        face_detector_width, face_detector_height = unpack_resolution(
            global_config().face_detector_size
        )
        detect_vision_frame = numpy.zeros(
            (face_detector_height, face_detector_width, 3)
        )
        detect_vision_frame[
            : temp_vision_frame.shape[0], : temp_vision_frame.shape[1], :
        ] = temp_vision_frame
        detect_vision_frame = (detect_vision_frame - 127.5) / 128.0
        detect_vision_frame = numpy.expand_dims(
            detect_vision_frame.transpose(2, 0, 1), axis=0
        ).astype(numpy.float32)
        return detect_vision_frame
