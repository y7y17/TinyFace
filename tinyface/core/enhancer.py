from typing import Optional

import cv2
import numpy
from onnxruntime import InferenceSession

from ..config import global_config
from ..download import download
from ..inference import create_inference_session
from ..typing import Detection, Face, VisionFrame
from ..utils import create_static_box_mask, paste_back, warp_face_by_face_landmark_5


class FaceEnhancer:
    def __init__(self) -> None:
        self._session: Optional[InferenceSession] = None
        self._model_path: Optional[str] = None
        self._model_size = (512, 512)
        self._model_warp_template = numpy.array(
            [
                [0.37691676, 0.46864664],
                [0.62285697, 0.46912813],
                [0.50123859, 0.61331904],
                [0.39308822, 0.72541100],
                [0.61150205, 0.72490465],
            ]
        )

    def prepare(self):
        self._model_path = global_config().face_enhancer_model
        if not self._model_path:
            self._model_path = download(
                url="https://github.com/idootop/TinyFace/releases/download/models-1.0.0/gfpgan_1.4.onnx",
                known_hash="accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385",
            )
        if not self._session:
            self._session = create_inference_session(self._model_path)

    def _forward(self, crop_vision_frame: VisionFrame) -> Detection:
        self.prepare()
        assert self._session is not None

        face_enhancer_inputs = {}
        for face_enhancer_input in self._session.get_inputs():
            if face_enhancer_input.name == "input":
                face_enhancer_inputs[face_enhancer_input.name] = crop_vision_frame
            if face_enhancer_input.name == "weight":
                weight = numpy.array([1]).astype(numpy.double)
                face_enhancer_inputs[face_enhancer_input.name] = weight

        return self._session.run(None, face_enhancer_inputs)[0][0]

    def enhance_face(
        self, temp_vision_frame: VisionFrame, target_face: Face
    ) -> VisionFrame:
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
            temp_vision_frame,
            target_face.landmark_5,
            self._model_warp_template,
            self._model_size,
        )
        box_mask = create_static_box_mask(
            crop_vision_frame.shape[:2][::-1],
            global_config().face_mask_blur,
            global_config().face_mask_padding,
        )
        crop_vision_frame = self._prepare_crop_frame(crop_vision_frame)
        crop_vision_frame = self._forward(crop_vision_frame)
        crop_vision_frame = self._normalize_crop_frame(crop_vision_frame)
        crop_mask = box_mask.clip(0, 1)
        paste_vision_frame = paste_back(
            temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix
        )
        temp_vision_frame = self._blend_frame(temp_vision_frame, paste_vision_frame)
        return temp_vision_frame

    def _prepare_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
        crop_vision_frame = numpy.expand_dims(
            crop_vision_frame.transpose(2, 0, 1), axis=0
        ).astype(numpy.float32)
        return crop_vision_frame

    def _normalize_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        crop_vision_frame = numpy.clip(crop_vision_frame, -1, 1)
        crop_vision_frame = (crop_vision_frame + 1) / 2
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        crop_vision_frame = (crop_vision_frame * 255.0).round()
        crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
        return crop_vision_frame

    def _blend_frame(
        self, temp_vision_frame: VisionFrame, paste_vision_frame: VisionFrame
    ) -> VisionFrame:
        face_face_enhancer_blend = 1 - (global_config().face_enhancer_blend / 100)
        temp_vision_frame = cv2.addWeighted(
            temp_vision_frame,
            face_face_enhancer_blend,
            paste_vision_frame,
            1 - face_face_enhancer_blend,
            0,
        )
        return temp_vision_frame
