from typing import Any, Optional

import numpy
from onnxruntime import InferenceSession

from ..config import global_config
from ..download import download
from ..inference import create_inference_session, load_model
from ..typing import Embedding, Face, VisionFrame
from ..utils import create_static_box_mask, paste_back, warp_face_by_face_landmark_5


class FaceSwapper:
    def __init__(self) -> None:
        self._model: Any = None
        self._session: Optional[InferenceSession] = None
        self.model_path = global_config().face_detector_model
        self._model_size = (128, 128)
        self._model_mean = [0.0, 0.0, 0.0]
        self._model_standard_deviation = [1.0, 1.0, 1.0]
        self._model_warp_template = numpy.array(
            [
                [0.36167656, 0.40387734],
                [0.63696719, 0.40235469],
                [0.50019687, 0.56044219],
                [0.38710391, 0.72160547],
                [0.61507734, 0.72034453],
            ]
        )

    def prepare(self):
        if not self.model_path:
            self.model_path = download(
                url="https://github.com/idootop/TinyFace/releases/download/models-1.0.0/inswapper_128_fp16.onnx",
                known_hash="c4eccca86ad177586c85c28bf1a64a9d9ed237e283a15818d831f7facfd3f420",
            )
        if not self._session:
            self._session = create_inference_session(self.model_path)
            self._model = load_model(self.model_path)

    def _forward(
        self, crop_vision_frame: VisionFrame, source_face: Face
    ) -> VisionFrame:
        self.prepare()
        assert self._session is not None

        face_swapper_inputs = {}
        for face_swapper_input in self._session.get_inputs():
            if face_swapper_input.name == "source":
                face_swapper_inputs[face_swapper_input.name] = (
                    self._prepare_source_embedding(source_face)
                )
            if face_swapper_input.name == "target":
                face_swapper_inputs[face_swapper_input.name] = crop_vision_frame

        return self._session.run(None, face_swapper_inputs)[0][0]

    def swap_face(
        self, temp_vision_frame: VisionFrame, source_face: Face, target_face: Face
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
        crop_vision_frame = self._forward(crop_vision_frame, source_face)
        crop_vision_frame = self._normalize_crop_frame(crop_vision_frame)
        crop_mask = box_mask.clip(0, 1)
        temp_vision_frame = paste_back(
            temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix
        )
        return temp_vision_frame

    def _prepare_source_embedding(self, source_face: Face) -> Embedding:
        source_embedding = source_face.embedding.reshape((1, -1))
        source_embedding = numpy.dot(source_embedding, self._model) / numpy.linalg.norm(
            source_embedding
        )
        return source_embedding

    def _prepare_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (
            crop_vision_frame - self._model_mean
        ) / self._model_standard_deviation
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0).astype(
            numpy.float32
        )
        return crop_vision_frame

    def _normalize_crop_frame(self, crop_vision_frame: VisionFrame) -> VisionFrame:
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        crop_vision_frame = crop_vision_frame.clip(0, 1)
        crop_vision_frame = crop_vision_frame[:, :, ::-1] * 255
        return crop_vision_frame
