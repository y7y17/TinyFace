from typing import Optional, Tuple

import numpy
from onnxruntime import InferenceSession

from ..config import global_config
from ..download import download
from ..inference import create_inference_session
from ..typing import Detection, Embedding, FaceLandmark5, VisionFrame
from ..utils import warp_face_by_face_landmark_5


class FaceEmbedder:
    def __init__(self) -> None:
        self._session: Optional[InferenceSession] = None
        self.model_path = global_config().face_embedder_model
        self._model_size = (112, 112)
        self._model_warp_template = numpy.array(
            [
                [0.34191607, 0.46157411],
                [0.65653393, 0.45983393],
                [0.50022500, 0.64050536],
                [0.37097589, 0.82469196],
                [0.63151696, 0.82325089],
            ]
        )

    def prepare(self):
        if not self.model_path:
            self.model_path = download(
                url="https://github.com/idootop/TinyFace/releases/download/models-1.0.0/arcface_w600k_r50.onnx",
                known_hash="f1f79dc3b0b79a69f94799af1fffebff09fbd78fd96a275fd8f0cbbea23270d1",
            )
        if not self._session:
            self._session = create_inference_session(self.model_path)

    def _forward(self, crop_vision_frame: VisionFrame) -> Detection:
        self.prepare()
        assert self._session is not None
        return self._session.run(None, {"input": crop_vision_frame})[0]  # type:ignore

    def calc_embedding(
        self, temp_vision_frame: VisionFrame, face_landmark_5: FaceLandmark5
    ) -> Tuple[Embedding, Embedding]:
        crop_vision_frame, _ = warp_face_by_face_landmark_5(
            temp_vision_frame,
            face_landmark_5,
            self._model_warp_template,
            self._model_size,
        )
        crop_vision_frame = crop_vision_frame / 127.5 - 1
        crop_vision_frame = (
            crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
        )
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0)
        embedding = self._forward(crop_vision_frame)
        embedding = embedding.ravel()
        normed_embedding = embedding / numpy.linalg.norm(embedding)
        return embedding, normed_embedding
