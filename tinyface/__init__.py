from typing import List

import numpy

from .config import global_config
from .core.detector import FaceDetector
from .core.embedder import FaceEmbedder
from .core.enhancer import FaceEnhancer
from .core.swapper import FaceSwapper
from .typing import BoundingBox, Face, FaceLandmark5, FacePair, VisionFrame


class TinyFace:
    def __init__(
        self,
    ):
        self.config = global_config()
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.swapper = FaceSwapper()
        self.enhancer = FaceEnhancer()

    def prepare(self, detection_only=False):
        self.detector.prepare()
        self.embedder.prepare()
        if detection_only:
            return
        self.swapper.prepare()
        self.enhancer.prepare()

    def get_many_faces(self, vision_frame: VisionFrame):
        boxes, scores, landmarks_5 = self.detector.detect_faces(vision_frame)
        return self._create_faces(vision_frame, boxes, scores, landmarks_5)

    def get_one_face(self, vision_frame: VisionFrame):
        faces = self.get_many_faces(vision_frame)
        return faces[0] if faces else None

    def get_target_faces(self, vision_frame: VisionFrame, reference_face: Face):
        faces = self.get_many_faces(vision_frame)
        return [face for face in faces if self._is_similar_face(face, reference_face)]

    def swap_face(
        self, vision_frame: VisionFrame, reference_face: Face, destination_face: Face
    ) -> VisionFrame:
        temp_vision_frame = vision_frame.copy()
        faces = self.get_target_faces(vision_frame, reference_face)
        for face in faces:
            temp_vision_frame = self.swapper.swap_face(
                temp_vision_frame, destination_face, face
            )
            temp_vision_frame = self.enhancer.enhance_face(temp_vision_frame, face)
        return temp_vision_frame

    def swap_faces(
        self, vision_frame: VisionFrame, face_pairs: List[FacePair]
    ) -> VisionFrame:
        temp_vision_frame = vision_frame.copy()
        faces = self.get_many_faces(vision_frame)
        for face in faces:
            for pair in face_pairs:
                reference_face = pair.reference
                destination_face = pair.destination
                if not self._is_similar_face(face, reference_face):
                    continue
                temp_vision_frame = self.swapper.swap_face(
                    temp_vision_frame, destination_face, face
                )
                temp_vision_frame = self.enhancer.enhance_face(temp_vision_frame, face)
        return temp_vision_frame

    def _is_similar_face(self, face: Face, reference_face: Face):
        cos_distance = numpy.dot(face.normed_embedding, reference_face.normed_embedding)
        return cos_distance > global_config().face_similarity_threshold

    def _create_faces(
        self,
        vision_frame: VisionFrame,
        bounding_boxes: List[BoundingBox],
        face_scores: List[float],
        face_landmarks_5: List[FaceLandmark5],
    ) -> List[Face]:
        faces = []
        for index, bounding_box in enumerate(bounding_boxes):
            face_score = face_scores[index]
            face_landmark_5 = face_landmarks_5[index]
            embedding, normed_embedding = self.embedder.calc_embedding(
                vision_frame, face_landmark_5
            )
            faces.append(
                Face(
                    bounding_box=bounding_box,
                    score=face_score,
                    landmark_5=face_landmark_5,
                    embedding=embedding,
                    normed_embedding=normed_embedding,
                )
            )
        return faces
