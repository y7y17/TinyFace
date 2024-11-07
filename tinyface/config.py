import os


class GlobalConfig:
    def __init__(self):
        self.face_detector_model = os.getenv("FACE_DETECTOR_MODEL")
        self.face_embedder_model = os.getenv("FACE_EMBEDDER_MODEL")
        self.face_swapper_model = os.getenv("FACE_SWAPPER_MODEL")
        self.face_enhancer_model = os.getenv("FACE_ENHANCER_MODEL")
        self.face_inference_providers = os.getenv(
            "FACE_INFERENCE_PROVIDERS", "CPUExecutionProvider"
        ).split(",")
        self.face_similarity_threshold = self._get_float(
            "FACE_SIMILARITY_THRESHOLD", 0.5
        )
        self.face_detector_size = os.getenv("FACE_DETECTOR_SIZE", "640x640")
        self.face_detector_score = self._get_float("FACE_DETECTOR_SCORE", 0.5)
        self.face_mask_blur = self._get_float("FACE_MASK_BLUR", 0.3)
        p = self._get_int("FACE_MASK_PADDING", 0)
        self.face_mask_padding = (p, p, p, p)
        self.face_enhancer_blend = self._get_float("FACE_ENHANCER_BLEND", 60)

    def _get_int(self, key: str, value: int) -> int:
        try:
            return int(os.getenv(key))  # type:ignore
        except BaseException:
            return value

    def _get_float(self, key: str, value: float) -> float:
        try:
            return float(os.getenv(key))  # type:ignore
        except BaseException:
            return value


_global_config = GlobalConfig()


def global_config():
    return _global_config
