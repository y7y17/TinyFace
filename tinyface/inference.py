from functools import lru_cache

import onnx
import onnxruntime as ort

from .config import global_config

ort.set_default_logger_severity(3)


def create_inference_session(model_path: str, providers=None):
    return ort.InferenceSession(
        model_path,
        providers=providers or global_config().face_inference_providers,
    )


@lru_cache(maxsize=None)
def load_model(model_path: str):
    model = onnx.load(model_path)
    return onnx.numpy_helper.to_array(model.graph.initializer[-1])
