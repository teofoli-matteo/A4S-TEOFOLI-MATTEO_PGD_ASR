import torch

from a4s_eval.data_model.evaluation import ModelConfig, ModelFramework
from a4s_eval.service.model_functional import (
    FunctionalModel,
    make_onnx_model,
    make_torch_model,
)
import onnxruntime as ort


def load_model(config: ModelConfig) -> FunctionalModel:
    if config.framework == ModelFramework.TORCH:
        model = torch.jit.load(config.path)
        model.eval()
        return make_torch_model(model)

    elif config.framework == ModelFramework.ONNX:
        session = ort.InferenceSession(config.path)
        return make_onnx_model(session)

    else:
        raise ValueError(f"Unsupported framework: {config.framework}")
