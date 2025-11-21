from dataclasses import dataclass
from typing import Callable
import onnxruntime as ort


from typing import Protocol, Union, Tuple
import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]


class PredictFn(Protocol):
    def __call__(self, x: Array) -> Array: ...


class PredictProbaFn(Protocol):
    def __call__(self, x: Array) -> Array: ...


class PredictWithGradFn(Protocol):
    def __call__(self, x: Array) -> Tuple[Array, Array]: ...


@dataclass(frozen=True)
class FunctionalModel:
    predict: Callable[[Array], Array]
    predict_proba: Callable[[Array], Array]
    predict_with_grad: Callable[[Array], Tuple[Array, Array]]


def make_onnx_model(session: ort.InferenceSession) -> FunctionalModel:
    def predict(x: Array) -> Array:
        inputs = {session.get_inputs()[0].name: x}
        return session.run(None, inputs)[0]

    def predict_proba(x: Array) -> Array:
        inputs = {session.get_inputs()[0].name: x}
        outputs = session.run(None, inputs)[0]
        exp = np.exp(outputs - np.max(outputs, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def predict_with_grad(x: Array):
        raise NotImplementedError("ONNX runtime does not natively support gradients")

    return FunctionalModel(
        predict=predict,
        predict_proba=predict_proba,
        predict_with_grad=predict_with_grad,
    )


def make_torch_model(model: torch.nn.Module) -> FunctionalModel:
    def predict_proba(x: Array) -> Array:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).float()
            torch.tensor(x).float()
        with torch.no_grad():
            return model(x).detach().numpy()

    def predict(x: Array) -> Array:
        y_pred = predict_proba(x)
        return np.argmax(y_pred, axis=-1)

    def predict_with_grad(x: Array) -> Tuple[Array, Array]:
        raise NotImplementedError("ONNX runtime does not natively support gradients")

    return FunctionalModel(
        predict=predict,
        predict_proba=predict_proba,
        predict_with_grad=predict_with_grad,
    )
