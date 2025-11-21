# tests/metrics/model_metrics/test_execute.py
import uuid
import os

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

from a4s_eval.metric_registries.model_metric_registry import (
    model_metric_registry,
    ModelMetric,
)
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.service.model_factory import load_model
from a4s_eval.service.model_functional import FunctionalModel
from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    ModelConfig,
    ModelFramework,
    ModelTask,
)

from tests.save_measures_utils import save_measures


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )
    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(tab_class_test_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = tab_class_test_data.copy()
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(tab_class_train_data, data_shape: DataShape) -> Dataset:
    data = tab_class_train_data.copy()
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(pid=uuid.uuid4(), model=None, dataset=ref_dataset)


@pytest.fixture
def functional_model() -> TabularClassificationModel:
    model_config = ModelConfig(
        path="./tests/data/lcld_v2_tabtransformer.pt",
        framework=ModelFramework.TORCH,
        task=ModelTask.CLASSIFICATION,
    )

    model = load_model(model_config)
    if not isinstance(model, TabularClassificationModel):
        raise TypeError("Loaded model is not TabularClassificationModel")
    return model


def test_non_empty_registry():
    assert len(model_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_data_metric_registry_contains_evaluator(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TabularClassificationModel,
):
    """
    Run every registered model metric. For pgd_asr we run a custom image-based scenario
    (single duck image => create dataset with _x_tensor/_y_tensor and a FunctionalModel wrapping resnet).
    For other metrics we call them with the standard tabular fixtures.
    """
    name, func = evaluator_function

    # special-case: run your PGD ASR metric (image / resnet) using a tiny dataset (one image)
    if name == "pgd_asr":
        device = "cpu"
        img_path = "tests/data/duck.png"
        assert os.path.exists(img_path), f"Image not found at {img_path}"

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        img = Image.open(img_path).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(device)

        # load a pretrained resnet for the PGD attack (we use it directly, it's only for testing)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(
            device
        )
        resnet.eval()
        with torch.no_grad():
            pred = resnet(x).argmax(dim=1)

        # Build a minimal DataShape accepted by the framework
        ds_shape = DataShape.model_validate(
            {
                "features": [],
                "target": {
                    "pid": uuid.uuid4(),
                    "name": "label",
                    "feature_type": "categorical",
                    "min_value": 0,
                    "max_value": 9999,
                },
                "date": None,
            }
        )

        # Create a Dataset and inject private tensors required by the PGD metric
        dataset = Dataset(
            pid=uuid.uuid4(), shape=ds_shape, data=pd.DataFrame([{"dummy": 0}])
        )
        # attach tensors used by the metric implementation
        object.__setattr__(dataset, "_x_tensor", x)
        object.__setattr__(dataset, "_y_tensor", pred)

        # Wrap the resnet into a FunctionalModel compatible with the metric registry
        functional_model_img = FunctionalModel(
            predict=lambda t: resnet(t),
            predict_proba=lambda t: F.softmax(resnet(t), dim=1).detach().cpu().numpy(),
            predict_with_grad=lambda t: (resnet(t), torch.zeros_like(resnet(t))),
        )

        measures = func(dataset.shape, None, dataset, functional_model_img)

    else:
        # normal path for tabular / other metrics
        measures = func(data_shape, ref_model, test_dataset, functional_model)

    save_measures(name, measures)
    assert len(measures) > 0
