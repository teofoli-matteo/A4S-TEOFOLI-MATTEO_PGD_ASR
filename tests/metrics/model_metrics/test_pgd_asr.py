import os
import glob
import uuid
from datetime import datetime

import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

import pytest


from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.service.model_functional import FunctionalModel

ROOT = "tiny-imagenet-200"
N_IMAGES = 500

MEASURES_DIR = "tests/data/measures"
os.makedirs(MEASURES_DIR, exist_ok=True)


def load_tiny_imagenet_paths():
    pattern = os.path.join(ROOT, "train/*/images/*.JPEG")
    paths = sorted(glob.glob(pattern))
    assert len(paths) >= N_IMAGES, "Pas assez d'images Tiny-ImageNet"
    return paths[:N_IMAGES]


def load_class_mapping():
    wnid_file = os.path.join(ROOT, "wnids.txt")
    with open(wnid_file, "r") as f:
        wnids = [w.strip() for w in f.readlines()]
    return {wnid: idx for idx, wnid in enumerate(wnids)}


def preprocess_image(path, device):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def wrap_model_for_a4s(torch_model, device):
    def predict_fn(x):
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        else:
            x = torch.tensor(x, dtype=torch.float32, device=device)
        return torch_model(x)

    return FunctionalModel(
        predict=predict_fn,
        predict_proba=lambda x: torch.softmax(predict_fn(x).detach(), dim=1)
        .cpu()
        .numpy(),
        predict_with_grad=lambda x: (predict_fn(x), torch.zeros(1)),
    )

@pytest.mark.parametrize(
    "model_name", ["resnet18", "mobilenet_v2", "vgg16", "densenet121"]
)
def test_pgd_asr_on_tiny_imagenet(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "resnet18":
        model = models.resnet18(models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif model_name == "vgg16":
        model = models.vgg16(models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == "densenet121":
        model = models.densenet121(models.DenseNet121_Weights.IMAGENET1K_V1)

    model = model.to(device)
    model.eval()

    class_to_idx = load_class_mapping()
    paths = load_tiny_imagenet_paths()
    xs = []
    ys = []

    for path in paths:
        x = preprocess_image(path, device)
        class_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        true_label = class_to_idx[class_folder]

        xs.append(x.cpu())
        ys.append(torch.tensor([true_label]))

    # change for numpy maybe better than _x_tensor & _y_tensor ?
    xs_np = [x.numpy().squeeze(0) for x in xs]  
    ys_np = [y.numpy() for y in ys]

    df = pd.DataFrame({"image": xs_np, "label": ys_np})

    shape = DataShape.model_validate(
        {
            "features": [],
            "target": {
                "pid": uuid.uuid4(),
                "name": "label",
                "feature_type": "categorical",
                "min_value": 0,
                "max_value": 199,
            },
            "date": None,
        }
    )

    dataset = Dataset(pid=uuid.uuid4(), shape=shape, data=df)
    evaluator = next((f for f in model_metric_registry if f[0] == "pgd_asr"), None)
    assert evaluator is not None

    functional_model = wrap_model_for_a4s(model, device)

    measures = evaluator[1](dataset.shape, None, dataset, functional_model)
    m = measures[0]

    csv_path = os.path.join(
        MEASURES_DIR,
        f"pgd_asr_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    df_out = pd.DataFrame({"pred_before": m.pred_before, "pred_after": m.pred_after})
    df_out["asr"] = (df_out["pred_before"] != df_out["pred_after"]).astype(float)

    print(f"Global ASR: {m.score:.4f}")
    df_out.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    assert 0.0 <= m.score <= 1.0
