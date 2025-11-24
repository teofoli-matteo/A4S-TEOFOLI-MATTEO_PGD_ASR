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
from a4s_eval.service.functional_model import TabularClassificationModel

ROOT = "tiny-imagenet-200"
N_IMAGES = 500
MEASURES_DIR = "tests/data/measures"
os.makedirs(MEASURES_DIR, exist_ok=True)

def load_tiny_imagenet_paths():
    pattern = os.path.join(ROOT, "train/*/images/*.JPEG")
    paths = sorted(glob.glob(pattern))
    assert len(paths) >= N_IMAGES, "Not enough Tiny-ImageNet images"
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
    torch_model.eval()
    torch_model.to(device)

    def predict_class(x):
        x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y_pred = torch_model(x_tensor.to(device))
        return y_pred.argmax(dim=1).cpu().numpy()

    def predict_proba(x):
        x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y_pred = torch_model(x_tensor.to(device))
        return torch.nn.functional.softmax(y_pred, dim=1).cpu().numpy()

    def predict_proba_grad(x):
        x_tensor = x.to(device)
        x_tensor.requires_grad_(True)
        logits = torch_model(x_tensor)
        return logits  

    return TabularClassificationModel(
        predict_class=predict_class,
        predict_proba=predict_proba,
        predict_proba_grad=predict_proba_grad,
    )


@pytest.mark.parametrize(
    "model_name", ["resnet18", "mobilenet_v2", "vgg16", "densenet121"]
)
def test_pgd_asr_on_tiny_imagenet(model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    model = model.to(device).eval()

    class_to_idx = load_class_mapping()
    paths = load_tiny_imagenet_paths()
    xs, ys = [], []

    for path in paths:
        x = preprocess_image(path, device)
        class_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
        true_label = class_to_idx[class_folder]
        xs.append(x.cpu())
        ys.append(torch.tensor([true_label]))

    xs_np = [x.numpy().squeeze(0) for x in xs]
    ys_np = [y.numpy() for y in ys]

    df = pd.DataFrame({"image": xs_np, "label": ys_np})
    shape = DataShape.model_validate({
        "features": [],
        "target": {"pid": uuid.uuid4(), "name": "label", "feature_type": "categorical", "min_value": 0, "max_value": 199},
        "date": None,
    })
    dataset = Dataset(pid=uuid.uuid4(), shape=shape, data=df)
    evaluator = next(f for f in model_metric_registry if f[0] == "pgd_asr")[1]

    functional_model = wrap_model_for_a4s(model, device)
    measures = evaluator(dataset.shape, None, dataset, functional_model)
    m = measures[0]

    csv_path = os.path.join(MEASURES_DIR, f"pgd_asr_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_out = pd.DataFrame({"pred_before": m.pred_before, "pred_after": m.pred_after})
    df_out["asr"] = (df_out["pred_before"] != df_out["pred_after"]).astype(float)
    df_out.to_csv(csv_path, index=False)
    print(f"Global ASR: {m.score:.4f}, Saved: {csv_path}")
    assert 0.0 <= m.score <= 1.0
