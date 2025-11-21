from datetime import datetime
import torch
import torch.nn.functional as F

from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.service.model_functional import FunctionalModel

from torch.utils.data import DataLoader, TensorDataset


def _build_loader_from_dataset(dataset: Dataset, datashape: DataShape, batch_size=32):
    x_tensor = getattr(dataset, "_x_tensor", None)
    y_tensor = getattr(dataset, "_y_tensor", None)

    if x_tensor is None or y_tensor is None:
        raise ValueError("Dataset must have _x_tensor and _y_tensor for PGD metric.")

    return DataLoader(
        TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=False
    )


def pgd_attack_torch(model, x, y, eps=0.01, alpha=0.005, iters=7, device="cpu"):
    x_adv = x.clone().detach().to(device)
    x_adv.requires_grad_(True)

    for _ in range(iters):
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)

        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
        x_adv.requires_grad_(True)

    return x_adv.detach()


@model_metric(name="pgd_asr")
def pgd_asr_metric(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if hasattr(functional_model, "as_torch_model"):
        torch_model = functional_model.as_torch_model().to(device)
    else:

        class Wrapper(torch.nn.Module):
            def forward(self, x):
                out = functional_model.predict(x)
                return torch.from_numpy(out) if isinstance(out, (list, tuple)) else out

        torch_model = Wrapper().to(device)

    torch_model.eval()

    dl = _build_loader_from_dataset(dataset, datashape, batch_size=32)

    preds_before = []
    preds_after = []
    total = 0
    fooled = 0

    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)

        with torch.no_grad():
            clean_pred = torch_model(xb).argmax(dim=1)

        preds_before.extend(clean_pred.cpu().tolist())

        x_adv = pgd_attack_torch(
            torch_model, xb, yb, eps=0.01, alpha=0.005, iters=7, device=device
        )

        with torch.no_grad():
            adv_pred = torch_model(x_adv).argmax(dim=1)

        preds_after.extend(adv_pred.cpu().tolist())

        fooled += (adv_pred != yb).sum().item()
        total += yb.size(0)

    asr = fooled / total if total > 0 else 0.0

    m = Measure(name="pgd_asr", score=asr, time=datetime.now())
    m.pred_before = preds_before
    m.pred_after = preds_after

    return [m]
