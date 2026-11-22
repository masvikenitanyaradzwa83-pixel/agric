"""
PyTorch models for the Agro Advisory System.
- CropSuitabilityNet: MLP that takes engineered features and outputs suitability scores.
- Simple and production-ready (TorchScript exportable).
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CropSuitabilityNet(nn.Module):
    """
    Small MLP that refines crop suitability.
    Input features (per sample, for one crop context or global):
      [soil_pH_norm, annual_rain_norm, mean_temp_norm,
       risk_averse, risk_seeking, input_low, input_med, input_high,
       drought_tol, input_req_high]
    Output: suitability logit / score in [0,1] after sigmoid.
    We train multi-crop by repeating for each crop or using a multi-head.
    For simplicity we use a shared backbone + crop embedding.
    """

    def __init__(
        self,
        n_crops: int = 8,
        emb_dim: int = 8,
        hidden: List[int] = [64, 32],
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_crops = n_crops
        self.crop_emb = nn.Embedding(n_crops, emb_dim)

        # Continuous + one-hot style features: 10 dims
        input_dim = 10 + emb_dim
        layers = []
        prev = input_dim
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, crop_idx: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 10) engineered features
        crop_idx: (B,) long
        returns: (B, 1) raw scores
        """
        emb = self.crop_emb(crop_idx)
        h = torch.cat([x, emb], dim=-1)
        return self.mlp(h).squeeze(-1)


def build_feature_vector(
    soil_pH: float,
    annual_rain: float,
    mean_temp: float,
    risk_pref: str,
    input_access: str,
    drought_tol: float,
    input_req_high: float,
) -> List[float]:
    """Normalize and one-hot encode farmer + environment features."""
    # Simple min-max style norms based on Gweru ranges
    pH_n = (soil_pH - 4.0) / 3.5  # ~4–7.5
    rain_n = (annual_rain - 300) / 700  # ~300–1000
    temp_n = (mean_temp - 10) / 25  # ~10–35
    risk_averse = 1.0 if risk_pref == "risk_averse" else 0.0
    risk_seeking = 1.0 if risk_pref == "risk_seeking" else 0.0
    inp_low = 1.0 if input_access == "low" else 0.0
    inp_med = 1.0 if input_access == "medium" else 0.0
    inp_high = 1.0 if input_access == "high" else 0.0
    return [
        pH_n,
        rain_n,
        temp_n,
        risk_averse,
        risk_seeking,
        inp_low,
        inp_med,
        inp_high,
        drought_tol,
        input_req_high,
    ]


CROP_TO_IDX = {
    "maize": 0,
    "sorghum": 1,
    "pearl_millet": 2,
    "finger_millet": 3,
    "groundnut": 4,
    "cowpea": 5,
    "sugar_bean": 6,
    "sunflower": 7,
}
IDX_TO_CROP = {v: k for k, v in CROP_TO_IDX.items()}


def create_synthetic_training_data(
    n_samples: int = 4000,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate realistic synthetic training data using the knowledge-base scores
    as soft labels + controlled noise. This is standard practice when true
    labelled farm outcomes are scarce.
    """
    import random
    from crop_knowledge import CROPS, get_crop_recommendation_features

    rng = random.Random(seed)
    torch.manual_seed(seed)

    X_list, crop_list, y_list = [], [], []

    risk_opts = ["risk_averse", "neutral", "risk_seeking"]
    input_opts = ["low", "medium", "high"]

    for _ in range(n_samples):
        soil_pH = rng.uniform(4.2, 7.0)
        annual_rain = rng.uniform(350, 950)
        mean_temp = rng.uniform(15.0, 24.0)
        risk = rng.choice(risk_opts)
        inp = rng.choice(input_opts)

        scores = get_crop_recommendation_features(
            soil_pH, annual_rain, mean_temp, risk, inp
        )

        for crop_key, sc in scores.items():
            feats = build_feature_vector(
                soil_pH,
                annual_rain,
                mean_temp,
                risk,
                inp,
                CROPS[crop_key]["drought_tolerance"],
                1.0 if CROPS[crop_key]["input_requirement"] == "high" else 0.0,
            )
            # Soft label = composite + small noise
            label = sc["composite"] + rng.gauss(0, 0.05)
            label = max(0.01, min(0.99, label))

            X_list.append(feats)
            crop_list.append(CROP_TO_IDX[crop_key])
            y_list.append(label)

    X = torch.tensor(X_list, dtype=torch.float32)
    crop_idx = torch.tensor(crop_list, dtype=torch.long)
    y = torch.tensor(y_list, dtype=torch.float32)
    return X, crop_idx, y


def train_suitability_model(
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> CropSuitabilityNet:
    from torch.utils.data import TensorDataset, DataLoader
    import copy

    X, crop_idx, y = create_synthetic_training_data()
    # Simple random split
    n = len(X)
    perm = torch.randperm(n)
    train_idx = perm[: int(0.8 * n)]
    val_idx = perm[int(0.8 * n) :]

    train_ds = TensorDataset(X[train_idx], crop_idx[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], crop_idx[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CropSuitabilityNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, cb, yb in train_loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            opt.zero_grad()
            pred = torch.sigmoid(model(xb, cb))
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, cb, yb in val_loader:
                xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
                pred = torch.sigmoid(model(xb, cb))
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(val_ds)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train MSE {train_loss:.4f} | val MSE {val_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"Best val MSE: {best_val:.4f}")
    return model


def predict_crop_scores(
    model: CropSuitabilityNet,
    soil_pH: float,
    annual_rain: float,
    mean_temp: float,
    risk_pref: str,
    input_access: str,
    device: str = "cpu",
) -> Dict[str, float]:
    """Return refined suitability scores for all crops."""
    from crop_knowledge import CROPS

    model.eval()
    scores = {}
    with torch.no_grad():
        for key, idx in CROP_TO_IDX.items():
            feats = build_feature_vector(
                soil_pH,
                annual_rain,
                mean_temp,
                risk_pref,
                input_access,
                CROPS[key]["drought_tolerance"],
                1.0 if CROPS[key]["input_requirement"] == "high" else 0.0,
            )
            x = torch.tensor([feats], dtype=torch.float32, device=device)
            c = torch.tensor([idx], dtype=torch.long, device=device)
            score = torch.sigmoid(model(x, c)).item()
            scores[key] = round(score, 4)
    return scores
