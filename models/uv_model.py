"""
UV model: 22 parallel 1D-CNN branches + dual head.
MPI model: simple 1D-CNN binary classifier.
Paper: Sections 5.1 and 6.4 of arXiv:2302.01751

Single GPU only. No DataParallel.

Run: python models/uv_model.py
Expected:
  UV params: ~8,377,547
  logits: torch.Size([4, 60])
  proj:   torch.Size([4, 64])
  siamese norms: all ~1.0
  MPI output: torch.Size([8, 2])
  All model tests passed.
"""

import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import cfg


class UVBranch(nn.Module):
    """
    One of 22 identical parallel branches.
    Input:  (batch, 4, T)
    Output: (batch, 256)
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(8)
        self.fc    = nn.Linear(128 * 8, 256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.fc(self.pool(x).flatten(1))


class UVModel(nn.Module):
    """
    Paper Section 6.4 architecture:
      22 parallel branches → concatenate → dual head:
        Head A: Linear → n_classes  (cross-entropy, classifier)
        Head B: Linear → L2-norm → MLP  (Siamese + SupCon)
    """
    def __init__(self, n_classes: int, n_features: int = 22):
        super().__init__()
        self.n_features = n_features
        self.embed_dim  = n_features * 256   # 22 * 256 = 5632

        self.branches      = nn.ModuleList(
            [UVBranch(in_channels=4) for _ in range(n_features)])
        self.head_a        = nn.Linear(self.embed_dim, n_classes)
        self.siamese_proj  = nn.Linear(self.embed_dim, 256)
        self.head_b        = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 64))

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paper Section 6.2: randomly cut 1.5-second series to 1 second,
        then add random Gaussian noise.
        """
        T_t = int(cfg.uv_window_sec * cfg.uv_sampling_rate)  # 50
        T   = x.size(-1)
        if T > T_t:
            start = torch.randint(0, T - T_t + 1, (1,)).item()
            x     = x[..., start : start + T_t]
        return x + torch.randn_like(x) * 0.01

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 22, 4, T) → (B, 5632)"""
        return torch.cat(
            [self.branches[i](x[:, i, :, :])
             for i in range(self.n_features)], dim=1)

    def forward(self, x: torch.Tensor, augment: bool = False):
        """
        x: (B, 22, 4, T)
        Returns: logits (B, n_classes),  proj (B, 64)
        """
        if augment:
            x = self._augment(x)
        emb     = self.extract_embedding(x)
        logits  = self.head_a(emb)
        siamese = F.normalize(self.siamese_proj(emb), dim=1)
        return logits, self.head_b(siamese)

    def get_siamese_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Inference only. Returns L2-normalised (B, 256) embedding."""
        with torch.no_grad():
            emb = self.extract_embedding(x)
            return F.normalize(self.siamese_proj(emb), dim=1)


class MPIModel(nn.Module):
    """
    Paper Section 5.1: CNN with pointwise convolutions,
    cross-entropy loss, 2 output classes (unlock / no-unlock).
    Input: (B, C, T) → (B, 2)
    """
    def __init__(self, n_channels: int = 18, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8, 256), nn.ReLU(),
            nn.Linear(256, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.net(x))


if __name__ == "__main__":
    torch.manual_seed(42)

    print("UV Model test")
    model = UVModel(n_classes=60)
    print(f"  UV params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(4, 22, 4, 50)
    logits, proj = model(x)
    assert logits.shape == (4, 60), f"Got {logits.shape}"
    assert proj.shape   == (4, 64), f"Got {proj.shape}"
    print(f"  logits: {logits.shape}")
    print(f"  proj:   {proj.shape}")

    x75 = torch.randn(4, 22, 4, 75)
    l2, _ = model(x75, augment=True)
    assert l2.shape == (4, 60)
    print(f"  augment T=75 input → {l2.shape}  OK")

    siam  = model.get_siamese_embed(x)
    norms = siam.norm(dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)
    print(f"  siamese norms: {norms.tolist()}  all ~1.0")

    print("\nMPI Model test")
    mpi = MPIModel(n_channels=18, n_classes=2)
    out = mpi(torch.randn(8, 18, 150))
    assert out.shape == (8, 2)
    print(f"  MPI output: {out.shape}")

    print("\nAll model tests passed.")
