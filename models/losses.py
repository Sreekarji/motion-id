"""
Loss functions for Motion ID UV stage.
Paper Section 6.4: L_total = LCE + alpha_TM * LTM + LSC
  LCE = cross-entropy (classifier head)
  LTM = Triplet Margin Loss, semi-hard mining (Schroff et al. 2015)
  LSC = Supervised Contrastive Loss (Khosla et al. 2020)

Run: python models/losses.py
Expected: All loss unit tests passed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self._ce(logits, labels)


class TripletMarginLoss(nn.Module):
    """
    Semi-hard negative mining.
    For each anchor: find negatives farther than the positive
    but within margin. Falls back to hardest negative if none found.
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        device = embeddings.device
        B      = embeddings.size(0)
        dist   = ((embeddings.unsqueeze(0) -
                   embeddings.unsqueeze(1)) ** 2).sum(2)   # (B, B)
        leq    = labels.unsqueeze(0) == labels.unsqueeze(1)
        eye    = torch.eye(B, dtype=torch.bool, device=device)

        losses = []
        for i in range(B):
            pm = leq[i] & ~eye[i]   # valid positives
            nm = ~leq[i]            # valid negatives
            if not pm.any() or not nm.any():
                continue
            d_ap  = dist[i][pm].max()
            d_neg = dist[i][nm]
            semi  = d_neg[(d_neg > d_ap) & (d_neg < d_ap + self.margin)]
            d_an  = semi.min() if semi.numel() > 0 else d_neg.min()
            losses.append(F.relu(d_ap - d_an + self.margin))

        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()


class SupervisedContrastiveLoss(nn.Module):
    """
    Khosla et al. 2020 — cited in paper Section 6.4.
    Expects proj_embeddings of shape (2B, D): two augmented views
    concatenated along the batch dimension. labels shape (B,).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_embeddings, labels):
        device     = proj_embeddings.device
        N          = proj_embeddings.size(0)
        B          = labels.size(0)
        labels_rep = labels.repeat(2) if N == 2 * B else labels

        z        = F.normalize(proj_embeddings, dim=1)
        sim      = torch.mm(z, z.T) / self.temperature
        eye      = torch.eye(N, dtype=torch.bool, device=device)
        pos_mask = (labels_rep.unsqueeze(0) ==
                    labels_rep.unsqueeze(1)) & ~eye

        # Numerical stability
        sim      = sim - sim.max(dim=1, keepdim=True).values.detach()
        exp_sim  = torch.exp(sim)
        denom    = exp_sim.masked_fill(eye, 0).sum(1, keepdim=True)
        log_prob = sim - torch.log(denom + 1e-8)

        n_pos = pos_mask.sum(1).float()
        valid = n_pos > 0
        loss  = -(log_prob * pos_mask.float()).sum(1)
        return (loss[valid] / n_pos[valid]).mean()


class TotalLoss(nn.Module):
    """
    Paper Section 6.4: L_total = LCE + alpha_TM * LTM + LSC
    """
    def __init__(self, alpha_tm: float = 1.0, temperature: float = 0.07):
        super().__init__()
        self.alpha_tm = alpha_tm
        self.ce       = CrossEntropyLoss()
        self.tm       = TripletMarginLoss(margin=1.0)
        self.sc       = SupervisedContrastiveLoss(temperature=temperature)

    def forward(self, logits, proj_embeds, labels):
        """
        logits:      (B, n_classes)
        proj_embeds: (2B, D)  — two augmented views concatenated
        labels:      (B,)
        Returns: (scalar total loss, dict of component values)
        """
        lce   = self.ce(logits, labels)
        ltm   = self.tm(
            F.normalize(proj_embeds[:labels.size(0)], dim=1), labels)
        lsc   = self.sc(proj_embeds, labels)
        total = lce + self.alpha_tm * ltm + lsc
        return total, {
            "lce": lce.item(),
            "ltm": ltm.item(),
            "lsc": lsc.item()}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D, n_cls = 8, 64, 10
    emb    = F.normalize(torch.randn(2 * B, D), dim=1)
    logits = torch.randn(B, n_cls)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

    loss_fn       = TotalLoss(alpha_tm=1.0, temperature=0.07)
    loss, parts   = loss_fn(logits, emb, labels)

    print(f"Total loss: {loss.item():.4f}")
    for k, v in parts.items():
        print(f"  {k}: {v:.4f}")
    assert torch.isfinite(loss) and loss.item() > 0
    print("All loss unit tests passed.")
