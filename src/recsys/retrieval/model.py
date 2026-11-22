"""The two-tower network (PyTorch, CPU)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TwoTowerConfig:
    dim: int = 64
    attr_dim: int = 16
    age_dim: int = 8
    history_len: int = 20
    temperature: float = 0.05
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 8
    patience: int = 2
    item_id_dropout: float = 0.1  # trains the attribute-only path used for unseen articles
    user_id_dropout: float = 0.1
    train_weeks: int = 8
    logq: bool = True
    eval_k: int = 100
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class TwoTowerModel(nn.Module):
    """User tower: id + age bucket + mean of the item-id embeddings in the recent history.
    Item tower: id + product type/group, colour group, department, garment group and
    index group. Both towers end in an MLP and an L2 normalisation, so the score is
    a cosine similarity.
    """

    def __init__(
        self, n_users: int, n_items: int, n_ages: int, attr_sizes: list[int], cfg: TwoTowerConfig
    ) -> None:
        super().__init__()
        d = cfg.dim
        self.user_emb = nn.Embedding(n_users, d)
        self.age_emb = nn.Embedding(n_ages, cfg.age_dim)
        self.item_emb = nn.Embedding(n_items, d)
        self.attr_embs = nn.ModuleList(nn.Embedding(s, cfg.attr_dim) for s in attr_sizes)
        self.user_mlp = nn.Sequential(
            nn.Linear(2 * d + cfg.age_dim, 2 * d), nn.ReLU(), nn.Linear(2 * d, d)
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(d + cfg.attr_dim * len(attr_sizes), 2 * d), nn.ReLU(), nn.Linear(2 * d, d)
        )
        for emb in (self.user_emb, self.item_emb, self.age_emb, *self.attr_embs):
            nn.init.normal_(emb.weight, std=0.05)

    def user_vec(self, user: torch.Tensor, age: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        mask = (hist > 0).unsqueeze(-1).float()
        pooled = (self.item_emb(hist) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        x = torch.cat([self.user_emb(user), self.age_emb(age), pooled], dim=-1)
        return F.normalize(self.user_mlp(x), dim=-1)

    def item_vec(self, item: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        parts = [self.item_emb(item)] + [emb(attrs[:, j]) for j, emb in enumerate(self.attr_embs)]
        return F.normalize(self.item_mlp(torch.cat(parts, dim=-1)), dim=-1)


def in_batch_loss(
    u: torch.Tensor,
    v: torch.Tensor,
    item_ids: torch.Tensor,
    log_q: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """Sampled softmax over the batch's positives, with logQ popularity correction.

    Popular items appear as in-batch negatives more often, which over-penalises
    them; subtracting log q(item) from each logit corrects the sampling bias.
    Columns holding the same item as the row's positive are masked out.
    """
    logits = u @ v.T / temperature
    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)
    same = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, float("-inf"))
    target = torch.arange(len(u), device=u.device)
    return F.cross_entropy(logits, target)
