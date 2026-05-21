"""LSTM baseline for clinical sequences.

A small, deliberately modest LSTM. The point isn't to win — it's to
provide a fair sequence-aware comparison point for the LLM, and to
demonstrate that the project handles raw temporal data correctly.

The model concatenates the value channel with the mask channel as input,
so the network can learn from both the magnitude of measurements and
their pattern of presence/absence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from clinical_llm.data.sequences import SequenceDataset, n_features


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.2
    bidirectional: bool = False
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 5
    seed: int = 42


class LSTMClassifier(nn.Module):
    """LSTM → final-state → MLP → logit.

    Input shape: (batch, time, 2 * n_features) — values and mask
    concatenated along the feature dimension.
    """

    def __init__(self, n_features_in: int, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        input_size = 2 * n_features_in  # values + mask
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        out_dim = config.hidden_size * (2 if config.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([values, mask], dim=-1)
        _, (h_n, _) = self.lstm(x)
        last = torch.cat([h_n[-1], h_n[-2]], dim=-1) if self.config.bidirectional else h_n[-1]
        return self.head(last).squeeze(-1)


class LSTMBaseline:
    """Wraps the LSTM in the same fit / predict_proba interface as the others.

    This deliberately mirrors the API used by LogisticRegressionBaseline
    and XGBoostBaseline, so the unified training script can drop it in
    without special-casing.
    """

    def __init__(self, config: LSTMConfig | None = None) -> None:
        self.config = config or LSTMConfig()
        self.model_: LSTMClassifier | None = None
        self.device_: torch.device | None = None

    def _seed_everything(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _make_loader(self, dataset: SequenceDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
        )

    def fit(
        self,
        train_dataset: SequenceDataset,
        val_dataset: SequenceDataset,
    ) -> LSTMBaseline:
        self._seed_everything()
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_ = LSTMClassifier(n_features_in=n_features(), config=self.config).to(
            self.device_
        )

        # Class-weighted BCE to handle imbalance.
        labels = np.array(train_dataset.labels)
        pos_weight = torch.tensor(
            max((labels == 0).sum() / max((labels == 1).sum(), 1), 1.0),
            dtype=torch.float32,
            device=self.device_,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = self._make_loader(val_dataset, shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        patience_left = self.config.early_stopping_patience

        for _epoch in range(1, self.config.epochs + 1):
            self.model_.train()
            train_loss = 0.0
            n_train = 0
            for values, mask, label in train_loader:
                values = values.to(self.device_)
                mask = mask.to(self.device_)
                label = label.to(self.device_)
                optimizer.zero_grad()
                logits = self.model_(values, mask)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * label.size(0)
                n_train += label.size(0)
            train_loss /= max(n_train, 1)

            val_loss = self._validation_loss(val_loader, criterion)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()
                }
                patience_left = self.config.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _validation_loss(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        assert self.model_ is not None and self.device_ is not None
        self.model_.eval()
        total, n = 0.0, 0
        for values, mask, label in loader:
            values = values.to(self.device_)
            mask = mask.to(self.device_)
            label = label.to(self.device_)
            logits = self.model_(values, mask)
            loss = criterion(logits, label)
            total += loss.item() * label.size(0)
            n += label.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def predict_proba(self, dataset: SequenceDataset) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("Model must be fit before calling predict_proba.")
        self.model_.eval()
        loader = self._make_loader(dataset, shuffle=False)
        probs: list[np.ndarray] = []
        for values, mask, _ in loader:
            values = values.to(self.device_)
            mask = mask.to(self.device_)
            logits = self.model_(values, mask)
            probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs, axis=0)
