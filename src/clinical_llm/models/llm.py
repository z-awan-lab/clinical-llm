"""LLM baseline: parameter-efficient fine-tuning for clinical sequence classification.

We use the same fit / predict_proba interface as the other baselines so the
training script treats this uniformly. Under the hood:

  * Any Hugging Face causal-LM identifier is accepted via config. Default:
    google/medgemma-4b-it. This is a Gemma-family model pretrained on a
    large medical corpus, so it begins fine-tuning with strong domain
    priors on vocabulary and reasoning patterns relevant to clinical text.
  * The base model is loaded in 4-bit precision (NF4 via bitsandbytes),
    which lets a 4B-parameter model train comfortably on a single 16-24GB
    GPU with batch sizes that make optimisation stable.
  * LoRA adapters (rank 16 by default) are added to the attention
    projections; the rest of the base model is frozen. This is the
    standard modern PEFT setup, and what an industry team would do.
  * For classification we add a small linear head on top of the final
    hidden state at the last non-pad token. Reading the last token
    matches causal-LM training and avoids the padding-position confusion
    that derails naive implementations.
  * Training uses class-weighted binary cross-entropy to handle the
    natural imbalance of in-hospital mortality (~10% positive in MIMIC).

The licence note: MedGemma is released under the Gemma terms of use.
Research and benchmarking is in-scope; this code does not deploy a
clinical decision tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class LLMConfig:
    """Configuration for the LLM baseline."""

    model_name: str = "google/medgemma-4b-it"
    # Tokenisation
    max_length: int = 2048
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    # Quantisation
    use_4bit: bool = True
    # Training
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    early_stopping_patience: int = 2
    warmup_ratio: float = 0.03
    seed: int = 42
    # Hardware
    device_map: str = "auto"
    dtype: str = "bfloat16"
    # Extras
    hf_token_env: str = "HF_TOKEN"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class _PromptDataset(Dataset):
    """Wraps prompts and labels into a torch Dataset.

    Tokenisation happens here (on init) so the trainer does not pay the
    cost in the inner loop. Padding to a per-batch dynamic length keeps
    memory tight without truncating away signal unnecessarily.
    """

    def __init__(
        self,
        prompts: list[str],
        labels: list[int] | None,
        tokenizer,
        max_length: int,
    ) -> None:
        self.encoded = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=True,
        )
        self.labels = labels  # may be None at inference time

    def __len__(self) -> int:
        return len(self.encoded["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {
            "input_ids": torch.tensor(self.encoded["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.encoded["attention_mask"][idx], dtype=torch.long
            ),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def _collate(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch to the longest sequence in the batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    has_labels = "label" in batch[0]
    labels = torch.zeros((len(batch),), dtype=torch.float32) if has_labels else None

    for i, item in enumerate(batch):
        L = item["input_ids"].size(0)
        input_ids[i, :L] = item["input_ids"]
        attention_mask[i, :L] = item["attention_mask"]
        if has_labels:
            labels[i] = item["label"]

    out = {"input_ids": input_ids, "attention_mask": attention_mask}
    if has_labels:
        out["label"] = labels
    return out


class _LLMClassifier(nn.Module):
    """Wraps a (LoRA-decorated, possibly quantised) base LM with a classification head.

    For each example we read the hidden state at the position of the
    last real (non-pad) token. This is the conventional position for
    causal-LM classification heads.
    """

    def __init__(self, base_model, hidden_size: int) -> None:
        super().__init__()
        self.base = base_model
        self.head = nn.Linear(hidden_size, 1)

    def _last_token_hidden(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Find the index of the final 1 in each row of the attention mask.
        lengths = attention_mask.sum(dim=1) - 1  # (B,)
        idx = lengths.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        return hidden.gather(1, idx).squeeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]
        pooled = self._last_token_hidden(last_hidden, attention_mask)
        # Match head's device and dtype — base may be on GPU in bf16/quantised.
        pooled = pooled.to(device=self.head.weight.device, dtype=self.head.weight.dtype)
        return self.head(pooled).squeeze(-1)


class LLMBaseline:
    """LoRA fine-tuning wrapper exposing the same API as the other baselines.

    Lazy imports of transformers / peft / bitsandbytes mean the rest of
    the project (logistic regression, XGBoost, LSTM, tests) runs without
    these heavy dependencies installed.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self.tokenizer = None
        self.model_: _LLMClassifier | None = None
        self.device_: torch.device | None = None
        self._dtype: torch.dtype | None = None

    # --------------------------------------------------------------- setup
    def _build_model(self):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self._dtype = torch.bfloat16 if self.config.dtype == "bfloat16" else torch.float16

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        load_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
            "torch_dtype": self._dtype,
        }
        if self.config.use_4bit and torch.cuda.is_available():
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self._dtype,
            )

        base = AutoModelForCausalLM.from_pretrained(self.config.model_name, **load_kwargs)

        if self.config.use_4bit and torch.cuda.is_available():
            base = prepare_model_for_kbit_training(base)

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, lora_config)
        base.print_trainable_parameters()

        # Gemma 3 nests text params under `text_config`; older/flat configs put
        # them at the top level. Probe both.
        cfg = base.config
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None and hasattr(cfg, "text_config"):
            hidden_size = cfg.text_config.hidden_size
        if hidden_size is None:
            raise ValueError(
                f"Could not determine hidden_size from {type(cfg).__name__}"
            )

        classifier = _LLMClassifier(base, hidden_size=hidden_size)
        # Place the classification head on the same device and dtype as the base.
        device = next(base.parameters()).device
        classifier.head = classifier.head.to(device=device, dtype=self._dtype)
        return classifier

    # --------------------------------------------------------------- train
    def fit(
        self,
        train_prompts: list[str],
        train_labels: list[int],
        val_prompts: list[str],
        val_labels: list[int],
    ) -> LLMBaseline:
        self.model_ = self._build_model()
        # Determine device from the base model (4-bit weights live on GPU).
        self.device_ = next(self.model_.base.parameters()).device

        train_ds = _PromptDataset(
            train_prompts, train_labels, self.tokenizer, self.config.max_length
        )
        val_ds = _PromptDataset(
            val_prompts, val_labels, self.tokenizer, self.config.max_length
        )

        collate = lambda b: _collate(b, self.tokenizer.pad_token_id)  # noqa: E731
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate,
        )

        # Class-weighted loss.
        labels = np.array(train_labels)
        pos_weight = torch.tensor(
            max((labels == 0).sum() / max((labels == 1).sum(), 1), 1.0),
            dtype=torch.float32,
            device=self.device_,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimise only trainable params (LoRA + head).
        trainable = [p for p in self.model_.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        n_steps = max(
            1,
            (len(train_loader) // self.config.gradient_accumulation_steps)
            * self.config.epochs,
        )
        n_warmup = int(self.config.warmup_ratio * n_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(n_warmup, 1)),
        )

        best_val = float("inf")
        patience_left = self.config.early_stopping_patience
        best_state: dict | None = None

        for _epoch in range(self.config.epochs):
            self.model_.train()
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device_)
                attention_mask = batch["attention_mask"].to(self.device_)
                label = batch["label"].to(self.device_)
                logits = self.model_(input_ids, attention_mask)
                loss = criterion(logits, label)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            val_loss = self._val_loss(val_loader, criterion)
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                # Snapshot only the trainable parameters (LoRA adapters + head).
                # Filtering on requires_grad avoids the multimodal vision-tower
                # state-dict noise and is robust to PEFT key naming changes.
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model_.named_parameters()
                    if v.requires_grad
                }
                patience_left = self.config.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            # Restore best trainable-only weights in place.
            with torch.no_grad():
                for name, param in self.model_.named_parameters():
                    if name in best_state:
                        param.copy_(
                            best_state[name].to(param.device, dtype=param.dtype)
                        )
        return self

    @torch.no_grad()
    def _val_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        assert self.model_ is not None and self.device_ is not None
        self.model_.eval()
        total, n = 0.0, 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device_)
            attention_mask = batch["attention_mask"].to(self.device_)
            label = batch["label"].to(self.device_)
            logits = self.model_(input_ids, attention_mask)
            loss = criterion(logits, label)
            total += loss.item() * label.size(0)
            n += label.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def predict_proba(self, prompts: list[str]) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("Model must be fit before calling predict_proba.")
        ds = _PromptDataset(prompts, None, self.tokenizer, self.config.max_length)
        collate = lambda b: _collate(b, self.tokenizer.pad_token_id)  # noqa: E731
        loader = DataLoader(
            ds, batch_size=self.config.batch_size, shuffle=False, collate_fn=collate
        )
        self.model_.eval()
        probs: list[np.ndarray] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device_)
            attention_mask = batch["attention_mask"].to(self.device_)
            logits = self.model_(input_ids, attention_mask)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(probs, axis=0)

    def save(self, out_dir: Path) -> None:
        """Save LoRA adapters and classification head separately.

        The base model is not saved — it's just the published HF weights
        and can be re-downloaded. This keeps checkpoints tiny (~50MB)
        instead of multi-gigabyte.
        """
        if self.model_ is None:
            raise RuntimeError("Nothing to save: model not fit.")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # PEFT adapters.
        self.model_.base.save_pretrained(out_dir / "lora_adapter")
        # Classification head.
        torch.save(self.model_.head.state_dict(), out_dir / "classifier_head.pt")
        # Tokenizer (so inference doesn't need redownload-and-pray).
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(out_dir / "tokenizer")
