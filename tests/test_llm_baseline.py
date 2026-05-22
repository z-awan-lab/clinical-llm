"""Structural tests for the LLM baseline.

We don't actually load MedGemma here — that's slow and requires GPU + HF
gated access. Instead we verify the config and class structure are
sensible, and skip the heavyweight integration when transformers/peft
aren't installed (which is the CI case).
"""

import pytest

# Skip the whole module if the heavy LLM deps aren't available.
pytest.importorskip("transformers")
pytest.importorskip("peft")

from clinical_llm.models.llm import LLMBaseline, LLMConfig  # noqa: E402


def test_default_config_targets_medgemma():
    cfg = LLMConfig()
    assert cfg.model_name == "google/medgemma-4b-it"


def test_default_config_uses_lora_with_sane_rank():
    cfg = LLMConfig()
    assert 4 <= cfg.lora_r <= 64
    assert cfg.lora_alpha >= cfg.lora_r


def test_default_config_targets_attention_projections():
    cfg = LLMConfig()
    for proj in ("q_proj", "v_proj"):
        assert proj in cfg.lora_target_modules


def test_baseline_raises_if_not_fit():
    model = LLMBaseline()
    with pytest.raises(RuntimeError):
        model.predict_proba(["dummy prompt"])


def test_config_overrides_are_respected():
    cfg = LLMConfig(model_name="meta-llama/Llama-3.2-3B-Instruct", epochs=1)
    assert cfg.model_name == "meta-llama/Llama-3.2-3B-Instruct"
    assert cfg.epochs == 1
