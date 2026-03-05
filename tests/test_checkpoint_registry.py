from __future__ import annotations

import json
from pathlib import Path

import pytest

from checkpoint_registry import (
    load_registry,
    resolve_checkpoint_for_profile,
    validate_checkpoint_dir,
)


@pytest.mark.parametrize(
    ("profile", "prompt_text"),
    [
        ("telecom", "need telecom dataset"),
        ("general", "need telecom customer dataset"),
    ],
)
def test_resolve_uses_telecom_mapping(tmp_path: Path, profile: str, prompt_text: str) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(
            {
                "general": "checkpoints/general",
                "telecom": "checkpoints/telecom",
                "_default": "checkpoints/default",
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(str(reg_path))
    ckpt = resolve_checkpoint_for_profile(profile, registry, prompt_text=prompt_text)
    assert ckpt == "checkpoints/telecom"


def test_resolve_uses_registry_key_not_in_domain_keywords(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(
            {
                "general": "checkpoints/general",
                "cybersecurity": "checkpoints/cyber",
                "_default": "checkpoints/default",
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(str(reg_path))
    ckpt = resolve_checkpoint_for_profile("general", registry, prompt_text="generate cybersecurity dataset")
    assert ckpt == "checkpoints/cyber"


def test_resolve_checkpoint_direct_mapping(tmp_path: Path) -> None:
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(json.dumps({"finance_v1": "checkpoints/finance"}), encoding="utf-8")
    registry = load_registry(str(reg_path))
    ckpt = resolve_checkpoint_for_profile("finance_v1", registry)
    assert ckpt == "checkpoints/finance"


def test_validate_checkpoint_dir_missing(tmp_path: Path) -> None:
    missing = validate_checkpoint_dir(str(tmp_path / "missing_dir"))
    assert "metadata.json" in missing
