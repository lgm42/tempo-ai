"""Fixtures communes : chaque test travaille dans un dossier temporaire, sans réseau ni planificateur."""

from __future__ import annotations

import os

os.environ.setdefault("TEMPO_SCHEDULER", "0")

import pytest  # noqa: E402

from tempo_ai import base, config  # noqa: E402


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    base.initialiser()
    return tmp_path
