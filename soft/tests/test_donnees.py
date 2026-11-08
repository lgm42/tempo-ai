"""Calendrier Tempo : jour de 6 h à 6 h (heure de Paris), saison au 1er septembre."""

from __future__ import annotations

import pandas as pd

from tempo_ai.donnees import _jour_tempo, calendrier


def test_jour_tempo_de_6h_a_6h():
    instants = pd.DatetimeIndex(["2026-11-10 04:59", "2026-11-10 05:00", "2026-07-10 03:59", "2026-07-10 04:00"],
                                tz="UTC")  # 5 h 59 / 6 h à Paris, en hiver puis en été
    assert [d.date().isoformat() for d in _jour_tempo(instants)] == [
        "2026-11-09", "2026-11-10", "2026-07-09", "2026-07-10"]


def test_calendrier():
    c = calendrier(pd.to_datetime(["2026-08-31", "2026-09-01", "2026-11-11", "2027-03-31"]))
    assert c["saison"].tolist() == [2025, 2026, 2026, 2026]
    assert c["jour_tempo"].tolist() == [365, 1, 72, 212]
    assert c["ferie"].tolist() == [False, False, True, False]
