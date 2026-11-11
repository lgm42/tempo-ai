"""Fabriques de prévisions au format de prevision.prevoir."""

from __future__ import annotations


def jour_prevu(cible: str, horizon: int, couleur: str, source: str = "model") -> dict:
    p = dict(blue=0.0, white=0.0, red=0.0)
    p[couleur] = 1.0
    return dict(
        date=cible, horizon=horizon, source=source, color=couleur, probabilities=p,
        white_left_before=40, red_left_before=20, threshold_white=1.3, threshold_red=1.8,
        net_load_mw=50000, peak_residual_mw=60000, consumption_mw=55000, wind_mw=4000, solar_mw=1000,
        score=0.5,
    )


def prevision(emise_le: str, jours: list[dict]) -> dict:
    return dict(issued_at=f"{emise_le}T12:00:00+01:00", issued_for=emise_le,
                today=dict(date=emise_le, color="blue"), days=jours)
