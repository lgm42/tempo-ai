"""Prévision J+1 à J+4 : enchaînement des stocks, couleurs publiées, changement de saison.

La table journalière et les modèles sont remplacés par des bouchons : on teste la logique, pas LightGBM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempo_ai import prevision
from tempo_ai.donnees import calendrier

BLANCS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-11-02", "2026-11-03"]
ROUGES = ["2026-01-08", "2026-11-04"]


def _table(dernier_publie: str) -> pd.DataFrame:
    t = calendrier(pd.date_range("2025-06-01", "2026-11-20"))
    angle = 2 * np.pi * (t["doy"] - 15) / 365
    t["cnet"] = 50000 + 15000 * np.cos(angle)  # plus haut en hiver
    t["temp"] = 12 - 8 * np.cos(angle)
    t["couleur"] = "Bleu"
    t.loc[BLANCS, "couleur"] = "Blanc"
    t.loc[ROUGES, "couleur"] = "Rouge"
    t.loc[t.index > dernier_publie, "couleur"] = np.nan
    return t


class Bouchon:
    def __init__(self, cnet: float) -> None:
        self.cnet = cnet

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(dict(cnet=self.cnet, pointe_resid=self.cnet, conso=self.cnet, eolien=0.0, solaire=0.0),
                            index=x.index)


def _prevoir(monkeypatch, jour: str, dernier_publie: str, cnet: float) -> dict:
    t = _table(dernier_publie)
    monkeypatch.setattr(prevision, "lire_meteo_prevue", lambda _f: None)
    monkeypatch.setattr(prevision, "table_journaliere", lambda _m: t)
    monkeypatch.setattr(prevision, "variables", lambda table, _h: table)
    modeles = {h: Bouchon(cnet) for h in (1, 2, 3, 4)}
    return prevision.prevoir(modeles, None, pd.Timestamp(jour))


def test_jours_calmes(monkeypatch):
    p = _prevoir(monkeypatch, "2026-11-10", "2026-11-11", cnet=20000)
    assert p["issued_for"] == "2026-11-10"
    assert p["today"] == dict(date="2026-11-10", color="blue")
    j1, j2, j3, j4 = p["days"]
    assert (j1["horizon"], j1["source"], j1["color"]) == (1, "rte", "blue")  # J+1 déjà publié
    assert j1["probabilities"] == dict(blue=1.0, white=0.0, red=0.0)
    for j in (j2, j3, j4):
        assert (j["source"], j["color"]) == ("model", "blue")
        assert j["probabilities"]["blue"] == 1.0
        assert j["net_load_mw"] == 20000
    # saison 2026-27 : 2 blancs et 1 rouge déjà passés
    assert (j2["white_left_before"], j2["red_left_before"]) == (41, 21)


def test_stocks_enchaines_par_grand_froid(monkeypatch):
    p = _prevoir(monkeypatch, "2026-11-10", "2026-11-11", cnet=200000)
    _j1, j2, j3, j4 = p["days"]
    # jeudi 12 et vendredi 13 rouges (tunnel de novembre : 3 au plus), samedi 14 jamais rouge
    assert [j["color"] for j in (j2, j3, j4)] == ["red", "red", "white"]
    assert [j["red_left_before"] for j in (j2, j3, j4)] == [21, 20, 19]
    for j in (j2, j3, j4):
        assert sum(j["probabilities"].values()) == pytest.approx(1.0)


def test_nouvelle_saison_au_1er_septembre(monkeypatch):
    p = _prevoir(monkeypatch, "2026-08-29", "2026-08-30", cnet=20000)
    stocks = {j["date"]: (j["white_left_before"], j["red_left_before"]) for j in p["days"]}
    assert stocks["2026-08-31"] == (40, 21)  # fin de la saison 2025-26
    assert stocks["2026-09-01"] == (43, 22)
    assert stocks["2026-09-02"] == (43, 22)


def test_donnees_manquantes(monkeypatch):
    with pytest.raises(RuntimeError, match="données insuffisantes"):
        _prevoir(monkeypatch, "2026-11-10", "2026-11-11", cnet=np.nan)
