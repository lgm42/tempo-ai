"""Modèle de consommation nette, décomposé en trois parties apprises séparément.

  consommation (moyenne 6h-6h et pointe horaire)  ← température, calendrier, niveau récent
  facteur de charge éolien (production / parc)    ← vent à 100 m sur les zones éoliennes
  facteur de charge solaire                       ← rayonnement, saison
puis  cnet = conso − fc_eol × parc_eol − fc_sol × parc_sol
et    pointe résiduelle ≈ pointe de conso − éolien moyen du jour.

Le parc (éolien, solaire) n'est pas appris par les arbres, qui n'extrapolent pas une croissance :
on utilise le max horaire de l'année écoulée, connu au moment de la prévision.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

PARAMS = dict(
    objective="regression_l1", learning_rate=0.03, n_estimators=1200, num_leaves=31,
    min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
)

CAL = ["dow", "ferie", "jour_tempo", "doy", "pont", "vacances_noel"]
TEMP = ["temp", "temp_min", "temp_soir", "temp_veille", "temp_avant_veille", "ray"]


def colonnes_vent(t: pd.DataFrame) -> list[str]:
    return ["vent", "vent3", "p_moy", "p_moy_soir"] + [c for c in t.columns if c.startswith(("v_eol", "p_eol"))]


def colonnes_meteo(t: pd.DataFrame) -> list[str]:
    return ["temp", "temp_min", "temp_soir", "ray"] + colonnes_vent(t)


def variables(t: pd.DataFrame, h: int, t_obs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Variables pour le jour D prévu depuis J = D − h. Le réalisé est connu jusqu'à J−1 inclus.

    t : table dont la météo peut être prévue ; t_obs : table observée pour l'ancrage (par défaut t).
    """
    t_obs = t if t_obs is None else t_obs
    x = pd.DataFrame(index=t.index)
    x["dow"], x["jour_tempo"], x["doy"] = t["dow"], t["jour_tempo"], t["doy"]
    x["ferie"] = t["ferie"].astype(float)
    ferie = t["ferie"].fillna(False).astype(bool)
    x["pont"] = (((t["dow"] == 0) & ferie.shift(-1, fill_value=False)) | ((t["dow"] == 4) & ferie.shift(1, fill_value=False))).astype(int)
    m, d = t.index.month, t.index.day
    x["vacances_noel"] = (((m == 12) & (d >= 22)) | ((m == 1) & (d <= 3))).astype(int)
    for c in colonnes_meteo(t):
        x[c] = t[c]
    x["temp_veille"] = t["temp"].shift(1)
    x["temp_avant_veille"] = t["temp"].shift(2)
    # ancrage : niveau de consommation récent, ramené à la température de l'époque
    decal = h + 1
    x["conso_7j"] = t_obs["conso"].shift(decal).rolling(7, min_periods=5).mean()
    x["pointe_conso_7j"] = t_obs["pointe_conso"].shift(decal).rolling(7, min_periods=5).mean()
    x["temp_7j"] = t_obs["temp"].shift(decal).rolling(7, min_periods=5).mean()
    x["dtemp"] = x["temp"] - x["temp_7j"]
    x["dtemp_soir"] = x["temp_soir"] - x["temp_7j"]
    x["cap_eol"], x["cap_sol"] = t_obs["cap_eol"].shift(h), t_obs["cap_sol"].shift(h)
    return x


def _fit(x: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
    ok = y.notna() & x.notna().all(axis=1)
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(x[ok], y[ok])
    return m


# Consommation : on apprend l'écart au niveau des 7 derniers jours (robuste à la baisse depuis 2022).
ANCRE = {"conso": "conso_7j", "pointe_conso": "pointe_conso_7j"}

FEATURES = {
    "conso": lambda t: CAL + TEMP + ["conso_7j", "temp_7j", "dtemp", "dtemp_soir"],
    "pointe_conso": lambda t: CAL + TEMP + ["pointe_conso_7j", "temp_7j", "dtemp", "dtemp_soir"],
    "fc_eol": lambda t: ["doy"] + colonnes_vent(t),
    "fc_sol": lambda t: ["doy", "ray", "temp"],
}


class ModeleCnet:
    def __init__(self) -> None:
        self.m: dict[str, lgb.LGBMRegressor | lgb.Booster] = {}
        self.feats: dict[str, list[str]] = {}

    def fit(self, x: pd.DataFrame, t: pd.DataFrame) -> "ModeleCnet":
        for cible, feats in FEATURES.items():
            y = t[cible] - x[ANCRE[cible]] if cible in ANCRE else t[cible]
            self.m[cible] = _fit(x[feats(t)], y)
        self.feats = {c: f(t) for c, f in FEATURES.items()}
        return self

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        p = pd.DataFrame(index=x.index)
        for cible, m in self.m.items():
            p[cible] = m.predict(x[self.feats[cible]].astype(float))
            if cible in ANCRE:
                p[cible] += x[ANCRE[cible]]
        p["eolien"] = p["fc_eol"].clip(0, 1) * x["cap_eol"]
        p["solaire"] = p["fc_sol"].clip(0, 1) * x["cap_sol"]
        p["cnet"] = p["conso"] - p["eolien"] - p["solaire"]
        p["pointe_resid"] = p["pointe_conso"] - p["eolien"]
        return p

    def sauver(self, dossier: Path) -> None:
        dossier.mkdir(parents=True, exist_ok=True)
        for cible, m in self.m.items():
            booster = m.booster_ if isinstance(m, lgb.LGBMRegressor) else m
            booster.save_model(str(dossier / f"{cible}.txt"))
        (dossier / "variables.json").write_text(json.dumps(self.feats, indent=1), encoding="utf-8")

    @classmethod
    def charger(cls, dossier: Path) -> "ModeleCnet":
        self = cls()
        self.feats = json.loads((dossier / "variables.json").read_text(encoding="utf-8"))
        self.m = {c: lgb.Booster(model_file=str(dossier / f"{c}.txt")) for c in self.feats}
        return self
