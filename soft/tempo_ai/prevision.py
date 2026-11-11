"""Prévision des couleurs de J+1 (si pas encore publiée) à J+4, à partir des données du jour."""

from __future__ import annotations

from datetime import datetime

import holidays
import numpy as np
import pandas as pd

from . import config
from .collecte import lire_meteo_prevue
from .donnees import table_journaliere
from .modele import ModeleCnet, variables
from .regles import GAMMA, K_TEMP, STOCK_BLANC, STOCK_ROUGE, couleur, seuils, stocks_reels

# Erreur relative typique sur la consommation nette (écart-type), mesurée au rejeu avec la météo
# prévue (erreur moyenne 3,9 / 4,5 / 4,8 % à J+2 / J+3 / J+4, × 1,25 pour un écart-type).
SIGMA = {1: 0.04, 2: 0.049, 3: 0.056, 4: 0.06}
TIRAGES = 400
CODES = {"Bleu": "blue", "Blanc": "white", "Rouge": "red"}


def _saison(d: pd.Timestamp) -> int:
    return d.year if d.month >= 9 else d.year - 1


def prevoir(modeles: dict[int, ModeleCnet], meteo_prevue_fichier, aujourd_hui: pd.Timestamp | None = None) -> dict:
    J = aujourd_hui or pd.Timestamp(datetime.now(config.TZ_PARIS).date())
    t = table_journaliere(lire_meteo_prevue(meteo_prevue_fichier))
    feries = set(holidays.France(years=range(J.year - 1, J.year + 2)).keys())

    # normalisation RTE : quantiles de l'année écoulée, connus le jour J
    q = t["cnet"].shift(1).rolling(365, min_periods=300)
    q40, q80 = q.quantile(0.4).at[J], q.quantile(0.8).at[J]
    qt30 = t["temp"].shift(1).rolling(365, min_periods=300).quantile(0.3).at[J]
    corr = float(np.exp(-GAMMA * (qt30 - K_TEMP)))

    # stocks après le dernier jour dont la couleur est publiée
    couleurs = t["couleur"].dropna()
    dernier = couleurs.index.max()
    st = stocks_reels(couleurs).loc[dernier]
    sb, sr, consec = int(st.stock_blanc), int(st.stock_rouge), int(st.rouges_consec)
    c = couleurs.at[dernier]
    sb, sr = sb - (c == "Blanc"), sr - (c == "Rouge")
    consec = consec + 1 if c == "Rouge" else 0

    rng = np.random.default_rng(int(J.strftime("%Y%m%d")))
    jours = []
    for h in [1, 2, 3, 4]:
        D = J + pd.Timedelta(days=h)
        if _saison(D) != _saison(dernier) and D > dernier:
            sb, sr, consec = STOCK_BLANC, STOCK_ROUGE, 0  # nouvelle saison au 1er septembre
        r = t.loc[D]
        args = (int(r.jour_tempo), bool(r.ferie), int(r.saison), consec, feries)
        s_blanc, s_rouge = seuils(int(r.jour_tempo), sb, sr)
        jour = dict(date=D.date().isoformat(), horizon=h, white_left_before=sb, red_left_before=sr,
                    threshold_white=round(float(s_blanc), 3), threshold_red=round(float(s_rouge), 3))
        if D <= dernier:
            col = couleurs.at[D]
            jour.update(source="rte", color=CODES[col],
                        probabilities={v: float(k == col) for k, v in CODES.items()})
        else:
            x = variables(t, h).loc[[D]]
            p = modeles[h].predict(x).iloc[0]
            if np.isnan(p.cnet):
                raise RuntimeError(f"données insuffisantes pour prévoir le {D.date()} (météo ou éCO2mix manquants)")
            score = (p.cnet - q40) / ((q80 - q40) * corr)
            col = couleur(D, score, sb, sr, *args)
            tirages = p.cnet * (1 + rng.normal(0, SIGMA[h], TIRAGES))
            ech = [couleur(D, (v - q40) / ((q80 - q40) * corr), sb, sr, *args) for v in tirages]
            jour.update(
                source="model", color=CODES[col],
                probabilities={v: round(ech.count(k) / TIRAGES, 3) for k, v in CODES.items()},
                net_load_mw=round(float(p.cnet)), peak_residual_mw=round(float(p.pointe_resid)),
                consumption_mw=round(float(p.conso)), wind_mw=round(float(p.eolien)),
                solar_mw=round(float(p.solaire)), score=round(float(score), 3),
            )
        jours.append(jour)
        sb -= col == "Blanc"
        sr -= col == "Rouge"
        consec = consec + 1 if col == "Rouge" else 0

    aujourd = couleurs.get(J)
    return dict(
        issued_at=datetime.now(config.TZ_PARIS).isoformat(timespec="seconds"),
        issued_for=J.date().isoformat(),
        today=dict(date=J.date().isoformat(), color=CODES.get(aujourd)),
        days=jours,
    )
