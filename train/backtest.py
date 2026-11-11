"""Rejeu « comme si on y était » des hivers 2024-25 et 2025-26 (+ début 2026-27).

Chaque jour J à midi (couleur de J+1 déjà publiée), on prévoit J+2, J+3 et J+4 :
  1. modèle de consommation nette (modele.py), entraîné uniquement sur les données antérieures
     à la saison testée, avec la météo observée ;
  2. en rejeu, il reçoit la météo PRÉVUE à l'époque (Open-Meteo previous runs, au délai h) ;
  3. la règle RTE (2014, + tunnel mensuel depuis 2026-27) transforme la consommation prévue en couleur, en enchaînant les stocks
     (J+1 réel → J+2 prévu → J+3 prévu → J+4).
Références : persistance (J+h = couleur de J+1) et règle appliquée à la consommation réelle (plafond).

Usage : python backtest.py            (rejeu complet, ~quelques minutes)
        python backtest.py rapport    (réaffiche les scores depuis resultats/rejeu.csv)
"""

from __future__ import annotations

import sys
from pathlib import Path

import _chemins  # noqa: F401  (soft/ dans le chemin, cache dans train/data)
import holidays
import numpy as np
import pandas as pd

from tempo_ai.donnees import agreger_meteo, meteo_lieux, table_journaliere
from tempo_ai.modele import ModeleCnet, colonnes_meteo, variables
from tempo_ai.regles import GAMMA, K_TEMP, couleur, normaliser, stocks_reels

HORIZONS = [2, 3, 4]
SAISONS_TEST = [2024, 2025, 2026]
SORTIE = Path(__file__).parent / "resultats"
SORTIE.mkdir(exist_ok=True)


def table_meteo_prevue(t: pd.DataFrame, h: int) -> pd.DataFrame:
    """Copie de t où la météo est remplacée par celle prévue h jours avant (là où elle existe)."""
    mp = agreger_meteo(meteo_lieux("previsions"), suffixe=f"_previous_day{h}")
    cols = [c for c in colonnes_meteo(t) if c in mp]
    th = t.copy()
    idx = mp.index.intersection(th.index)
    th[cols] = np.nan  # hors périodes de prévisions : pas de rejeu
    th.loc[idx, cols] = mp.loc[idx, cols]
    return th


def main() -> None:
    t = table_journaliere()
    feries = set(holidays.France(years=range(2014, 2030)).keys())
    t = t.join(stocks_reels(t["couleur"].dropna()))
    t["cstd_reel"] = normaliser(t["cnet"], t["temp"])
    q = t["cnet"].shift(1).rolling(365, min_periods=300)
    q40, q80 = q.quantile(0.4), q.quantile(0.8)
    qt30 = t["temp"].shift(1).rolling(365, min_periods=300).quantile(0.3)
    corr = np.exp(-GAMMA * (qt30 - K_TEMP))

    tables_prevues = {h: table_meteo_prevue(t, h) for h in HORIZONS}
    lignes = []
    for saison in SAISONS_TEST:
        debut, fin = pd.Timestamp(f"{saison}-09-01"), pd.Timestamp(f"{saison + 1}-08-31")
        pred = {}
        for h in HORIZONS:
            x = variables(t, h)
            modele = ModeleCnet().fit(x[x.index < debut], t[x.index < debut])
            xp = variables(tables_prevues[h], h, t_obs=t)
            xp = xp[(xp.index >= debut) & (xp.index <= fin)].dropna()
            pred[h] = modele.predict(xp)
            print(f"saison {saison}-{saison + 1} J+{h}: {len(xp)} jours prévisibles")

        for J in t.index[(t.index >= debut - pd.Timedelta(days=1)) & (t.index <= fin)]:
            j1 = J + pd.Timedelta(days=1)
            if j1 not in t.index or pd.isna(t.at[j1, "couleur"]) or pd.isna(t.at[j1, "stock_rouge"]):
                continue
            # stocks après J+1 (couleur réelle publiée à J 10h30)
            sb, sr, consec = int(t.at[j1, "stock_blanc"]), int(t.at[j1, "stock_rouge"]), int(t.at[j1, "rouges_consec"])
            c1 = t.at[j1, "couleur"]
            sb -= c1 == "Blanc"
            sr -= c1 == "Rouge"
            consec = consec + 1 if c1 == "Rouge" else 0
            for h in HORIZONS:
                D = J + pd.Timedelta(days=h)
                if D not in pred[h].index:
                    break
                p = pred[h].loc[D]
                r = t.loc[D]
                cstd_p = (p.cnet - q40.at[J]) / ((q80.at[J] - q40.at[J]) * corr.at[J])
                args = (int(r.jour_tempo), bool(r.ferie), saison, consec, feries)
                coul_p = couleur(D, cstd_p, sb, sr, *args)
                coul_plafond = couleur(D, r.cstd_reel, sb, sr, *args)
                lignes.append(
                    dict(
                        emis=J, jour=D, horizon=h, vrai=r.couleur, prevu=coul_p, persistance=c1,
                        plafond=coul_plafond, cnet_prevu=p.cnet, cnet_reel=r.cnet,
                        pointe_prevue=p.pointe_resid, pointe_reelle=r.pointe_resid,
                        eolien_prevu=p.eolien, eolien_reel=r.eolien,
                        cstd_prevu=cstd_p, cstd_reel=r.cstd_reel,
                        stock_blanc=sb, stock_rouge=sr,
                    )
                )
                sb -= coul_p == "Blanc"
                sr -= coul_p == "Rouge"
                consec = consec + 1 if coul_p == "Rouge" else 0

    p = pd.DataFrame(lignes)
    p.to_csv(SORTIE / "rejeu.csv", index=False)
    rapport(p)


def _scores(g: pd.DataFrame, col: str) -> dict:
    vr, pr = g.vrai == "Rouge", g[col] == "Rouge"
    vb, pb = g.vrai == "Blanc", g[col] == "Blanc"
    return dict(
        ok=(g[col] == g.vrai).mean(),
        rouge_rappel=(vr & pr).sum() / max(vr.sum(), 1),
        rouge_precision=(vr & pr).sum() / max(pr.sum(), 1),
        blanc_rappel=(vb & pb).sum() / max(vb.sum(), 1),
        blanc_precision=(vb & pb).sum() / max(pb.sum(), 1),
    )


def rapport(p: pd.DataFrame) -> None:
    p = p.dropna(subset=["vrai"]).copy()
    p["jour"] = pd.to_datetime(p["jour"])
    p["saison"] = [d.year if d.month >= 9 else d.year - 1 for d in p.jour]
    hiver = p[p.jour.dt.month.isin([11, 12, 1, 2, 3])]
    lignes = []
    groupes = list(hiver.groupby(["saison", "horizon"])) + [(("tout", h), g) for h, g in hiver.groupby("horizon")]
    for (s, h), g in groupes:
        lignes.append(
            dict(
                saison=s if s == "tout" else f"{s}-{s + 1}", horizon=f"J+{h}", jours=len(g),
                **{f"modele_{k}": v for k, v in _scores(g, "prevu").items()},
                **{f"persist_{k}": v for k, v in _scores(g, "persistance").items()},
                **{f"plafond_{k}": v for k, v in _scores(g, "plafond").items()},
                err_cnet=((g.cnet_prevu - g.cnet_reel).abs() / g.cnet_reel).mean(),
                err_pointe=((g.pointe_prevue - g.pointe_reelle).abs() / g.pointe_reelle).mean(),
                err_eolien_MW=(g.eolien_prevu - g.eolien_reel).abs().mean(),
            )
        )
    r = pd.DataFrame(lignes)
    r.to_csv(SORTIE / "scores.csv", index=False)
    pd.set_option("display.width", 250)
    pct = lambda v: f"{v:.0%}" if isinstance(v, float) and v <= 1.5 else (f"{v:.0f}" if isinstance(v, float) else v)  # noqa: E731
    print("\nHiver (novembre → mars) :")
    cols = ["saison", "horizon", "jours", "modele_ok", "persist_ok", "plafond_ok",
            "modele_rouge_rappel", "modele_rouge_precision", "persist_rouge_rappel",
            "modele_blanc_rappel", "modele_blanc_precision", "err_cnet", "err_pointe", "err_eolien_MW"]
    print(r[cols].map(pct).to_string(index=False))
    for h, g in hiver.groupby("horizon"):
        print(f"\nJ+{h}, tous hivers : vraie couleur (lignes) × prévue (colonnes)")
        print(pd.crosstab(g.vrai, g.prevu))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rapport":
        rapport(pd.read_csv(SORTIE / "rejeu.csv"))
    else:
        main()
