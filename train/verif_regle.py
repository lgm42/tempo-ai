"""Fidélité de l'algorithme de 2014 : appliqué à la consommation nette RÉELLE, retrouve-t-il les couleurs RTE ?

Les stocks sont ceux de l'historique réel (pas de dérive). C'est le plafond atteignable
avec une prévision parfaite de la consommation.
"""

from __future__ import annotations

import sys

import _chemins  # noqa: F401
import holidays
import pandas as pd

from tempo_ai.donnees import table_journaliere
from tempo_ai.regles import couleur_2014, normaliser, stocks_reels


def rejouer(t: pd.DataFrame, col_cstd: str = "cstd") -> pd.DataFrame:
    st = stocks_reels(t["couleur"].dropna())
    t = t.join(st)
    feries = set(holidays.France(years=range(2014, 2030)).keys())
    res = []
    for d, r in t.dropna(subset=["couleur", "stock_rouge"]).iterrows():
        c = couleur_2014(
            d, r[col_cstd], int(r.stock_blanc), int(r.stock_rouge), int(r.jour_tempo),
            bool(r.ferie), int(r.saison), int(r.rouges_consec), feries,
        )
        res.append((d, r.couleur, c))
    return pd.DataFrame(res, columns=["date", "vrai", "regle"]).set_index("date")


def bilan(res: pd.DataFrame, titre: str) -> None:
    print(f"\n=== {titre}")
    res = res.copy()
    res["saison"] = [d.year if d.month >= 9 else d.year - 1 for d in res.index]
    hiver = res[res.index.month.isin([11, 12, 1, 2, 3])]
    for s, g in hiver.groupby("saison"):
        ok = (g.vrai == g.regle).mean()
        rouge_v = g.vrai == "Rouge"
        rouge_p = g.regle == "Rouge"
        print(
            f"{s}-{s+1}  hiver ok {ok:5.1%} | rouges vrais {rouge_v.sum():2d} trouvés {(rouge_v & rouge_p).sum():2d}"
            f" (fausses alertes {(rouge_p & ~rouge_v).sum():2d}) | blancs vrais {(g.vrai=='Blanc').sum():2d}"
            f" trouvés {((g.vrai=='Blanc') & (g.regle=='Blanc')).sum():2d}"
        )
    print(pd.crosstab(hiver.vrai, hiver.regle, rownames=["vrai"], colnames=["règle"]))


if __name__ == "__main__":
    t = table_journaliere()
    t["cstd_brut"] = normaliser(t["cnet"])
    bilan(rejouer(t, "cstd_brut"), "normalisation sans température")
    if "temp" in t and t["temp"].notna().sum() > 500:
        t["cstd"] = normaliser(t["cnet"], t["temp"])
        bilan(rejouer(t, "cstd"), "normalisation avec température (fiche RTE)")
