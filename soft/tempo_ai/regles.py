"""Règles RTE de placement des jours Tempo.

Blancs (et rouges jusqu'à l'hiver 2025-26) : algorithme de 2014, fiche RTE « Méthode de choix des
jours Tempo », indice 2 du 07/01/2025 (docs/methode_tempo_rte_2025-01.pdf).
Rouges depuis le 1er novembre 2026 : jours PP du mécanisme de capacité (délibération CRE 2026-189,
note RTE du 16/09/2026). Les seuils PP de l'hiver 2026-27 n'étant pas publiés, les rouges sont
approchés par l'algorithme de 2014, encadré par le tunnel mensuel de la délibération.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GAMMA = -0.1176  # sensibilité de la consommation à la température
K_TEMP = 8.3042  # °C, moyenne du quantile 30 % de la température journalière

STOCK_ROUGE, STOCK_BLANC = 22, 43


def normaliser(cnet: pd.Series, temp: pd.Series | None = None, fenetre: int = 365) -> pd.Series:
    """Consommation nette normalisée sur un an glissant (jours précédents uniquement)."""
    hist = cnet.shift(1).rolling(fenetre, min_periods=300)
    q40, q80 = hist.quantile(0.4), hist.quantile(0.8)
    correction = 1.0
    if temp is not None:
        qt30 = temp.shift(1).rolling(fenetre, min_periods=300).quantile(0.3)
        correction = np.exp(-GAMMA * (qt30 - K_TEMP))
    return (cnet - q40) / ((q80 - q40) * correction)


def seuils(jour_tempo, stock_blanc, stock_rouge):
    s_br = 4.0 - 0.015 * jour_tempo - 0.026 * (stock_blanc + stock_rouge)
    s_r = 3.15 - 0.010 * jour_tempo - 0.031 * stock_rouge
    return s_br, s_r


def rouge_autorise(d: pd.Timestamp, ferie: bool, saison: int) -> bool:
    if d.day_of_week >= 5 or ferie:
        return False
    if not (d.month in (11, 12, 1, 2, 3)):
        return False
    if saison >= 2026 and d.month == 12 and d.day in (24, 31):
        return False
    return True


def blanc_autorise(d: pd.Timestamp) -> bool:
    return d.day_of_week != 6


def jours_rouges_eligibles_restants(d: pd.Timestamp, feries: set, saison: int) -> int:
    fin = pd.Timestamp(f"{saison + 1}-03-31")
    jours = pd.date_range(d, fin)
    return sum(rouge_autorise(x, x.date() in feries, saison) for x in jours)


def couleur_2014(
    d: pd.Timestamp, cstd: float, stock_blanc: int, stock_rouge: int, jour_tempo: int,
    ferie: bool, saison: int, rouges_consecutifs: int, feries: set, apurement: bool = True,
    rouge_permis: bool = True,
) -> str:
    """Couleur d'un jour selon l'algorithme de 2014, stocks connus en début de journée."""
    s_br, s_r = seuils(jour_tempo, stock_blanc, stock_rouge)
    eligible = rouge_permis and stock_rouge > 0 and rouge_autorise(d, ferie, saison)
    if eligible and apurement and stock_rouge >= jours_rouges_eligibles_restants(d, feries, saison):
        return "Rouge"  # apurement de fin d'hiver : la limite de 5 rouges d'affilée ne s'applique plus
    ok_rouge = eligible and rouges_consecutifs < 5
    if np.isnan(cstd):
        return "?"
    if ok_rouge and cstd > s_r:
        return "Rouge"
    if stock_blanc > 0 and blanc_autorise(d) and cstd > s_br:
        return "Blanc"
    return "Bleu"


# Tunnel mensuel depuis 2026-27 : nombre cumulé de rouges (min, max) à la fin de chaque mois.
TUNNEL = {11: (0, 3), 12: (6, 6), 1: (13, 19), 2: (20, 22), 3: (22, 22)}


def _eligibles_mois_restants(d: pd.Timestamp, feries: set, saison: int) -> int:
    fin = d + pd.offsets.MonthEnd(0)
    return sum(rouge_autorise(x, x.date() in feries, saison) for x in pd.date_range(d, fin))


def couleur(
    d: pd.Timestamp, cstd: float, stock_blanc: int, stock_rouge: int, jour_tempo: int,
    ferie: bool, saison: int, rouges_consecutifs: int, feries: set,
) -> str:
    """Couleur selon la règle en vigueur pour la saison.

    Jusqu'en 2025-26 : algorithme de 2014. Depuis 2026-27 : blancs selon 2014 ; rouges approchés par
    l'algorithme de 2014 (seuils PP non publiés), bornés par le tunnel mensuel CRE 2026-189.
    """
    args = (d, cstd, stock_blanc, stock_rouge, jour_tempo, ferie, saison, rouges_consecutifs, feries)
    if saison < 2026 or d.month not in TUNNEL:
        return couleur_2014(*args)
    cumul = STOCK_ROUGE - stock_rouge
    mini, maxi = TUNNEL[d.month]
    eligible = stock_rouge > 0 and rouge_autorise(d, ferie, saison)
    if eligible and 0 < mini - cumul >= _eligibles_mois_restants(d, feries, saison):
        return "Rouge"  # apurement mensuel : minimum du tunnel à atteindre
    return couleur_2014(*args, rouge_permis=cumul < maxi)


def stocks_reels(couleurs: pd.Series) -> pd.DataFrame:
    """Stocks restants en début de chaque journée, et rouges consécutifs, d'après l'historique."""
    out = []
    for saison, s in couleurs.groupby(lambda d: d.year if d.month >= 9 else d.year - 1):
        sb, sr, consec = STOCK_BLANC, STOCK_ROUGE, 0
        for d, c in s.sort_index().items():
            out.append((d, sb, sr, consec))
            if c == "Blanc":
                sb -= 1
            if c == "Rouge":
                sr -= 1
            consec = consec + 1 if c == "Rouge" else 0  # le week-end (jamais rouge) coupe la série
    return pd.DataFrame(out, columns=["date", "stock_blanc", "stock_rouge", "rouges_consec"]).set_index("date")
