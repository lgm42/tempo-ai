"""Règles RTE : seuils, contraintes de calendrier, apurement, tunnel mensuel 2026-27."""

from __future__ import annotations

import math

import holidays
import pandas as pd
import pytest

from tempo_ai.regles import (
    STOCK_BLANC, STOCK_ROUGE, blanc_autorise, couleur, couleur_2014, rouge_autorise, seuils, stocks_reels,
)

FERIES = set(holidays.France(years=range(2024, 2028)).keys())
HAUT, MILIEU, BAS = 5.0, 0.7, -1.0  # le 14/01/2025 : seuil blanc 0,27, seuil rouge 1,11


def _c(fonction, jour: str, cstd: float, sb: int = STOCK_BLANC, sr: int = STOCK_ROUGE, consec: int = 0):
    d = pd.Timestamp(jour)
    saison = d.year if d.month >= 9 else d.year - 1
    jour_tempo = (d - pd.Timestamp(f"{saison}-09-01")).days + 1
    return fonction(d, cstd, sb, sr, jour_tempo, d.date() in FERIES, saison, consec, FERIES)


# ------------------------------------------------------------------ seuils et calendrier
def test_seuils_debut_de_saison():
    s_blanc, s_rouge = seuils(1, STOCK_BLANC, STOCK_ROUGE)
    assert s_blanc == pytest.approx(4.0 - 0.015 - 0.026 * 65)
    assert s_rouge == pytest.approx(3.15 - 0.010 - 0.031 * 22)


def test_seuils_montent_quand_les_stocks_baissent():
    s_blanc, s_rouge = seuils(100, 10, 5)
    assert s_blanc > seuils(100, 43, 22)[0] and s_rouge > seuils(100, 43, 22)[1]


@pytest.mark.parametrize("jour, attendu", [
    ("2025-01-14", True),   # mardi de janvier
    ("2025-01-18", False),  # samedi
    ("2025-01-19", False),  # dimanche
    ("2024-11-11", False),  # férié
    ("2024-10-15", False),  # hors novembre-mars
    ("2025-04-01", False),
])
def test_rouge_autorise(jour, attendu):
    d = pd.Timestamp(jour)
    assert rouge_autorise(d, d.date() in FERIES, 2024) is attendu


def test_24_et_31_decembre_exclus_depuis_2026():
    assert rouge_autorise(pd.Timestamp("2025-12-24"), False, 2025)
    assert not rouge_autorise(pd.Timestamp("2026-12-24"), False, 2026)
    assert not rouge_autorise(pd.Timestamp("2026-12-31"), False, 2026)


def test_jamais_blanc_le_dimanche():
    assert not blanc_autorise(pd.Timestamp("2025-01-19"))
    assert blanc_autorise(pd.Timestamp("2025-01-18"))


# ------------------------------------------------------------------ algorithme 2014
def test_couleurs_2014_selon_le_score():
    assert _c(couleur_2014, "2025-01-14", HAUT) == "Rouge"
    assert _c(couleur_2014, "2025-01-14", MILIEU) == "Blanc"
    assert _c(couleur_2014, "2025-01-14", BAS) == "Bleu"


def test_score_inconnu():
    assert _c(couleur_2014, "2025-01-14", math.nan) == "?"


def test_pas_de_rouge_le_week_end_ni_de_blanc_le_dimanche():
    assert _c(couleur_2014, "2025-01-18", HAUT) == "Blanc"
    assert _c(couleur_2014, "2025-01-19", HAUT) == "Bleu"


def test_cinq_rouges_consecutifs_au_plus():
    assert _c(couleur_2014, "2025-01-14", HAUT, consec=4) == "Rouge"
    assert _c(couleur_2014, "2025-01-14", HAUT, consec=5) == "Blanc"


def test_stocks_epuises():
    assert _c(couleur_2014, "2025-01-14", HAUT, sr=0) == "Blanc"
    assert _c(couleur_2014, "2025-01-14", HAUT, sb=0, sr=0) == "Bleu"


def test_apurement_de_fin_d_hiver():
    # du mardi 25 au lundi 31 mars 2025 : 5 jours rouges possibles, il en reste 5 à placer
    assert _c(couleur_2014, "2025-03-25", BAS, sr=5) == "Rouge"
    assert _c(couleur_2014, "2025-03-25", BAS, sr=4) == "Bleu"
    # l'apurement passe outre la limite de 5 rouges d'affilée
    assert _c(couleur_2014, "2025-03-25", BAS, sr=5, consec=5) == "Rouge"


# ------------------------------------------------------------------ règle en vigueur
def test_avant_2026_la_regle_2014_s_applique():
    for score in (HAUT, MILIEU, BAS):
        assert _c(couleur, "2025-01-14", score) == _c(couleur_2014, "2025-01-14", score)


def test_tunnel_maximum_de_novembre():
    assert _c(couleur, "2026-11-10", HAUT, sr=STOCK_ROUGE - 2) == "Rouge"
    assert _c(couleur, "2026-11-10", HAUT, sr=STOCK_ROUGE - 3) == "Blanc"  # 3 rouges déjà placés


def test_tunnel_minimum_force_le_rouge():
    # lundi 28 décembre 2026 : restent 28, 29, 30 (le 31 est exclu) ; minimum 6 rouges fin décembre
    assert _c(couleur, "2026-12-28", BAS, sr=STOCK_ROUGE - 3) == "Rouge"  # 3 manquants, 3 jours
    assert _c(couleur, "2026-12-28", BAS, sr=STOCK_ROUGE - 4) == "Bleu"   # 2 manquants, 3 jours
    assert _c(couleur, "2026-12-28", BAS, sr=STOCK_ROUGE - 6) == "Bleu"   # minimum atteint


def test_veille_de_noel_jamais_rouge_depuis_2026():
    assert _c(couleur, "2026-12-24", HAUT, sr=STOCK_ROUGE - 5) != "Rouge"


# ------------------------------------------------------------------ stocks
def test_stocks_reels():
    jours = pd.to_datetime(["2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-09-01"])
    s = stocks_reels(pd.Series(["Rouge", "Rouge", "Blanc", "Bleu", "Bleu"], index=jours))
    assert s.loc["2025-01-13"].tolist() == [43, 22, 0]
    assert s.loc["2025-01-15"].tolist() == [43, 20, 2]
    assert s.loc["2025-01-16"].tolist() == [42, 20, 0]
    assert s.loc["2025-09-01"].tolist() == [43, 22, 0]  # nouvelle saison
