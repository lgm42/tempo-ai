"""Construction de la table journalière (une ligne par jour Tempo, de 6 h à 6 h, heure de Paris)."""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from . import config
from .lieux import LIEUX

TZ = "Europe/Paris"


def _jour_tempo(index_utc: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Jour Tempo d'un instant : de 6 h à 6 h le lendemain (heure de Paris)."""
    local = index_utc.tz_convert(TZ)
    return (local - pd.Timedelta(hours=6)).tz_localize(None).normalize()


def eco2mix_horaire() -> pd.DataFrame:
    df = pd.concat(
        [pd.read_parquet(config.DATA / "eco2mix_consolide.parquet"), pd.read_parquet(config.DATA / "eco2mix_temps_reel.parquet")]
    )
    df["t"] = pd.to_datetime(df["date_heure"], utc=True)
    df = df.drop(columns="date_heure").drop_duplicates("t").set_index("t").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    h = df.resample("1h").mean()  # pas de 15 ou 30 min → horaire
    h["cnet"] = h["consommation"] - h["eolien"].fillna(0) - h["solaire"].fillna(0)
    return h


def table_eco2mix() -> pd.DataFrame:
    h = eco2mix_horaire()
    h["jour"] = _jour_tempo(h.index)
    g = h.groupby("jour")
    j = pd.DataFrame(
        {
            "conso": g["consommation"].mean(),
            "eolien": g["eolien"].mean(),
            "solaire": g["solaire"].mean(),
            "cnet": g["cnet"].mean(),  # grandeur de l'algorithme de 2014 (blancs)
            "n_heures": g["cnet"].count(),
        }
    )
    # Pointe de consommation résiduelle (algorithme PP, rouges depuis 2026) : max horaire du jour civil.
    h["jour_civil"] = h.index.tz_convert(TZ).tz_localize(None).normalize()
    j["pointe_resid"] = h.groupby("jour_civil")["cnet"].max()
    j["pointe_conso"] = h.groupby("jour_civil")["consommation"].max()
    j = j[j["n_heures"] >= 22]
    # Puissance installée approchée : max horaire observé sur l'année écoulée (le parc grandit).
    cap = lambda s: s.shift(1).rolling(365, min_periods=200).max()  # noqa: E731
    j["cap_eol"] = cap(g["eolien"].max().reindex(j.index))
    j["cap_sol"] = cap(g["solaire"].max().reindex(j.index))
    j["fc_eol"] = j["eolien"] / j["cap_eol"]
    j["fc_sol"] = j["solaire"] / j["cap_sol"]
    return j


def table_tempo() -> pd.DataFrame:
    fichiers = sorted((config.DATA / "tempo").glob("*.csv"))
    t = pd.concat(pd.read_csv(f) for f in fichiers)
    t["date"] = pd.to_datetime(t["date"])
    t = t[t["couleur"].isin(["Bleu", "Blanc", "Rouge"])]
    return t.set_index("date")["couleur"]


def calendrier(index: pd.DatetimeIndex) -> pd.DataFrame:
    fr = holidays.France(years=range(index.min().year, index.max().year + 2))
    c = pd.DataFrame(index=index)
    c["dow"] = index.day_of_week
    c["ferie"] = [d.date() in fr for d in index]
    debut_saison = pd.to_datetime([f"{d.year if d.month >= 9 else d.year - 1}-09-01" for d in index])
    c["saison"] = debut_saison.year
    c["jour_tempo"] = (index - debut_saison).days + 1  # 1 = 1er septembre
    c["mois"] = index.month
    c["doy"] = index.day_of_year
    return c


def meteo_lieux(source: str = "meteo") -> dict[str, pd.DataFrame]:
    """Lit les fichiers horaires par lieu (source = 'meteo' ou 'previsions')."""
    out = {}
    for nom in LIEUX:
        fs = sorted((config.DATA / source).glob(f"{nom}_*.parquet"))
        if not fs:
            continue
        df = pd.concat(pd.read_parquet(f) for f in fs)
        df["t"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
        out[nom] = df.drop(columns="time").drop_duplicates("t").set_index("t").sort_index()
    return out


def agreger_meteo(lieux: dict[str, pd.DataFrame], suffixe: str = "") -> pd.DataFrame:
    """Moyennes pondérées France par jour Tempo : température (conso), vent à 100 m (éolien), rayonnement."""
    def pond(role: str, var: str) -> pd.Series:
        num, den = 0, 0
        for nom, (_la, _lo, r, p) in LIEUX.items():
            if r == role and nom in lieux and var + suffixe in lieux[nom]:
                num = num + lieux[nom][var + suffixe] * p
                den += p
        return num / den

    h = pd.DataFrame(
        {
            "temp": pond("conso", "temperature_2m"),
            "vent": pond("eolien", "wind_speed_100m"),
            "ray": pond("conso", "shortwave_radiation"),
        }
    )
    h["vent3"] = h["vent"] ** 3  # la production éolienne suit ~ le cube du vent (avant saturation)
    # vent de chaque zone éolienne, et courbe de puissance simplifiée (démarrage 3 m/s, nominal 12 m/s)
    sites = [n for n, (_a, _b, r, _p) in LIEUX.items() if r == "eolien" and n in lieux]
    for n in sites:
        v = lieux[n]["wind_speed_100m" + suffixe]
        h[f"v_{n}"] = v
        h[f"p_{n}"] = (((v - 3) / 9).clip(0, 1) ** 3).where(v < 25, 0)
    h["p_moy"] = h[[f"p_{n}" for n in sites]].mean(axis=1)
    h["jour"] = _jour_tempo(h.index)
    local_h = h.index.tz_convert(TZ).hour
    g = h.groupby("jour")
    j = pd.DataFrame(
        {
            "temp": g["temp"].mean(),
            "temp_min": g["temp"].min(),
            "temp_soir": h[(local_h >= 17) & (local_h <= 20)].groupby("jour")["temp"].mean(),
            "vent": g["vent"].mean(),
            "vent3": g["vent3"].mean(),
            "ray": g["ray"].mean(),
            "p_moy": g["p_moy"].mean(),
            "p_moy_soir": h[(local_h >= 17) & (local_h <= 20)].groupby("jour")["p_moy"].mean(),
            "n": g["temp"].count(),
        }
    )
    for n in sites:
        j[f"v_{n}"] = g[f"v_{n}"].mean()
        j[f"p_{n}"] = g[f"p_{n}"].mean()
    return j[j["n"] >= 20].drop(columns="n")


def table_journaliere(meteo_prevue: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Une ligne par jour Tempo, de 2015 à aujourd'hui + 10 jours.

    Météo : réanalyse ERA5 là où elle existe, complétée par `meteo_prevue` (jours récents et à venir).
    """
    j = table_eco2mix()
    full = pd.date_range(j.index.min(), pd.Timestamp.today().normalize() + pd.Timedelta(days=10))
    j = j.reindex(full)
    j["couleur"] = table_tempo()  # après le reindex : aujourd'hui et demain n'ont pas encore d'éCO2mix complet
    j[["cap_eol", "cap_sol"]] = j[["cap_eol", "cap_sol"]].ffill()  # parc connu le jour de la prévision
    j = j.join(calendrier(j.index))
    met = meteo_lieux("meteo")
    if len(met) == len(LIEUX):  # collecte météo terminée
        m = agreger_meteo(met)
        if meteo_prevue is not None and len(meteo_prevue) == len(LIEUX):
            m = m.combine_first(agreger_meteo(meteo_prevue))
        j = j.join(m)
    return j

