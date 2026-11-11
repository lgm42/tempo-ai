"""Collecte des données brutes, mises en cache dans le dossier de données.

- Couleurs Tempo officielles : api-couleur-tempo.fr, depuis la saison 2019-2020.
- Consommation, éolien et solaire nationaux : ODRÉ / éCO2mix RTE, consolidé et temps réel.
- Météo observée : Open-Meteo, réanalyse ERA5, aux points de `lieux.py`.
- Météo prévue : Open-Meteo forecast, les 14 derniers jours et les 7 suivants. Chaque prévision
  utilisée est archivée.

Chaque fonction ne retélécharge que ce qui manque ou ce qui a pu changer.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from . import config
from .lieux import LIEUX

log = logging.getLogger(__name__)
VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation"]
Progres = Callable[[str], None]


def _dossier(nom: str) -> Path:
    d = config.DATA / nom
    d.mkdir(parents=True, exist_ok=True)
    return d


def _age_jours(f: Path) -> float:
    return (time.time() - f.stat().st_mtime) / 86400 if f.exists() else float("inf")


def _get(url: str, params: dict, timeout: int = 120) -> requests.Response:
    """GET avec nouveaux essais : réseau, limite de débit (429), erreurs serveur."""
    for essai in range(6):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": config.USER_AGENT}, timeout=timeout)
        except requests.exceptions.RequestException as e:
            log.warning("erreur réseau %s (%s), nouvel essai dans 20 s", url, e.__class__.__name__)
            time.sleep(20)
            continue
        if r.status_code == 429:
            log.warning("limite de débit atteinte (%s), pause %d s", url, 60 * (essai + 1))
            time.sleep(60 * (essai + 1))
            continue
        if r.status_code >= 500:
            log.warning("HTTP %d sur %s, nouvel essai dans 30 s", r.status_code, url)
            time.sleep(30)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"{url} refuse toujours la requête")


def _json(url: str, params: dict) -> dict | list:
    for _ in range(3):
        r = _get(url, params)
        if r.text.strip()[:1] in "{[":
            return r.json()
        time.sleep(30)
    raise RuntimeError(f"{url} ne renvoie pas de JSON")


def saison_de(d: date) -> int:
    return d.year if d.month >= 9 else d.year - 1


# ---------------------------------------------------------------- Tempo
def collecte_tempo(progres: Progres = log.info) -> None:
    dossier = _dossier("tempo")
    en_cours = saison_de(date.today())
    for a in range(2019, en_cours + 1):
        f = dossier / f"{a}-{a + 1}.csv"
        if f.exists() and a != en_cours:
            continue
        j = _json("https://www.api-couleur-tempo.fr/api/joursTempo", {"periode": f"{a}-{a + 1}"})
        df = pd.DataFrame(j)
        if df.empty:
            continue
        df = df.rename(columns={"dateJour": "date", "libCouleur": "couleur"})
        df[["date", "couleur"]].to_csv(f, index=False)
        progres(f"couleurs Tempo {a}-{a + 1} : {len(df)} jours")


# ---------------------------------------------------------------- éCO2mix
CHAMPS = "date_heure,consommation,eolien,solaire"


def _export_odre(dataset: str, debut: str) -> pd.DataFrame:
    url = f"https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv"
    r = _get(
        url,
        {
            "select": CHAMPS,
            "where": f"date_heure >= '{debut}' and consommation is not null",
            "order_by": "date_heure",
            "timezone": "UTC",
            "delimiter": ";",
        },
        timeout=900,
    )
    return pd.read_csv(io.StringIO(r.text), sep=";")


def collecte_eco2mix(progres: Progres = log.info) -> None:
    """Consolidé depuis 2015 (rafraîchi chaque mois), puis temps réel depuis la fin du consolidé."""
    f_def = config.DATA / "eco2mix_consolide.parquet"
    if _age_jours(f_def) > 35:
        progres("éCO2mix consolidé : téléchargement de l'historique depuis 2015 (quelques minutes)")
        df = _export_odre("eco2mix-national-cons-def", f"{config.ANNEE_DEBUT}-01-01")
        df.to_parquet(f_def)
        progres(f"éCO2mix consolidé : {len(df)} lignes jusqu'au {df['date_heure'].max()[:10]}")
    fin_def = pd.to_datetime(pd.read_parquet(f_def, columns=["date_heure"])["date_heure"], utc=True).max()
    debut = (fin_def - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = _export_odre("eco2mix-national-tr", debut)
    df.to_parquet(config.DATA / "eco2mix_temps_reel.parquet")
    progres(f"éCO2mix temps réel : {len(df)} lignes jusqu'au {str(df['date_heure'].max())[:16]}")


# ---------------------------------------------------------------- Météo observée (ERA5)
def _complet(f: Path, fin: date) -> bool:
    if not f.exists():
        return False
    t = pd.to_datetime(pd.read_parquet(f, columns=["time"])["time"]).max()
    return t.date() >= fin


def collecte_meteo(progres: Progres = log.info) -> None:
    """Réanalyse ERA5 (disponible avec ~6 jours de retard), un fichier par lieu et par année."""
    dossier = _dossier("meteo")
    fin_dispo = date.today() - timedelta(days=6)
    taches = [(n, a) for n in LIEUX for a in range(config.ANNEE_DEBUT, fin_dispo.year + 1)]
    for i, (nom, an) in enumerate(taches, 1):
        lat, lon, _role, _poids = LIEUX[nom]
        f = dossier / f"{nom}_{an}.parquet"
        fin = min(date(an, 12, 31), fin_dispo)
        if _complet(f, fin) or (an == fin_dispo.year and _age_jours(f) < 1):
            continue
        j = _json(
            "https://archive-api.open-meteo.com/v1/archive",
            {
                "latitude": lat, "longitude": lon,
                "start_date": f"{an}-01-01", "end_date": fin.isoformat(),
                "hourly": ",".join(VARS), "timezone": "UTC", "wind_speed_unit": "ms",
            },
        )
        pd.DataFrame(j["hourly"]).to_parquet(f)
        progres(f"météo observée {i}/{len(taches)} : {nom} {an}")
        time.sleep(2)


# ---------------------------------------------------------------- Météo prévue
def collecte_meteo_prevue(progres: Progres = log.info) -> Path:
    """Prévision du jour (14 jours passés + 7 à venir) pour tous les lieux, archivée par date."""
    noms = list(LIEUX)
    j = _json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": ",".join(str(LIEUX[n][0]) for n in noms),
            "longitude": ",".join(str(LIEUX[n][1]) for n in noms),
            "hourly": ",".join(VARS), "timezone": "UTC", "wind_speed_unit": "ms",
            "past_days": 14, "forecast_days": 7,
        },
    )
    j = j if isinstance(j, list) else [j]
    morceaux = []
    for nom, bloc in zip(noms, j):
        df = pd.DataFrame(bloc["hourly"])
        df.insert(0, "lieu", nom)
        morceaux.append(df)
    df = pd.concat(morceaux, ignore_index=True)
    f = _dossier("meteo_prevue") / f"{datetime.now(config.TZ_PARIS):%Y-%m-%d}.parquet"
    df.to_parquet(f)
    progres(f"météo prévue : {df['time'].min()[:10]} → {df['time'].max()[:10]}")
    return f


def lire_meteo_prevue(f: Path) -> dict[str, pd.DataFrame]:
    """Même structure que donnees.meteo_lieux() : un tableau horaire par lieu."""
    df = pd.read_parquet(f)
    out = {}
    for nom, g in df.groupby("lieu"):
        g = g.drop(columns="lieu").copy()
        g["t"] = pd.to_datetime(g["time"]).dt.tz_localize("UTC")
        out[nom] = g.drop(columns="time").set_index("t").sort_index()
    return out


def derniere_meteo_prevue() -> Path | None:
    fs = sorted(_dossier("meteo_prevue").glob("*.parquet"))
    return fs[-1] if fs else None


def collecte_historique(progres: Progres = log.info) -> None:
    """Tout ce qu'il faut pour entraîner : couleurs, éCO2mix et météo observée."""
    collecte_tempo(progres)
    collecte_eco2mix(progres)
    collecte_meteo(progres)
