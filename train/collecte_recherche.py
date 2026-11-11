"""Collecte pour la recherche (cache dans train/data).

- historique : couleurs Tempo, éCO2mix, météo observée (mêmes fonctions que l'app) ;
- prévisions météo passées, telles qu'émises 1 à 5 jours avant (Open-Meteo « previous runs »,
  disponibles depuis 2024), pour rejouer les hivers « comme si on y était ».

Usage : python collecte_recherche.py [historique|previsions|tout]
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta

import _chemins  # noqa: F401
import pandas as pd

from tempo_ai import config
from tempo_ai.collecte import VARS, _json, collecte_historique
from tempo_ai.lieux import LIEUX

# Seulement les périodes utiles au rejeu (octobre → avril).
PERIODES_PREVI = [("2024-10-01", "2025-04-30"), ("2025-10-01", "2026-04-30"), ("2026-10-01", None)]
DELAIS = [1, 2, 3, 4, 5]


def collecte_previsions() -> None:
    dossier = config.DATA / "previsions"
    dossier.mkdir(parents=True, exist_ok=True)
    for debut, fin_fixe in PERIODES_PREVI:
        fin = fin_fixe or (date.today() - timedelta(days=1)).isoformat()
        for nom, (lat, lon, _role, _poids) in LIEUX.items():
            f = dossier / f"{nom}_{debut[:7]}.parquet"
            if f.exists() and fin_fixe:
                continue
            morceaux = []
            for v in VARS:  # 1 variable × 5 délais par requête
                j = _json(
                    "https://previous-runs-api.open-meteo.com/v1/forecast",
                    {
                        "latitude": lat, "longitude": lon,
                        "start_date": debut, "end_date": fin,
                        "hourly": ",".join(f"{v}_previous_day{d}" for d in DELAIS),
                        "timezone": "UTC", "wind_speed_unit": "ms",
                    },
                )
                morceaux.append(pd.DataFrame(j["hourly"]).set_index("time"))
                time.sleep(2)
            df = pd.concat(morceaux, axis=1).reset_index()
            df.to_parquet(f)
            print(f"prévisions {nom} {debut[:7]}: {len(df)} h")


if __name__ == "__main__":
    quoi = sys.argv[1] if len(sys.argv) > 1 else "tout"
    if quoi in ("historique", "tout"):
        collecte_historique(print)
    if quoi in ("previsions", "tout"):
        collecte_previsions()
