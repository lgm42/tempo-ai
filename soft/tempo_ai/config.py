"""Réglages, lus dans les variables d'environnement (voir README)."""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

from . import __version__

# Dossier persistant : données brutes, modèles, base SQLite.
DATA = Path(os.environ.get("TEMPO_DATA_DIR", "/data"))

# Heure locale (Europe/Paris) de la prévision quotidienne. La couleur de J+1 est publiée vers 10 h 40.
HEURE_PREVISION = os.environ.get("TEMPO_FORECAST_TIME", "12:00")

# Réentraînement automatique quand le dernier entraînement a plus de N jours (0 = jamais).
JOURS_REENTRAINEMENT = int(os.environ.get("TEMPO_RETRAIN_DAYS", "30"))

# Jeton facultatif exigé pour les actions (POST). Vide = actions ouvertes à tous.
API_TOKEN = os.environ.get("TEMPO_API_TOKEN", "")

# Désactive la planification (tests, usage en bibliothèque).
PLANIFICATION = os.environ.get("TEMPO_SCHEDULER", "1") != "0"

TZ_PARIS = ZoneInfo("Europe/Paris")
USER_AGENT = f"tempo-ai/{__version__} (+https://github.com/lgm42/tempo-ai)"

HORIZONS = [2, 3, 4]
ANNEE_DEBUT = 2015  # début de l'historique éCO2mix et météo
