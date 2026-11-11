"""Fait pointer les scripts de recherche vers la bibliothèque de l'app (soft/) et le cache local data/."""

import os
import sys
from pathlib import Path

ICI = Path(__file__).resolve().parent
os.environ.setdefault("TEMPO_DATA_DIR", str(ICI / "data"))
sys.path.insert(0, str(ICI.parent / "soft"))
