"""Entraînement des modèles de consommation nette, un jeu de modèles par horizon (J+1 à J+4).

Apprentissage sur tout l'historique disponible avec la météo observée (ERA5). Une validation rapide
mesure l'erreur sur les 12 derniers mois avec un modèle entraîné sur les années précédentes.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from . import config
from .donnees import table_journaliere
from .modele import ModeleCnet, variables

log = logging.getLogger(__name__)
HORIZONS_MODELE = [1, 2, 3, 4]  # J+1 sert seulement avant la publication de RTE (10 h 40)


def dossier_modeles() -> Path:
    d = config.DATA / "modeles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _erreur_hiver(modele: ModeleCnet, x: pd.DataFrame, t: pd.DataFrame) -> float | None:
    hiver = x.index.month.isin([11, 12, 1, 2, 3])
    x = x[hiver].dropna()
    if x.empty:
        return None
    p = modele.predict(x)
    reel = t.loc[p.index, "cnet"]
    ok = reel.notna()
    return float(((p.cnet[ok] - reel[ok]).abs() / reel[ok]).mean()) if ok.any() else None


def entrainer(progres: Callable[[str], None] = log.info) -> dict:
    debut_chrono = time.time()
    t = table_journaliere()
    aujourd_hui = pd.Timestamp.today().normalize()
    connu = t[(t.index < aujourd_hui) & t["cnet"].notna()]
    coupure = aujourd_hui - pd.Timedelta(days=365)
    nom = datetime.now(config.TZ_PARIS).strftime("%Y%m%d-%H%M%S")
    cible = dossier_modeles() / nom
    validation = {}
    for h in HORIZONS_MODELE:
        progres(f"entraînement J+{h}")
        x = variables(t, h).loc[connu.index]
        # validation : appris avant la coupure, testé sur les 12 derniers mois (météo observée)
        m_val = ModeleCnet().fit(x[x.index < coupure], connu[connu.index < coupure])
        validation[f"J+{h}"] = _erreur_hiver(m_val, x[x.index >= coupure], t)
        ModeleCnet().fit(x, connu).sauver(cible / f"J{h}")
    meta = dict(
        version=nom,
        trained_at=datetime.now(config.TZ_PARIS).isoformat(timespec="seconds"),
        data_from=connu.index.min().date().isoformat(),
        data_to=connu.index.max().date().isoformat(),
        training_days=int(len(connu)),
        duration_s=round(time.time() - debut_chrono, 1),
        validation_winter_net_load_error=validation,
    )
    (cible / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (dossier_modeles() / "courant.txt").write_text(nom, encoding="utf-8")
    for ancien in sorted(p for p in dossier_modeles().iterdir() if p.is_dir())[:-3]:
        shutil.rmtree(ancien, ignore_errors=True)  # on garde les 3 derniers
    progres(f"modèle {nom} entraîné sur {meta['training_days']} jours en {meta['duration_s']} s")
    return meta


def modele_courant() -> tuple[dict[int, ModeleCnet], dict] | None:
    f = dossier_modeles() / "courant.txt"
    if not f.exists():
        return None
    d = dossier_modeles() / f.read_text(encoding="utf-8").strip()
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return {h: ModeleCnet.charger(d / f"J{h}") for h in HORIZONS_MODELE}, meta
