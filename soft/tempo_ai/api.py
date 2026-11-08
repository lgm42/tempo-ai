"""API REST et page web.

GET  /api/summary        tout en une requête (pour Home Assistant)
GET  /api/forecast       dernière prévision ; ?refresh=true pour recalculer maintenant
GET  /api/stats          taux de réussite (global, saison, mois, 7 derniers jours)
GET  /api/history        prévisions passées et vraies couleurs
GET  /api/model          entraînements
GET  /api/runs           journal des tâches
GET  /api/health         état du service
POST /api/run            lance la prévision quotidienne maintenant
POST /api/train          réentraîne les modèles maintenant
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, base, config
from .regles import STOCK_BLANC, STOCK_ROUGE
from .service import Occupe, Service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
service = Service()
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def cycle_de_vie(_app: FastAPI):
    service.demarrer()
    yield
    service.arreter()


app = FastAPI(title="Tempo AI", version=__version__, lifespan=cycle_de_vie,
              description="Local forecast of the French Tempo electricity color at D+2, D+3 and D+4.")


def _autoriser(authorization: str | None) -> None:
    if config.API_TOKEN and authorization != f"Bearer {config.API_TOKEN}":
        raise HTTPException(401, "jeton invalide (en-tête Authorization: Bearer …)")


def _occupe(e: Occupe) -> HTTPException:
    return HTTPException(409, str(e))


def _stocks(prev: dict | None) -> dict | None:
    """Stocks restants après le dernier jour publié par RTE."""
    if not prev:
        return None
    for d in prev["days"]:
        if d["source"] == "model":
            saison = int(d["date"][:4]) if int(d["date"][5:7]) >= 9 else int(d["date"][:4]) - 1
            return dict(season=f"{saison}-{saison + 1}", white_left=d["white_left_before"],
                        red_left=d["red_left_before"], white_total=STOCK_BLANC, red_total=STOCK_ROUGE)
    return None


def _application() -> dict:
    return dict(
        version=__version__,
        status=service.etat["status"],
        task=service.etat["task"],
        message=service.etat["message"],
        last_daily_run=base.derniere_prevision_enregistree(),
        next_daily_run=service.prochaine_prevision(),
        forecast_time=config.HEURE_PREVISION,
        retrain_every_days=config.JOURS_REENTRAINEMENT,
    )


def _modele() -> dict | None:
    if not service.meta:
        return None
    return dict(
        **service.meta,
        red_rule=("2026-27 : rouges approchés par l'algorithme Tempo 2014 + tunnel mensuel CRE 2026-189 "
                  "(seuils PP de RTE non publiés)"),
    )


@app.get("/api/summary")
def summary() -> dict:
    prev = service.prevision
    jour = service._aujourd_hui()
    return dict(
        app=_application(),
        model=_modele(),
        today=prev["today"] if prev else None,
        tomorrow=prev["days"][0] if prev else None,
        forecast=[d for d in prev["days"] if d["horizon"] in config.HORIZONS] if prev else [],
        forecast_issued_at=prev["issued_at"] if prev else None,
        stock=_stocks(prev),
        accuracy=base.statistiques(jour),
    )


@app.get("/api/forecast")
def forecast(refresh: bool = False, authorization: str | None = Header(None)) -> dict:
    if refresh:
        _autoriser(authorization)
        try:
            return service.calcul_a_la_demande()
        except Occupe as e:
            raise _occupe(e) from e
    if not service.prevision:
        raise HTTPException(503, service.etat["message"])
    return service.prevision


@app.get("/api/stats")
def stats() -> dict:
    return base.statistiques(service._aujourd_hui())


@app.get("/api/history")
def history(days: int = Query(30, ge=1, le=3650)) -> list[dict]:
    return base.historique(days, service._aujourd_hui())


@app.get("/api/model")
def model() -> dict:
    return dict(current=_modele(), trainings=base.entrainements())


@app.get("/api/runs")
def runs(limit: int = Query(20, ge=1, le=500)) -> list[dict]:
    return base.dernieres_taches(limit)


@app.get("/api/health")
def health() -> dict:
    return dict(status=service.etat["status"], version=__version__)


@app.post("/api/run", status_code=202)
def run(authorization: str | None = Header(None)) -> dict:
    _autoriser(authorization)
    try:
        service.prevision_maintenant()
    except Occupe as e:
        raise _occupe(e) from e
    return dict(started="daily_forecast")


@app.post("/api/train", status_code=202)
def train(authorization: str | None = Header(None)) -> dict:
    _autoriser(authorization)
    try:
        service.entrainer_maintenant()
    except Occupe as e:
        raise _occupe(e) from e
    return dict(started="training")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
