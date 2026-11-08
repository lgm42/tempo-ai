"""Base SQLite : prévisions émises chaque jour, vraies couleurs, entraînements, journal des tâches.

Auto-évaluation : chaque prévision J+2 / J+3 / J+4 est conservée telle qu'émise ; dès que RTE publie
la vraie couleur du jour visé, la comparaison alimente les taux de réussite.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    issued_for TEXT NOT NULL,      -- jour J où la prévision est émise
    target TEXT NOT NULL,          -- jour visé
    horizon INTEGER NOT NULL,      -- 2, 3 ou 4
    color TEXT NOT NULL,           -- blue / white / red
    p_blue REAL, p_white REAL, p_red REAL,
    net_load_mw REAL, score REAL, threshold_white REAL, threshold_red REAL,
    model_version TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (issued_for, horizon)
);
CREATE INDEX IF NOT EXISTS predictions_target ON predictions(target);
CREATE TABLE IF NOT EXISTS colors (
    day TEXT PRIMARY KEY,
    color TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trainings (
    version TEXT PRIMARY KEY,
    trained_at TEXT NOT NULL,
    meta TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL,
    message TEXT
);
"""


@contextmanager
def connexion():
    config.DATA.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(config.DATA / "tempo_ai.sqlite3", timeout=30)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def initialiser() -> None:
    with connexion() as c:
        c.executescript(SCHEMA)


def _maintenant() -> str:
    return datetime.now(config.TZ_PARIS).isoformat(timespec="seconds")


# ---------------------------------------------------------------- écriture
def enregistrer_prevision(prev: dict, version: str) -> int:
    """Conserve la prévision du jour (une seule par jour et par horizon : la première émise)."""
    n = 0
    with connexion() as c:
        for d in prev["days"]:
            if d["horizon"] not in config.HORIZONS or d["source"] != "model":
                continue
            p = d["probabilities"]
            cur = c.execute(
                "INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (prev["issued_for"], d["date"], d["horizon"], d["color"], p["blue"], p["white"], p["red"],
                 d["net_load_mw"], d["score"], d["threshold_white"], d["threshold_red"], version, _maintenant()),
            )
            n += cur.rowcount
    return n


def enregistrer_couleurs(couleurs: dict[str, str]) -> None:
    with connexion() as c:
        c.executemany("INSERT OR REPLACE INTO colors VALUES (?,?)", couleurs.items())


def enregistrer_entrainement(meta: dict) -> None:
    with connexion() as c:
        c.execute("INSERT OR REPLACE INTO trainings VALUES (?,?,?)",
                  (meta["version"], meta["trained_at"], json.dumps(meta)))


def debut_tache(kind: str) -> int:
    with connexion() as c:
        return c.execute("INSERT INTO runs (kind, started_at, status) VALUES (?,?,?)",
                         (kind, _maintenant(), "running")).lastrowid


def fin_tache(run_id: int, status: str, message: str = "") -> None:
    with connexion() as c:
        c.execute("UPDATE runs SET ended_at=?, status=?, message=? WHERE id=?",
                  (_maintenant(), status, message[:2000], run_id))


# ---------------------------------------------------------------- lecture
def prevision_emise(jour: date) -> bool:
    with connexion() as c:
        return c.execute("SELECT 1 FROM predictions WHERE issued_for=?", (jour.isoformat(),)).fetchone() is not None


def derniere_prevision_enregistree() -> str | None:
    with connexion() as c:
        return c.execute("SELECT MAX(created_at) FROM predictions").fetchone()[0]


def dernieres_taches(n: int = 20) -> list[dict]:
    with connexion() as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (n,))]


def derniere_tache(kind: str, status: str | None = None) -> dict | None:
    q, a = "SELECT * FROM runs WHERE kind=?", [kind]
    if status:
        q, a = q + " AND status=?", a + [status]
    with connexion() as c:
        r = c.execute(q + " ORDER BY id DESC LIMIT 1", a).fetchone()
    return dict(r) if r else None


def entrainements() -> list[dict]:
    with connexion() as c:
        return [json.loads(r["meta"]) for r in c.execute("SELECT meta FROM trainings ORDER BY trained_at DESC")]


def _evaluees(c: sqlite3.Connection, depuis: str | None = None, jusqu_a: str | None = None) -> list[sqlite3.Row]:
    q = """SELECT p.target, p.horizon, p.color AS predicted, k.color AS actual
           FROM predictions p JOIN colors k ON k.day = p.target WHERE 1=1"""
    a = []
    if depuis:
        q, a = q + " AND p.target >= ?", a + [depuis]
    if jusqu_a:
        q, a = q + " AND p.target <= ?", a + [jusqu_a]
    return c.execute(q, a).fetchall()


def _taux(lignes: list) -> dict:
    total = len(lignes)
    justes = sum(r["predicted"] == r["actual"] for r in lignes)
    return dict(rate=round(justes / total, 4) if total else None, correct=justes, total=total)


def _bloc(lignes: list) -> dict:
    b = _taux(lignes)
    b["by_horizon"] = {f"J+{h}": _taux([r for r in lignes if r["horizon"] == h]) for h in config.HORIZONS}
    rouges = [r for r in lignes if r["actual"] == "red"]
    alertes = [r for r in lignes if r["predicted"] == "red"]
    b["red_found"] = dict(correct=sum(r["predicted"] == "red" for r in rouges), total=len(rouges))
    b["red_alerts_right"] = dict(correct=sum(r["actual"] == "red" for r in alertes), total=len(alertes))
    return b


def statistiques(aujourd_hui: date) -> dict:
    debut_saison = date(aujourd_hui.year if aujourd_hui.month >= 9 else aujourd_hui.year - 1, 9, 1)
    with connexion() as c:
        premiere = c.execute("SELECT MIN(issued_for), COUNT(*) FROM predictions").fetchone()
        tout = _evaluees(c)
        stats = dict(
            tracking_since=premiere[0],
            predictions_issued=premiere[1],
            predictions_evaluated=len(tout),
            overall=_bloc(tout),
            season=_bloc(_evaluees(c, debut_saison.isoformat())),
            month=_bloc(_evaluees(c, aujourd_hui.replace(day=1).isoformat())),
            last_7_days=_bloc(_evaluees(c, (aujourd_hui - timedelta(days=6)).isoformat())),
        )
    return stats


def historique(jours: int, aujourd_hui: date) -> list[dict]:
    """Par jour visé : vraie couleur et prévisions émises à J-2, J-3, J-4."""
    debut = (aujourd_hui - timedelta(days=jours)).isoformat()
    with connexion() as c:
        rows = c.execute(
            """SELECT p.target, p.horizon, p.color, p.p_blue, p.p_white, p.p_red, k.color AS actual
               FROM predictions p LEFT JOIN colors k ON k.day = p.target
               WHERE p.target >= ? ORDER BY p.target DESC, p.horizon""", (debut,)).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        j = out.setdefault(r["target"], dict(date=r["target"], actual=r["actual"], predictions={}))
        j["predictions"][f"J+{r['horizon']}"] = dict(
            color=r["color"], probabilities=dict(blue=r["p_blue"], white=r["p_white"], red=r["p_red"]))
    return list(out.values())
