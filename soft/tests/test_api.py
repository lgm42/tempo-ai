"""API REST, avec un service sans tâche de fond (pas de collecte ni d'entraînement)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from outils import jour_prevu, prevision

from tempo_ai import api, base, config
from tempo_ai.service import Occupe

META = dict(version="20261108-120000", trained_at="2026-11-08T12:00:00+01:00", training_days=4329)


@pytest.fixture
def client(data_dir, monkeypatch):
    s = api.service
    monkeypatch.setattr(s, "etat", dict(status="ready", task=None, message="prêt"))
    monkeypatch.setattr(s, "meta", META)
    monkeypatch.setattr(s, "modeles", {})
    monkeypatch.setattr(s, "prevision", prevision("2026-11-10", [
        jour_prevu("2026-11-11", 1, "blue", source="rte"), jour_prevu("2026-11-12", 2, "white"),
        jour_prevu("2026-11-13", 3, "blue"), jour_prevu("2026-11-14", 4, "red"),
    ]))
    monkeypatch.setattr(config, "API_TOKEN", "")
    return TestClient(api.app)  # sans « with » : le cycle de vie (démarrage du service) n'est pas lancé


def test_summary_contient_tout_pour_home_assistant(client):
    r = client.get("/api/summary")
    assert r.status_code == 200
    j = r.json()
    assert set(j) >= {"app", "model", "today", "tomorrow", "forecast", "forecast_issued_at", "stock", "accuracy"}
    assert [d["horizon"] for d in j["forecast"]] == [2, 3, 4]
    assert [d["color"] for d in j["forecast"]] == ["white", "blue", "red"]
    assert j["tomorrow"]["source"] == "rte"
    assert j["model"]["trained_at"] == META["trained_at"] and "red_rule" in j["model"]
    assert j["stock"] == dict(season="2026-2027", white_left=40, red_left=20, white_total=43, red_total=22)
    assert j["app"]["status"] == "ready"
    assert set(j["accuracy"]) >= {"tracking_since", "overall", "season", "month", "last_7_days"}


def test_summary_pendant_l_initialisation(client, monkeypatch):
    monkeypatch.setattr(api.service, "prevision", None)
    monkeypatch.setattr(api.service, "meta", None)
    j = client.get("/api/summary").json()
    assert j["forecast"] == [] and j["stock"] is None and j["model"] is None


def test_forecast_indisponible_au_demarrage(client, monkeypatch):
    monkeypatch.setattr(api.service, "prevision", None)
    assert client.get("/api/forecast").status_code == 503


def test_lectures(client):
    base.enregistrer_entrainement(META)
    assert client.get("/api/health").json()["status"] == "ready"
    assert client.get("/api/stats").json()["predictions_issued"] == 0
    assert client.get("/api/history?days=7").json() == []
    assert client.get("/api/history?days=0").status_code == 422
    assert client.get("/api/model").json()["trainings"][0]["version"] == META["version"]
    assert client.get("/api/runs").json() == []
    assert client.get("/").headers["content-type"].startswith("text/html")


def test_actions_sans_jeton_configure(client, monkeypatch):
    lance = []
    monkeypatch.setattr(api.service, "prevision_maintenant", lambda: lance.append("run"))
    monkeypatch.setattr(api.service, "entrainer_maintenant", lambda: lance.append("train"))
    assert client.post("/api/run").status_code == 202
    assert client.post("/api/train").status_code == 202
    assert lance == ["run", "train"]


@pytest.mark.parametrize("requete", [
    lambda c, h: c.post("/api/run", headers=h),
    lambda c, h: c.post("/api/train", headers=h),
    lambda c, h: c.get("/api/forecast?refresh=true", headers=h),
])
def test_jeton_exige(client, monkeypatch, requete):
    monkeypatch.setattr(config, "API_TOKEN", "secret")
    monkeypatch.setattr(api.service, "prevision_maintenant", lambda: None)
    monkeypatch.setattr(api.service, "entrainer_maintenant", lambda: None)
    monkeypatch.setattr(api.service, "calcul_a_la_demande", lambda: {"ok": True})
    assert requete(client, {}).status_code == 401
    assert requete(client, {"Authorization": "Bearer faux"}).status_code == 401
    assert requete(client, {"Authorization": "Bearer secret"}).status_code in (200, 202)


def test_conflit_quand_une_tache_tourne(client, monkeypatch):
    def occupe():
        raise Occupe("tâche en cours : training")
    monkeypatch.setattr(api.service, "prevision_maintenant", occupe)
    monkeypatch.setattr(api.service, "calcul_a_la_demande", occupe)
    r = client.post("/api/run")
    assert r.status_code == 409 and "training" in r.json()["detail"]
    assert client.get("/api/forecast?refresh=true").status_code == 409
