"""Base SQLite : conservation des prévisions et taux de réussite de l'auto-évaluation."""

from __future__ import annotations

from datetime import date

from outils import jour_prevu, prevision

from tempo_ai import base


def test_seule_la_premiere_prevision_du_jour_est_gardee(data_dir):
    jours = [jour_prevu("2026-11-11", 1, "blue", source="rte"), jour_prevu("2026-11-12", 2, "blue"),
             jour_prevu("2026-11-13", 3, "white"), jour_prevu("2026-11-14", 4, "blue")]
    assert base.enregistrer_prevision(prevision("2026-11-10", jours), "v1") == 3  # J+1 publié : ignoré
    autre = [jour_prevu("2026-11-12", 2, "red"), jour_prevu("2026-11-13", 3, "red")]
    assert base.enregistrer_prevision(prevision("2026-11-10", autre), "v2") == 0
    assert base.prevision_emise(date(2026, 11, 10))
    assert not base.prevision_emise(date(2026, 11, 11))
    h = base.historique(30, date(2026, 11, 15))
    assert {j["date"]: j["predictions"] for j in h}["2026-11-12"]["J+2"]["color"] == "blue"


def _remplir():
    """Prévisions J+2 et J+3 émises pendant 4 semaines, vraies couleurs connues jusqu'au 30/11."""
    vraies = {}
    for jour in range(1, 31):
        vraies[f"2026-11-{jour:02d}"] = "red" if jour in (17, 24) else "white" if jour % 5 == 0 else "blue"
    base.enregistrer_couleurs(vraies)
    for emis in range(1, 29):
        cible2, cible3 = f"2026-11-{emis + 2:02d}", f"2026-11-{min(emis + 3, 30):02d}"
        p2 = vraies[cible2]                              # J+2 toujours juste
        p3 = "red" if emis + 3 in (18, 24) else "blue"  # J+3 : rouges du 24 trouvé, du 17 raté, fausse alerte le 18
        base.enregistrer_prevision(prevision(f"2026-11-{emis:02d}", [
            jour_prevu(cible2, 2, p2), jour_prevu(cible3, 3, p3), jour_prevu("2026-12-31", 4, "blue"),
        ]), "v1")
    return vraies


def test_statistiques(data_dir):
    vraies = _remplir()
    s = base.statistiques(date(2026, 11, 30))
    assert s["tracking_since"] == "2026-11-01"
    assert s["predictions_issued"] == 28 * 3
    assert s["predictions_evaluated"] == 28 * 2  # le 31/12 n'est pas encore connu

    j2 = s["overall"]["by_horizon"]["J+2"]
    assert (j2["correct"], j2["total"], j2["rate"]) == (28, 28, 1.0)
    j3 = s["overall"]["by_horizon"]["J+3"]
    attendues = [vraies[f"2026-11-{min(e + 3, 30):02d}"] for e in range(1, 29)]
    predites = ["red" if e + 3 in (18, 24) else "blue" for e in range(1, 29)]
    assert j3["correct"] == sum(a == p for a, p in zip(attendues, predites))
    assert s["overall"]["by_horizon"]["J+4"]["total"] == 0
    assert s["overall"]["by_horizon"]["J+4"]["rate"] is None

    assert s["overall"]["correct"] == j2["correct"] + j3["correct"]
    # rouges réels : 17 et 24 vus par J+2 et J+3 → 4 ; trouvés : 2 par J+2 + le 24 par J+3
    assert s["overall"]["red_found"] == dict(correct=3, total=4)
    # alertes rouges : 2 par J+2 (justes) + 18 et 24 par J+3 (une juste)
    assert s["overall"]["red_alerts_right"] == dict(correct=3, total=4)
    assert s["season"] == s["overall"]
    assert s["month"]["total"] == s["overall"]["total"]


def test_fenetre_des_7_derniers_jours(data_dir):
    _remplir()
    s = base.statistiques(date(2026, 11, 30))
    # cibles du 24 au 30 : J+2 émises du 22 au 28 → 7 ; J+3 émises du 21 au 28 (celle du 28 vise aussi le 30) → 8
    assert s["last_7_days"]["by_horizon"]["J+2"]["total"] == 7
    assert s["last_7_days"]["by_horizon"]["J+3"]["total"] == 8


def test_base_vide(data_dir):
    s = base.statistiques(date(2026, 11, 8))
    assert s["tracking_since"] is None
    assert s["overall"]["rate"] is None
    assert base.derniere_prevision_enregistree() is None


def test_journal_des_taches(data_dir):
    run = base.debut_tache("daily_forecast")
    assert base.derniere_tache("daily_forecast")["status"] == "running"
    base.fin_tache(run, "ok", "fini")
    assert base.derniere_tache("daily_forecast", "ok")["message"] == "fini"
    assert base.dernieres_taches()[0]["id"] == run


def test_entrainements(data_dir):
    base.enregistrer_entrainement(dict(version="20261101-120000", trained_at="2026-11-01T12:00:00+01:00"))
    base.enregistrer_entrainement(dict(version="20261108-120000", trained_at="2026-11-08T12:00:00+01:00"))
    assert [e["version"] for e in base.entrainements()] == ["20261108-120000", "20261101-120000"]
