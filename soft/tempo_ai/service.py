"""Orchestration : premier démarrage, tâches planifiées, calcul à la demande, état de l'application."""

from __future__ import annotations

import json
import logging
import threading
import traceback
from datetime import date, datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler

from . import base, collecte, config
from .donnees import table_tempo
from .entrainement import entrainer, modele_courant
from .prevision import CODES, prevoir

log = logging.getLogger(__name__)


class Occupe(RuntimeError):
    pass


class Service:
    def __init__(self) -> None:
        self.verrou = threading.Lock()
        self.etat = dict(status="starting", task=None, message="démarrage")
        self.modeles = None
        self.meta: dict | None = None
        self.prevision: dict | None = None
        self.planif: BackgroundScheduler | None = None

    # ------------------------------------------------------------ outils
    def _progres(self, message: str) -> None:
        log.info(message)
        self.etat["message"] = message

    def _tache(self, kind: str, fonction, attendre: bool = True):
        """Exécute une tâche lourde sous verrou, avec journal en base."""
        if not self.verrou.acquire(blocking=attendre):
            raise Occupe(f"tâche en cours : {self.etat.get('task')}")
        run = base.debut_tache(kind)
        precedent = self.etat["status"]
        self.etat.update(status="busy" if precedent == "ready" else precedent, task=kind)
        try:
            resultat = fonction()
            base.fin_tache(run, "ok", self.etat.get("message", ""))
            self.etat.update(status="ready" if self.modeles else precedent, task=None)
            return resultat
        except Exception as e:  # noqa: BLE001 — on garde l'app en vie et on journalise
            log.error("tâche %s en échec : %s", kind, traceback.format_exc())
            base.fin_tache(run, "error", f"{e.__class__.__name__}: {e}")
            self.etat.update(status="ready" if self.modeles else "error", task=None,
                             message=f"échec {kind} : {e}")
            raise
        finally:
            self.verrou.release()

    def _aujourd_hui(self) -> date:
        return datetime.now(config.TZ_PARIS).date()

    def _apres_heure_prevision(self) -> bool:
        h, m = (int(v) for v in config.HEURE_PREVISION.split(":"))
        return datetime.now(config.TZ_PARIS).time() >= datetime.min.replace(hour=h, minute=m).time()

    def _sauver_prevision(self, prev: dict) -> None:
        self.prevision = prev
        (config.DATA / "derniere_prevision.json").write_text(json.dumps(prev, indent=1), encoding="utf-8")

    def _charger(self) -> None:
        m = modele_courant()
        if m:
            self.modeles, self.meta = m
        f = config.DATA / "derniere_prevision.json"
        if f.exists():
            self.prevision = json.loads(f.read_text(encoding="utf-8"))

    def _couleurs_en_base(self) -> None:
        base.enregistrer_couleurs({d.date().isoformat(): CODES[c] for d, c in table_tempo().items()})

    # ------------------------------------------------------------ étapes
    def _entrainer(self) -> dict:
        meta = entrainer(self._progres)
        base.enregistrer_entrainement(meta)
        self.modeles, self.meta = modele_courant()
        return meta

    def _reentrainement_du(self) -> bool:
        if not self.meta or config.JOURS_REENTRAINEMENT <= 0:
            return not self.meta
        fait = datetime.fromisoformat(self.meta["trained_at"])
        return datetime.now(config.TZ_PARIS) - fait > timedelta(days=config.JOURS_REENTRAINEMENT)

    def _calculer(self) -> dict:
        collecte.collecte_tempo(self._progres)
        collecte.collecte_eco2mix(self._progres)
        f = collecte.collecte_meteo_prevue(self._progres)
        self._couleurs_en_base()
        prev = prevoir(self.modeles, f)
        prev["model_version"] = self.meta["version"]
        self._sauver_prevision(prev)
        self._progres(f"prévision calculée pour le {prev['issued_for']}")
        return prev

    def _quotidien(self) -> dict:
        collecte.collecte_meteo(self._progres)  # complète la météo observée (ERA5)
        if self._reentrainement_du():
            self._entrainer()
        prev = self._calculer()
        n = base.enregistrer_prevision(prev, self.meta["version"])
        self._progres(f"prévision du {prev['issued_for']} enregistrée ({n} horizons)")
        return prev

    def _initialisation(self) -> None:
        if not self.modeles:
            self.etat["status"] = "initializing"
            self._progres("premier démarrage : collecte de l'historique (environ 10 minutes)")
            collecte.collecte_historique(self._progres)
            self._entrainer()
        if not base.prevision_emise(self._aujourd_hui()) and self._apres_heure_prevision():
            self._quotidien()
        else:
            self._calculer()

    # ------------------------------------------------------------ actions publiques
    def demarrer(self) -> None:
        base.initialiser()
        self._charger()
        self.etat.update(status="ready" if self.modeles else "initializing",
                         message="prêt" if self.modeles else "initialisation")
        if config.PLANIFICATION:
            h, m = (int(v) for v in config.HEURE_PREVISION.split(":"))
            self.planif = BackgroundScheduler(timezone=config.TZ_PARIS)
            self.planif.add_job(self.tache_quotidienne, "cron", hour=h, minute=m, id="daily_forecast",
                                misfire_grace_time=3600, coalesce=True)
            self.planif.add_job(self.rafraichir_couleurs, "cron", minute=5, id="hourly_colors",
                                misfire_grace_time=600, coalesce=True)
            self.planif.start()
        threading.Thread(target=self._sans_erreur, args=("startup", self._initialisation), daemon=True).start()

    def arreter(self) -> None:
        if self.planif:
            self.planif.shutdown(wait=False)

    def _sans_erreur(self, kind: str, fonction) -> None:
        try:
            self._tache(kind, fonction)
        except Exception:  # noqa: BLE001 — déjà journalisé
            pass

    def tache_quotidienne(self) -> None:
        self._sans_erreur("daily_forecast", self._quotidien)

    def rafraichir_couleurs(self) -> None:
        def f():
            collecte.collecte_tempo(self._progres)
            self._couleurs_en_base()
            # nouveau jour, ou J+1 vient d'être publié : on recalcule l'affichage (pas la base)
            if self.modeles and self.prevision:
                connues = {d.date().isoformat() for d in table_tempo().index}
                perime = self.prevision["issued_for"] != self._aujourd_hui().isoformat()
                if perime or any(d["source"] == "model" and d["date"] in connues for d in self.prevision["days"]):
                    self._calculer()
        self._sans_erreur("hourly_colors", f)

    def calcul_a_la_demande(self) -> dict:
        if not self.modeles:
            raise Occupe("modèle pas encore entraîné")
        return self._tache("on_demand", self._calculer, attendre=False)

    def entrainer_maintenant(self) -> None:
        if not self.verrou.locked():
            threading.Thread(target=self._sans_erreur, args=("training", self._entrainer), daemon=True).start()
        else:
            raise Occupe(f"tâche en cours : {self.etat.get('task')}")

    def prevision_maintenant(self) -> None:
        if not self.modeles:
            raise Occupe("modèle pas encore entraîné")
        if self.verrou.locked():
            raise Occupe(f"tâche en cours : {self.etat.get('task')}")
        threading.Thread(target=self.tache_quotidienne, daemon=True).start()

    def prochaine_prevision(self) -> str | None:
        if not self.planif:
            return None
        job = self.planif.get_job("daily_forecast")
        return job.next_run_time.isoformat() if job and job.next_run_time else None
