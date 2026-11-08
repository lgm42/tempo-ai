# Prévision Tempo J+2 / J+3 / J+4

Petit modèle local qui prévoit la couleur Tempo 2 à 4 jours à l'avance.
Pour une explication pas à pas sans connaissances en IA : [COMPRENDRE.md](COMPRENDRE.md).
La couleur de J+1 est publiée par RTE la veille à 10 h 30.

## Principe

Plutôt que de deviner directement la couleur (seulement 22 rouges par an pour apprendre), on
reproduit la chaîne de décision de RTE :

1. **Prévoir la consommation nette** (consommation − éolien − solaire) à partir de la météo prévue.
   Modèles LightGBM (`soft/tempo_ai/modele.py`) :
   - **consommation** : moyenne de 6 h à 6 h et pointe horaire. On prédit l'écart au niveau des
     7 derniers jours, ce qui résiste à la baisse de consommation depuis 2022.
   - **facteur de charge éolien** : à partir du vent à 100 m sur 8 zones éoliennes.
   - **facteur de charge solaire**.

   Le parc installé (éolien, solaire) est pris comme le maximum horaire de l'année écoulée, car un
   modèle à arbres n'extrapole pas la croissance du parc.
2. **Appliquer la règle officielle de RTE** (`soft/tempo_ai/regles.py`). On enchaîne les stocks : la couleur réelle
   de J+1, puis les couleurs prévues de J+2 et J+3.

## Règles RTE (sources : `docs/SOURCES.md`)

- **Algorithme de 2014** (fiche RTE indice 2 du 07/01/2025) :
  - la consommation nette est normalisée sur un an glissant, avec une correction de température ;
  - seuil blanc + rouge = 4 − 0,015·j − 0,026·(stock blanc + stock rouge) ;
  - seuil rouge = 3,15 − 0,010·j − 0,031·stock rouge.
- **Contraintes de placement des rouges :**
  - novembre à mars uniquement, du lundi au vendredi, hors jours fériés ;
  - au plus 5 rouges d'affilée, le week-end coupant la série ;
  - en fin d'hiver, apurement forcé quand le stock restant égale le nombre de jours éligibles restants. La limite de 5 ne s'applique plus alors, comme en mars 2026 avec 13 rouges d'affilée.
- **Depuis le 1er novembre 2026** (délibération CRE n° 2026-189) :
  - les **rouges** sont les **jours PP** du mécanisme de capacité ;
  - un jour est rouge si la **pointe horaire** prévue de consommation résiduelle dépasse un seuil
    publié par RTE (tableau jour × rouges restants) ;
  - le cumul des rouges en fin de mois doit rester dans une fourchette : 0 à 3 fin novembre, exactement 6 fin décembre,
    13 à 19 fin janvier, 20 à 22 fin février, 22 fin mars ;
  - les 24 et 31 décembre sont exclus.
  Les **blancs** suivent toujours l'algorithme de 2014.
  **Le tableau des seuils PP 2026-27 n'est pas encore intégré** : il n'a pas été trouvé en ligne.

## Données (`soft/tempo_ai/collecte.py` + `collecte_recherche.py`, cache dans `train/data/`)

| Donnée | Source | Depuis |
|---|---|---|
| Couleurs Tempo | api-couleur-tempo.fr | saison 2019-2020 |
| Consommation, éolien, solaire | ODRÉ / éCO2mix RTE (consolidé + temps réel) | 2015 |
| Météo observée | Open-Meteo, réanalyse ERA5, 16 points | 2015 |
| Météo prévue à l'époque (J-1 à J-5) | Open-Meteo « previous runs » | octobre 2024 |

## Étape 1 : rejeu des hivers 2024-25 et 2025-26

`backtest.py` rejoue chaque jour « comme si on y était ». Le modèle est entraîné seulement sur les
saisons précédentes et ne reçoit que la météo **prévue** à l'époque. Scores de novembre à mars,
306 jours par horizon :

| | J+2 | J+3 | J+4 |
|---|---|---|---|
| **Couleur exacte (modèle)** | **88 %** | **86 %** | **85 %** |
| Référence naïve : « même couleur que J+1 » | 61 % | 51 % | 54 % |
| Plafond : règle appliquée à la consommation réelle | 91 % | 91 % | 91 % |
| Rouges trouvés (rappel) | 93 % | 91 % | 93 % |
| Rouges annoncés qui l'étaient vraiment (précision) | 89 % | 87 % | 79 % |
| Blancs trouvés | 68 % | 65 % | 65 % |
| Erreur moyenne sur la consommation nette | 4 % | 4 % | 5 % |

À retenir :
- Le modèle atteint 95 % du plafond. Une prévision météo parfaite ne ferait gagner que 3 à 6 points.
- **Un rouge n'est jamais pris pour un bleu à J+2 et J+3**, et un bleu n'est quasiment jamais annoncé rouge.
- **Le point faible est la frontière bleu/blanc** : environ 20 blancs ratés par hiver. Le
  plafond lui-même y perd des points, ce qui laisse penser que la normalisation RTE réelle diffère
  un peu de notre reconstitution.
- Ces scores valent pour l'ancienne règle des rouges. Pour les rouges de cet hiver, il faudra les
  seuils PP de RTE. Notre pointe résiduelle est prévue à environ 4-5 % près à J+2…J+4. RTE indique
  pour sa propre prévision un écart typique de 2,8 % entre J-2 et J-1.

## Utilisation

```
cd train
python collecte_recherche.py   # historique + prévisions météo archivées (cache dans train/data)
python verif_regle.py          # fidélité de la règle sur la consommation réelle
python backtest.py             # rejeu complet → resultats/rejeu.csv, resultats/scores.csv
python backtest.py rapport     # réaffiche les scores
```

Entraînement en quelques secondes sur processeur : pas besoin de GPU, le NAS suffit.

## Suite

- [ ] Seuils PP 2026-27 de RTE → règle des rouges de cet hiver (pointe résiduelle vs seuil, fourchette mensuelle).
- [ ] Améliorer la frontière bleu/blanc (recalage de la normalisation sur l'historique).
- [ ] Étape 2 : service quotidien sur le NAS (prévision à 12 h), base SQLite des prévisions et des vraies
      couleurs, capteurs dans Home Assistant, suivi du taux de réussite par horizon.
