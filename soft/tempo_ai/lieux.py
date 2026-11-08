"""Points météo de référence : (latitude, longitude, rôle, poids).

- « conso » : grandes agglomérations, pondérées par la population (millions d'hab., aires urbaines),
  pour une température « France » représentative de la consommation (chauffage).
- « eolien » : zones à forte densité de parcs éoliens, terrestres et en mer.
Le rayonnement solaire est pris aux points « conso ».
"""

LIEUX: dict[str, tuple[float, float, str, float]] = {
    # consommation
    "paris": (48.86, 2.35, "conso", 13.1),
    "lyon": (45.76, 4.84, "conso", 2.3),
    "marseille": (43.30, 5.37, "conso", 1.9),
    "lille": (50.63, 3.06, "conso", 1.5),
    "toulouse": (43.60, 1.44, "conso", 1.5),
    "bordeaux": (44.84, -0.58, "conso", 1.4),
    "nantes": (47.22, -1.55, "conso", 1.0),
    "strasbourg": (48.57, 7.75, "conso", 0.9),
    # éolien
    "eol_picardie": (49.85, 3.00, "eolien", 1.0),
    "eol_champagne": (48.90, 4.20, "eolien", 1.0),
    "eol_beauce": (48.10, 1.60, "eolien", 0.6),
    "eol_bretagne": (48.40, -2.90, "eolien", 0.5),
    "eol_occitanie": (43.20, 2.90, "eolien", 0.5),
    "eol_poitou": (46.50, -0.20, "eolien", 0.5),
    "eol_lorraine": (48.90, 6.40, "eolien", 0.5),
    "eol_mer_manche": (49.90, 0.30, "eolien", 0.7),
}
