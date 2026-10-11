# 🛵 Food Delivery Time Prediction

Prédire le **temps de livraison** d'une commande (en minutes) avec une **régression linéaire**, à partir de la distance, de la météo, du trafic, du type de véhicule, etc.

## 🎯 Objectif

Construire un modèle de **régression** capable d'estimer le temps de livraison, le comparer à un modèle de référence (la moyenne), et comprendre quels facteurs allongent la livraison.

## 📊 Dataset

`Food_Delivery_Times.csv` : 1000 commandes.

> Source des données : *(ajoute ici le lien Kaggle du dataset)*

| Variable | Description |
|---|---|
| `Order_ID` | Identifiant de la commande (non utilisé) |
| `Distance_km` | Distance de livraison (km) |
| `Weather` | Météo (Clear, Rainy, Foggy, Snowy, Windy) |
| `Traffic_Level` | Niveau de trafic (Low, Medium, High) |
| `Time_of_Day` | Moment de la journée (Morning, Afternoon, Evening, Night) |
| `Vehicle_Type` | Véhicule (Bike, Scooter, Car) |
| `Preparation_Time_min` | Temps de préparation (min) |
| `Courier_Experience_yrs` | Expérience du livreur (années) |
| **`Delivery_Time_min`** | **Temps de livraison (min) : la variable à prédire** |

Le dataset contient 30 valeurs manquantes (3 %) dans 4 colonnes, traitées dans le notebook.

## 🔍 Démarche

1. **Analyse exploratoire (EDA)** : distributions, relation distance / temps, valeurs extrêmes
2. **Séparation train / test** (80 % / 20 %)
3. **Nettoyage** : médiane (numérique) ou valeur la plus fréquente (catégoriel), calculées sur le train uniquement pour éviter la fuite de données
4. **Encodage** (`OneHotEncoder`) et **standardisation** (`StandardScaler`)
5. **Modèle de référence** : toujours prédire la moyenne
6. **Régression linéaire** (`LinearRegression`), interprétation des coefficients, analyse des erreurs
7. **Descente de gradient** (`SGDRegressor`) et comparaison avec la solution exacte

## 🏆 Résultats (jeu de test, 200 commandes)

| Modèle | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| Référence (moyenne) | 17,50 | 21,23 | -0,01 |
| **Régression linéaire** | **6,06** | **8,95** | **0,82** |
| Descente de gradient (SGD) | 6,01 | 8,94 | 0,82 |

## 💡 Ce qu'on retient

- **En moyenne, la prédiction se trompe d'environ 6 minutes** (contre 17,5 pour la référence), et le modèle explique 82 % de la variabilité du temps de livraison.
- **La distance est le facteur le plus important**, devant le trafic, la météo (neige, brouillard, pluie) et le temps de préparation.
- **La descente de gradient converge vers la même solution** que la régression linéaire classique.
- **Seules 5 commandes sur 200 ont une erreur de plus de 20 minutes**, toutes des livraisons anormalement longues.

## ⚠️ Limites

- Dataset petit (1000 commandes) et évaluation sur un seul découpage de 200 commandes.
- Les livraisons très longues ne sont pas bien prédites par un modèle linéaire.
- Pistes d'amélioration : validation croisée, `Pipeline` scikit-learn, modèles non linéaires (Random Forest, Gradient Boosting).

## 🚀 Lancer le projet

Le projet utilise [uv](https://docs.astral.sh/uv/) pour gérer Python et les dépendances.

```bash
# 1. Installer uv (une seule fois) : https://docs.astral.sh/uv/getting-started/installation/

# 2. Installer les dépendances (crée le dossier .venv)
uv sync

# 3. Placer Food_Delivery_Times.csv dans data/raw/

# 4. Ouvrir le notebook
uv run jupyter lab
```

Dans VS Code, ouvre `notebooks/food_delivery_time_prediction.ipynb` et choisis l'interpréteur `.venv` comme kernel.

## 📁 Structure du projet

```
├── data/
│   └── raw/
│       └── Food_Delivery_Times.csv
├── notebooks/
│   └── food_delivery_time_prediction.ipynb
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

## 🛠️ Technologies

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · uv
