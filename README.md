# Projet IAD & SMA - Gestion Intelligente du Trafic (Partie 1)

Ce projet implémente un agent intelligent utilisant l'algorithme Q-Learning tabulé "from scratch" pour optimiser les feux d'un carrefour isolé.

## Structure du projet

```
trafic_urbain/
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── src/
│   ├── env/
│   │   └── traffic_env.py    # Simulateur de carrefour
│   ├── agent/
│   │   ├── q_learning.py     # Agent Q-Learning (from scratch)
│   │   └── baseline.py       # Agent Baseline (feu à durée fixe)
│   └── utils/
│       └── visualization.py  # Utilitaires de tracé des courbes
├── train.py                  # Script d'entraînement de l'agent
└── evaluate.py               # Script d'évaluation (comparaison des agents)
```

## Installation

Assurez-vous d'avoir Python 3.x installé.

1. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Exécution

### 1. Entraîner l'agent Q-Learning
Exécutez le script d'entraînement. L'agent va s'entraîner sur 2000 épisodes (par défaut) et sauvegardera sa politique (Table Q) dans le dossier `models/`, ainsi que la courbe d'apprentissage dans le dossier `results/`.

```bash
python train.py
```

### 2. Évaluer et comparer
Une fois le modèle entraîné, exécutez le script d'évaluation pour comparer les performances de l'agent Q-Learning avec un agent Baseline (feu à minuterie fixe) sur deux scénarios (Trafic Équilibré et Trafic Asymétrique).

```bash
python evaluate.py
```

Les graphiques de comparaison seront générés dans le dossier `results/`.

**Démonstration Live** : Pour visualiser le carrefour en temps réel dans le terminal, ouvrez `evaluate.py` et modifiez le paramètre `render=True` lors de l'appel à la fonction `evaluate_scenario(...)` à la fin du fichier.
