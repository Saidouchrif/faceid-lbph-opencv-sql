# FaceID LBPH – OpenCV + SQL

> Mini-projet de reconnaissance faciale “présence/absence” basé sur **OpenCV (LBPH)** avec stockage des métadonnées en **SQL** (MySQL/MariaDB).  
L’objectif est double :  
1) **Enrôlement** — enregistrer des visages depuis des images, les détecter/recadrer, et entraîner un modèle **LBPH**.  
2) **Reconnaissance en direct** — ouvrir la **webcam** pour vérifier si la personne cible (ex. *Said*) est présente ou non, même s’il y a d’autres personnes.

## Fonctionnalités
- Détection de visage (Haar Cascade), recadrage et normalisation.
- Enrôlement multi-personnes avec persistance des chemins d’images en base **SQL**.
- Entraînement et sauvegarde d’un modèle **LBPH** (`data/model.yml`) + mapping labels (`data/labels.json`).
- Reconnaissance temps réel via webcam avec seuil de confiance ajustable.
- Logique “**Est-ce moi ?**” (présent/absent) configurable via `TARGET_NAME`.

## Stack
- **Python 3**, **OpenCV (opencv-contrib-python)**, **NumPy**
- **MySQL/MariaDB** (modifiable vers PostgreSQL)
- Architecture simple de répertoires (`data/dataset`, `data/model.yml`, etc.)

## Démarrage rapide
1. Installer les dépendances :
   ```bash
   pip install opencv-contrib-python numpy mysql-connector-python
