# Guide d'Utilisation - Reconnaissance Faciale avec LBPH

## 📋 Prérequis

- Docker et Docker Compose installés
- Python 3.11+ (pour exécuter les scripts localement)
- Une caméra webcam connectée

## 🚀 Démarrage Rapide

### 1. Lancer les services Docker (MySQL + phpMyAdmin)

```bash
docker-compose up -d
```

Cela démarre :
- **MySQL** : Base de données (port 3306)
- **phpMyAdmin** : Interface d'administration (http://localhost:8080)
- **App Container** : Conteneur Python

### 2. Importer les images des personnes

Les images doivent être organisées dans un dossier `people/` comme suit :

```
people/
├── Said/
│   ├── said1.jpg
│   ├── said2.jpg
│   └── said3.jpg
├── Bilal/
│   ├── bilal1.jpg
│   ├── bilal2.jpg
│   └── bilal3.jpg
└── Abdlatif/
    ├── abdlatif1.jpg
    ├── abdlatif2.jpg
    └── abdlatif3.jpg
```

Ensuite, importer les images dans la base de données :

```bash
docker-compose exec app python import_people_mysql.py
```

Cela va :
- Détecter les visages dans chaque image
- Sauvegarder les visages recadrés dans `data/dataset/`
- Enregistrer les chemins dans MySQL
- Entraîner le modèle LBPH
- Sauvegarder le modèle dans `data/model.yml`

### 3. Lancer la reconnaissance faciale en direct

#### Option A : Depuis votre machine (avec accès à la caméra)

```bash
# Installation des dépendances (première fois)
python -m pip install -r requirements.txt

# Lancer la reconnaissance faciale
python recognize_camera_local.py
```

#### Option B : Depuis le conteneur Docker (sans caméra)

```bash
docker-compose exec app python recognize_live_mysql.py
```

**Note** : Cette option ne fonctionne que si Docker a accès à la caméra (configuration avancée).

## 🎮 Contrôles

- **Q** ou **ESC** : Quitter l'application
- **C** : Changer de caméra (si plusieurs caméras disponibles)

## 📊 Vérifier les données importées

Accédez à phpMyAdmin : http://localhost:8080

- **Utilisateur** : `faceid_user`
- **Mot de passe** : `faceid_pass`
- **Base de données** : `faceid_db`

Vous pouvez voir :
- Table `persons` : Liste des personnes
- Table `images` : Chemins des images importées

## 🔧 Configuration

### Modifier le seuil de reconnaissance

Ouvrez `recognize_camera_local.py` et modifiez :

```python
THRESHOLD = 70.0  # Plus bas = plus strict, plus haut = plus permissif
```

### Modifier la base de données

Modifiez `db_config.py` :

```python
DB_CONFIG = {
    "host": "localhost",  # ou "127.0.0.1" pour accès local
    "user": "faceid_user",
    "password": "faceid_pass",
    "database": "faceid_db",
    "port": 3306,
}
```

## 📁 Structure du Projet

```
faceid-lbph-opencv-sql/
├── people/                      # Dossier des images brutes (à créer)
├── data/
│   ├── dataset/                 # Visages recadrés et redimensionnés
│   ├── model.yml                # Modèle LBPH entraîné
│   └── labels.json              # Mapping label -> nom
├── sql/
│   └── schema.sql               # Schéma de la base de données
├── Dockerfile                   # Configuration Docker
├── docker-compose.yml           # Orchestration des services
├── requirements.txt             # Dépendances Python
├── db_config.py                 # Configuration de la base de données
├── db_utils.py                  # Utilitaires pour MySQL
├── import_people_mysql.py       # Script d'importation des images
├── recognize_camera_local.py    # Script de reconnaissance (machine locale)
├── recognize_live_mysql.py      # Script de reconnaissance (Docker)
└── enroll_face_mysql.py         # Script d'enrôlement d'une personne
```

## 🐛 Dépannage

### "Aucune caméra trouvée"

- Vérifiez que votre caméra est connectée
- Vérifiez que Windows a les permissions pour accéder à la caméra
- Essayez de redémarrer l'application

### "Modèle non trouvé"

- Assurez-vous d'avoir exécuté `import_people_mysql.py` d'abord
- Vérifiez que le dossier `data/` existe

### "Erreur de connexion MySQL"

- Vérifiez que Docker Compose est en cours d'exécution : `docker-compose ps`
- Vérifiez les logs : `docker-compose logs mysqldb`
- Redémarrez les services : `docker-compose restart`

## 📝 Ajouter une nouvelle personne

1. Créez un dossier dans `people/` avec le nom de la personne
2. Ajoutez ses photos (JPG, PNG, etc.)
3. Exécutez : `docker-compose exec app python import_people_mysql.py`
4. Le modèle sera réentraîné automatiquement

## 🎯 Résultats Attendus

Lors de la reconnaissance faciale :
- **Vert** : Visage reconnu (confiance < seuil)
- **Rouge** : Visage non reconnu (confiance > seuil)
- **Texte** : Nom de la personne et score de confiance

## 📞 Support

Pour plus d'informations, consultez :
- `README.md` : Documentation générale
- `db_utils.py` : Fonctions de base de données
- `recognize_camera_local.py` : Code de reconnaissance faciale
