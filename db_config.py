# db_config.py
# ------------------------------------------------------------
# Configuration de la connexion MySQL (à adapter selon ton env)
# ------------------------------------------------------------
DB_CONFIG = {
    "host": "mysqldb",     # nom du service Docker
    "user": "faceid_user", # remplace par ton user
    "password": "faceid_pass", # remplace par ton mot de passe
    "database": "faceid_db",
    "port": 3306,          # 3306 par défaut
}
