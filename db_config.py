# db_config.py
# ------------------------------------------------------------
# Configuration de la connexion MySQL (à adapter selon ton env)
# ------------------------------------------------------------
DB_CONFIG = {
    "host": "127.0.0.1",   # si tu utilises Docker: "localhost" / "127.0.0.1" depuis la machine hôte
    "user": "faceid_user", # remplace par ton user
    "password": "faceid_pass", # remplace par ton mot de passe
    "database": "faceid_db",
    "port": 3306,          # 3306 par défaut
}
