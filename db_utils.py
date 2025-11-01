# db_utils.py
# ------------------------------------------------------------
# Fonctions utilitaires pour interagir avec MySQL:
# - get_conn(): ouvrir la connexion
# - get_or_create_person_id(name): récupérer/créer une personne
# - add_image_record(person_id, path): insérer un chemin d'image
# - person_exists(name): vérifier existence par nom
# - fetch_people_and_images(): récupérer (persons, images)
# ------------------------------------------------------------
import mysql.connector
from mysql.connector import Error
from db_config import DB_CONFIG

def get_conn():
    """Ouvrir une connexion MySQL en utilisant DB_CONFIG."""
    return mysql.connector.connect(**DB_CONFIG)

def get_or_create_person_id(name: str) -> int:
    """Retourner l'ID d'une personne par 'name'; la créer si absente."""
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT id FROM persons WHERE name=%s", (name,))
        row = c.fetchone()
        if row:
            pid = row[0]
        else:
            c.execute("INSERT INTO persons(name) VALUES(%s)", (name,))
            conn.commit()
            pid = c.lastrowid
        return pid
    finally:
        c.close()
        conn.close()

def add_image_record(person_id: int, path: str):
    """Insérer une image liée à person_id avec son chemin sur disque."""
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO images(person_id, path) VALUES(%s, %s)",
            (person_id, path)
        )
        conn.commit()
    finally:
        c.close()
        conn.close()

def person_exists(name: str) -> bool:
    """Vérifier si une personne existe en base par son nom."""
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT 1 FROM persons WHERE name=%s LIMIT 1", (name,))
        return c.fetchone() is not None
    finally:
        c.close()
        conn.close()

def fetch_people_and_images():
    """
    Récupérer:
      - people: liste [(id, name), ...]
      - images: liste [(person_id, path), ...]
    """
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute("SELECT id, name FROM persons ORDER BY id")
        people = c.fetchall()
        c.execute("SELECT person_id, path FROM images ORDER BY id")
        images = c.fetchall()
        return people, images
    finally:
        c.close()
        conn.close()
