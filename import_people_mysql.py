# import_people_mysql.py
# ------------------------------------------------------------
# Objectif :
# - Parcourir le dossier "people/<Nom>/*" (images brutes)
# - Pour chaque image : détecter/recadrer le visage (Haar), sauvegarder
#   dans "data/dataset/<Nom>/...", puis indexer le chemin dans MySQL.
# - À la fin : entraîner le modèle LBPH et sauvegarder "data/model.yml"
#   + "data/labels.json".
# ------------------------------------------------------------
import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path

from db_utils import get_or_create_person_id, add_image_record, fetch_people_and_images

# Dossiers/fichiers
DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
MODEL_PATH = os.path.join(DATA_DIR, "model.yml")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
PEOPLE_DIR = "people"  # dossier des photos brutes

# Extensions autorisées
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Détecteur Haar pour visages
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(DATASET_DIR, exist_ok=True)

def detect_and_crop_face(img_bgr, size=(200, 200)):
    """Détecter le plus grand visage ; retourner image recadrée (grayscale, resized)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    roi = gray[y:y + h, x:x + w]
    return cv2.resize(roi, size)

def build_training_data():
    """Construire X, y, labels_to_name depuis DB + disque."""
    people, images = fetch_people_and_images()
    if not people:
        return [], np.array([], dtype=np.int32), {}

    id_to_label = {pid: idx for idx, (pid, _) in enumerate(people)}
    labels_to_name = {id_to_label[pid]: name for (pid, name) in people}

    X, y = [], []
    for (person_id, path) in images:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        X.append(img)
        y.append(id_to_label[person_id])

    return X, np.array(y, dtype=np.int32), labels_to_name

def train_and_save_model():
    """Entraîner LBPH et sauvegarder modèle + labels.json."""
    X, y, labels_to_name = build_training_data()
    if len(X) == 0:
        print("[INFO] Aucun échantillon pour l'entraînement.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(X, y)
    os.makedirs(DATA_DIR, exist_ok=True)
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_to_name, f, ensure_ascii=False, indent=2)
    print(f"[OK] Modèle entraîné → {MODEL_PATH}")
    print(f"[OK] Labels sauvegardés → {LABELS_PATH}")

def import_people(root="people"):
    """Scanner 'people/<Nom>/*' et importer toutes les images valides."""
    root_path = Path(root)
    if not root_path.exists():
        print(f"[ERREUR] Dossier '{root}' introuvable.")
        return

    total_ok, total_fail = 0, 0

    for person_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        name = person_dir.name.strip()
        if not name:
            continue

        pid = get_or_create_person_id(name)
        out_dir = Path(DATASET_DIR) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compter ce qui existe déjà pour nommer en séquence
        existing_count = len([f for f in out_dir.iterdir() if f.is_file()])

        for img_path in sorted(person_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in ALLOWED_EXT:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Lecture impossible: {img_path}")
                total_fail += 1
                continue

            face = detect_and_crop_face(img)
            if face is None:
                print(f"[WARN] Aucun visage détecté: {img_path}")
                total_fail += 1
                continue

            existing_count += 1
            save_path = out_dir / f"{name}_{existing_count:03d}.png"
            cv2.imwrite(str(save_path), face)

            add_image_record(pid, str(save_path))
            total_ok += 1
            print(f"[OK] {name} <= {img_path.name} → {save_path.name}")

    print(f"\n[SUMMARY] Import réussi: {total_ok} | Échecs: {total_fail}")
    train_and_save_model()

def main():
    parser = argparse.ArgumentParser(description="Importer toutes les photos depuis 'people/<Nom>/' puis entraîner LBPH.")
    parser.add_argument("--root", default=PEOPLE_DIR, help="Dossier racine des photos (par défaut: people)")
    args = parser.parse_args()
    import_people(args.root)

if __name__ == "__main__":
    import argparse
    main()
