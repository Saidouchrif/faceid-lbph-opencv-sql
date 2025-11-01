# enroll_face_mysql.py
# ------------------------------------------------------------
# Objectif :
# 1) Charger une image fournie (jpg/png)
# 2) Détecter et recadrer le visage (Haar Cascade)
# 3) Sauvegarder l'image recadrée dans data/dataset/<nom>/
# 4) Insérer (personne, image) dans MySQL
# 5) Ré-entraîner un modèle LBPH sur tout le dataset et sauvegarder:
#    - data/model.yml
#    - data/labels.json (mapping label_num -> nom)
# ------------------------------------------------------------
import os
import cv2
import json
import argparse
import numpy as np

from db_utils import get_or_create_person_id, add_image_record, fetch_people_and_images

# --- Chemins ---
DATA_DIR = "data"
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
MODEL_PATH = os.path.join(DATA_DIR, "model.yml")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

# --- Détecteur Haar pour le visage ---
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Assurer les dossiers
os.makedirs(DATASET_DIR, exist_ok=True)

def detect_and_crop_face(img_bgr, size=(200, 200)):
    """
    Détecter le plus grand visage dans l'image, retourner une version
    recadrée en niveaux de gris, redimensionnée à 'size'.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # Choisir le plus grand visage (souvent le plus pertinent)
    x, y, w, h = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
    face = gray[y:y+h, x:x+w]
    return cv2.resize(face, size)

def build_training_data():
    """
    Lire toutes les personnes/images depuis la DB et charger
    les images recadrées depuis le disque pour construire:
    - X: liste d'images (np.array)
    - y: labels numériques
    - labels_to_name: dict {label_num: nom}
    """
    people, images = fetch_people_and_images()
    if not people:
        return [], np.array([], dtype=np.int32), {}

    # map DB-id -> label (0..N-1)
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
    """Entraîner LBPH sur tout le dataset et sauvegarder modèle + labels."""
    X, y, labels_to_name = build_training_data()
    if len(X) == 0:
        print("[INFO] Aucun échantillon pour l'entraînement.")
        return

    # Créer le reconnaisseur LBPH (module 'cv2.face' via opencv-contrib)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(X, y)

    os.makedirs(DATA_DIR, exist_ok=True)
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels_to_name, f, ensure_ascii=False, indent=2)

    print(f"[OK] Modèle entraîné → {MODEL_PATH}")
    print(f"[OK] Labels sauvegardés → {LABELS_PATH}")

def main():
    parser = argparse.ArgumentParser(description="Enrôler une image de visage et ré-entraîner le modèle LBPH.")
    parser.add_argument("--name", required=True, help="Nom de la personne (ex: Ayoub)")
    parser.add_argument("--image", required=True, help="Chemin de l'image (jpg/png)")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print("[ERREUR] Impossible de lire l'image:", args.image)
        return

    face = detect_and_crop_face(img)
    if face is None:
        print("[ERREUR] Aucun visage détecté correctement. Essaie une image plus claire/centrée.")
        return

    # Sauvegarder l'image recadrée dans data/dataset/<nom>/
    person_dir = os.path.join(DATASET_DIR, args.name)
    os.makedirs(person_dir, exist_ok=True)
    idx = len([f for f in os.listdir(person_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]) + 1
    save_path = os.path.join(person_dir, f"{args.name}_{idx:03d}.png")
    cv2.imwrite(save_path, face)

    # Mettre à jour la DB
    pid = get_or_create_person_id(args.name)
    add_image_record(pid, save_path)

    print(f"[OK] Image enrôlée pour '{args.name}' → {save_path}")
    # Ré-entraîner
    train_and_save_model()

if __name__ == "__main__":
    main()
