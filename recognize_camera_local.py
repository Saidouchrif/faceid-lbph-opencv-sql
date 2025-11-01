# recognize_camera_local.py
# Script pour reconnaître les visages en direct depuis la caméra
# Fonctionne directement sur la machine hôte (pas de Docker)

import cv2
import json
import os
import sys
from pathlib import Path

# Ajouter le répertoire courant au chemin
sys.path.insert(0, str(Path(__file__).parent))

# --- Chemins et constantes ---
DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "model.yml")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")

THRESHOLD = 70.0  # LBPH: plus la "conf" est petite, mieux c'est

# Détecteur Haar pour visages
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
print(f"[INFO] Chargement du détecteur Haar: {haar_cascade_path}")
FACE_CASCADE = cv2.CascadeClassifier(haar_cascade_path)

if FACE_CASCADE.empty():
    print("[ERREUR] Impossible de charger le détecteur Haar!")
    sys.exit(1)
else:
    print("[INFO] Détecteur Haar chargé avec succès")

def load_model_and_labels():
    """Charger le modèle LBPH + le mapping label->nom depuis labels.json."""
    if not os.path.exists(MODEL_PATH):
        print(f"[ERREUR] Modèle non trouvé: {MODEL_PATH}")
        return None, None
    
    if not os.path.exists(LABELS_PATH):
        print(f"[ERREUR] Labels non trouvés: {LABELS_PATH}")
        return None, None
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_to_name = json.load(f)
    
    # Normaliser les clés JSON en int
    labels_to_name = {int(k): v for k, v in labels_to_name.items()}
    return recognizer, labels_to_name

def open_camera():
    """Ouvrir la caméra."""
    print("[INFO] Ouverture de la caméra...")
    
    # Essayer différents indices de caméra
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"[OK] Caméra trouvée à l'index {idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            return cap
    
    print("[ERREUR] Aucune caméra trouvée!")
    return None

def recognize_faces(cap, recognizer, labels_to_name):
    """Boucle principale de reconnaissance faciale."""
    print("[INFO] Appuyez sur 'q' ou 'ESC' pour quitter")
    print("[INFO] Appuyez sur 'c' pour changer de caméra")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Impossible de lire la frame")
            break
        
        # Redimensionner pour accélérer le traitement
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        # Traiter chaque visage détecté
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))
            
            # Prédire l'identité
            label, confidence = recognizer.predict(roi_gray)
            
            # Déterminer le nom et la couleur
            if confidence < THRESHOLD:
                name = labels_to_name.get(label, "Inconnu")
                color = (0, 255, 0)  # Vert = reconnu
                text = f"{name} ({confidence:.1f})"
            else:
                name = "Inconnu"
                color = (0, 0, 255)  # Rouge = non reconnu
                text = f"Inconnu ({confidence:.1f})"
            
            # Dessiner le rectangle et le texte
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Afficher la frame
        cv2.imshow("Reconnaissance Faciale - Appuyez sur Q pour quitter", frame)
        
        # Gestion des touches
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q ou ESC
            print("[INFO] Fermeture...")
            break
        elif key == ord('c'):
            print("[INFO] Changement de caméra...")
            cap.release()
            cap = open_camera()
            if cap is None:
                break

def main():
    # Charger le modèle et les labels
    recognizer, labels_to_name = load_model_and_labels()
    if recognizer is None or labels_to_name is None:
        print("[ERREUR] Impossible de charger le modèle. Avez-vous exécuté import_people_mysql.py?")
        sys.exit(1)
    
    print(f"[OK] Modèle chargé avec {len(labels_to_name)} personnes")
    print(f"[OK] Personnes reconnues: {', '.join(labels_to_name.values())}")
    
    # Ouvrir la caméra
    cap = open_camera()
    if cap is None:
        sys.exit(1)
    
    try:
        # Boucle de reconnaissance
        recognize_faces(cap, recognizer, labels_to_name)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Caméra fermée")

if __name__ == "__main__":
    main()
