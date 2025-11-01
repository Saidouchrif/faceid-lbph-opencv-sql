# recognize_live_mysql.py
# ------------------------------------------------------------
# Objectif :
# - Charger modèle LBPH + labels
# - Ouvrir la webcam
# - Détecter les visages (Haar), prédire les noms via LBPH
# - Vérifier si le nom prédit existe en base (persons)
# - Afficher un message global: "TOI: PRESENT/ABSENT" selon TARGET_NAME
# ------------------------------------------------------------
import cv2
import json
import time
from db_utils import person_exists

DATA_DIR = "data"
MODEL_PATH = f"{DATA_DIR}/model.yml"
LABELS_PATH = f"{DATA_DIR}/labels.json"

# Mets ton nom ici pour la vérification "est-ce moi ?"
TARGET_NAME = "Ayoub"

# LBPH: plus 'conf' est petit, mieux c'est. Ajuste selon tes données
THRESHOLD = 70.0

# Détecteur Haar
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def load_model_and_labels():
    """Charger le modèle LBPH et le mapping labels->nom."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_to_name = json.load(f)
    # S'assurer que les clés sont des int (si le JSON les a sauvées en str)
    labels_to_name = {int(k): v for k, v in labels_to_name.items()}
    return recognizer, labels_to_name

def main():
    try:
        recognizer, labels_to_name = load_model_and_labels()
    except Exception as e:
        print("[ERREUR] Modèle/labels introuvables. Entraîne d'abord avec enroll_face_mysql.py")
        print(e)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la webcam.")
        return

    last_status = ""
    last_change = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        is_target_present = False
        global_text = "TOI: ABSENT ❌"

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))

            label_pred, conf = recognizer.predict(roi)

            if conf < THRESHOLD and label_pred in labels_to_name:
                name = labels_to_name[label_pred]
                exists = person_exists(name)

                status = f"{name} (conf={conf:.1f})"
                if exists:
                    status += " — Succès: trouvé en base ✅"
                else:
                    status += " — Non trouvé en base ❌"

                if name == TARGET_NAME and exists:
                    is_target_present = True

                color = (0, 255, 0) if exists else (0, 0, 255)
            else:
                name = "Inconnu"
                status = f"Inconnu (conf={conf:.1f}) — Non trouvé en base ❌"
                color = (0, 0, 255)

            # Dessin
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, status, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        if is_target_present:
            global_text = "TOI: PRESENT ✅"

        # Bandeau global
        cv2.putText(frame, global_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,255,0) if is_target_present else (0,0,255), 3, cv2.LINE_AA)

        cv2.imshow("Reconnaissance (LBPH + MySQL)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC ou q pour quitter
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
