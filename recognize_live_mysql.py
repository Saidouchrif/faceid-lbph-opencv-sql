# recognize_live_mysql.py - VERSION CORRIGÉE
# ------------------------------------------------------------
# Objectif :
# - Charger le modèle LBPH + le mapping des labels
# - Ouvrir la webcam de manière robuste (Windows: DirectShow d'abord)
# - Lire des frames de façon sûre, détecter les visages (Haar)
# - Prédire l'identité via LBPH et vérifier l'existence en base (MySQL)
# - Afficher un bandeau global "TOI: PRESENT/ABSENT" selon TARGET_NAME
# Contrôles:
#   q / ESC : quitter
#   c       : re-sélectionner/rouvrir la caméra
# ------------------------------------------------------------

import cv2
import json
import time
import os
import numpy as np
from db_utils import person_exists

# --- Chemins et constantes ---
DATA_DIR = "data"
MODEL_PATH = f"{DATA_DIR}/model.yml"
LABELS_PATH = f"{DATA_DIR}/labels.json"

TARGET_NAME = "Ayoub"    # ← mets ici ton nom cible
THRESHOLD   = 70.0       # LBPH: plus la "conf" est petite, mieux c'est (ajuste selon tes données)

# Détecteur Haar pour visages - VERSION CORRIGÉE
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
print(f"[INFO] Chargement du détecteur Haar: {haar_cascade_path}")
FACE_CASCADE = cv2.CascadeClassifier(haar_cascade_path)

# Vérifier que le détecteur est chargé
if FACE_CASCADE.empty():
    print("[ERREUR] Impossible de charger le détecteur Haar!")
    exit(1)
else:
    print("[INFO] Détecteur Haar chargé avec succès")

def load_model_and_labels():
    """
    Charger le modèle LBPH (opencv-contrib) + le mapping label->nom depuis labels.json.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_to_name = json.load(f)
    # Normaliser les clés JSON en int
    labels_to_name = {int(k): v for k, v in labels_to_name.items()}
    return recognizer, labels_to_name

def open_fixed_cam():
    """
    Ouvrir la caméra frontale (webcam intégrée) de façon optimisée.
    Spécialement conçu pour les caméras frontales d'ordinateurs portables.
    """
    print("[INFO] Ouverture caméra frontale...")
    
    # Configurations spécifiques pour caméras frontales
    configs = [
        # Config 1: DirectShow basique (le plus courant)
        {
            'api': cv2.CAP_DSHOW,
            'name': 'DirectShow',
            'width': 640, 'height': 480, 'fps': 30,
            'fourcc': None, 'buffer': 1
        },
        # Config 2: DirectShow avec MJPG
        {
            'api': cv2.CAP_DSHOW, 
            'name': 'DirectShow+MJPG',
            'width': 640, 'height': 480, 'fps': 30,
            'fourcc': 'MJPG', 'buffer': 1
        },
        # Config 3: Media Foundation
        {
            'api': cv2.CAP_MSMF,
            'name': 'MediaFoundation', 
            'width': 640, 'height': 480, 'fps': 30,
            'fourcc': None, 'buffer': 1
        },
        # Config 4: Résolution plus basse
        {
            'api': cv2.CAP_DSHOW,
            'name': 'DirectShow_LowRes',
            'width': 320, 'height': 240, 'fps': 30,
            'fourcc': None, 'buffer': 1
        },
        # Config 5: Auto (dernier recours)
        {
            'api': cv2.CAP_ANY,
            'name': 'Auto',
            'width': 640, 'height': 480, 'fps': 15,
            'fourcc': None, 'buffer': 2
        }
    ]
    
    # Tester uniquement l'index 0 (caméra frontale)
    for config in configs:
        print(f"[DEBUG] Test: {config['name']} - {config['width']}x{config['height']}")
        
        try:
            # Ouvrir la caméra
            cap = cv2.VideoCapture(0, config['api'])
            
            if not cap.isOpened():
                print(f"[DEBUG] {config['name']} ne s'ouvre pas")
                continue
            
            # Configuration des propriétés
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['height'])
            cap.set(cv2.CAP_PROP_FPS, config['fps'])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, config['buffer'])
            
            # Auto-exposition et auto-focus pour caméras frontales
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto-exposition
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)         # Auto-focus
            
            # FOURCC si spécifié
            if config['fourcc']:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config['fourcc']))
            
            # Test de lecture avec patience (caméras frontales sont parfois lentes)
            print(f"[DEBUG] Test de lecture...")
            success_count = 0
            
            for attempt in range(15):  # Plus d'essais pour caméras frontales
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    # Vérifier que l'image n'est pas complètement noire
                    if np.std(frame) > 5:  # Image avec du contenu
                        success_count += 1
                        if success_count >= 3:  # 3 frames valides suffisent
                            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            actual_fps = cap.get(cv2.CAP_PROP_FPS)
                            
                            print(f"[INFO] Caméra frontale OK!")
                            print(f"[INFO] Config: {config['name']}")
                            print(f"[INFO] Résolution: {actual_w}x{actual_h}")
                            print(f"[INFO] FPS: {actual_fps:.1f}")
                            return cap
                
                time.sleep(0.1)  # Attendre plus longtemps entre les essais
            
            print(f"[DEBUG] {config['name']} - Frames invalides ({success_count}/15)")
            cap.release()
            
        except Exception as e:
            print(f"[DEBUG] {config['name']} - Erreur: {e}")
            if 'cap' in locals():
                cap.release()
    
    print("[ERREUR] Impossible d'ouvrir la caméra frontale")
    print("[INFO] Solutions à essayer:")
    print("  1. Vérifier Paramètres > Confidentialité > Caméra")
    print("  2. Fermer toutes les apps utilisant la caméra")
    print("  3. Redémarrer l'ordinateur")
    print("  4. Mettre à jour les pilotes de caméra")
    return None

def diagnose_cameras():
    """
    Diagnostique les caméras disponibles sur le système.
    Affiche des informations détaillées pour le débogage.
    """
    print("[DIAGNOSTIC] Recherche des caméras disponibles...")
    
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MediaFoundation"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    found_cameras = []
    
    for idx in range(5):  # Tester jusqu'à l'index 4
        for api, api_name in backends:
            try:
                cap = cv2.VideoCapture(idx, api)
                if cap.isOpened():
                    # Tester la lecture d'un frame
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        found_cameras.append((idx, api_name, w, h, fps))
                        print(f"[DIAGNOSTIC] ✅ Caméra trouvée: idx={idx}, backend={api_name}, résolution={w}x{h}, fps={fps:.1f}")
                    else:
                        print(f"[DIAGNOSTIC] ⚠️  Caméra idx={idx}, backend={api_name} s'ouvre mais ne lit pas de frames")
                    cap.release()
                else:
                    print(f"[DIAGNOSTIC] ❌ Caméra idx={idx}, backend={api_name} ne s'ouvre pas")
            except Exception as e:
                print(f"[DIAGNOSTIC] ❌ Erreur idx={idx}, backend={api_name}: {e}")
    
    if not found_cameras:
        print("[DIAGNOSTIC] ❌ Aucune caméra fonctionnelle trouvée!")
        print("[DIAGNOSTIC] Vérifiez:")
        print("  - Que la caméra est branchée et allumée")
        print("  - Qu'aucune autre application n'utilise la caméra")
        print("  - Les permissions Windows pour accéder à la caméra")
        print("  - Les pilotes de la caméra")
    else:
        print(f"[DIAGNOSTIC] ✅ {len(found_cameras)} caméra(s) fonctionnelle(s) trouvée(s)")
    
    return found_cameras

def safe_read(cap, max_tries=20):
    """
    Lecture "robuste" d'un frame avec quelques tentatives rapides.
    Évite de bloquer si la caméra met un peu de temps à répondre.
    """
    tries = 0
    while tries < max_tries:
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        tries += 1
        time.sleep(0.02)  # 20 ms entre tentatives
    return False, None

def main():
    # --- Charger modèle + labels ---
    try:
        recognizer, labels_to_name = load_model_and_labels()
    except Exception as e:
        print("[ERREUR] Modèle/labels introuvables. Entraîne d'abord (enroll/import).")
        print(e)
        return

    # --- Ouvrir la caméra de manière robuste ---
    cap = open_fixed_cam()
    if cap is None:
        print("[ERREUR] Aucune caméra accessible.")
        print("[INFO] Lancement du diagnostic automatique...")
        diagnose_cameras()
        return

    print("[INFO] Contrôles: 'q' pour quitter | 'c' pour re-sélectionner la caméra.")

    while True:
        # Lecture directe du flux (plus simple et fiable)
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Lecture échouée. Tentative de re-ouverture de la caméra...")
            cap.release()
            cap = open_fixed_cam()
            if cap is None:
                print("[ERREUR] Impossible de rouvrir la caméra.")
                break
            continue

        # --- Pipeline de détection/reconnaissance ---
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détection de visages avec paramètres plus permissifs
        faces = FACE_CASCADE.detectMultiScale(
            gray, 
            scaleFactor=1.05,      # Plus petit = plus sensible
            minNeighbors=3,        # Plus petit = plus de détections
            minSize=(30, 30),      # Plus petit = détecte des visages plus petits
            maxSize=(300, 300)     # Limite la taille max
        )
        
        # Afficher seulement quand des visages sont détectés
        if len(faces) > 0:
            print(f"[INFO] {len(faces)} visage(s) détecté(s)")

        is_target_present = False
        global_text = "TOI: ABSENT ❌"

        # Dessiner tous les visages détectés (même sans reconnaissance)
        for (x, y, w, h) in faces:
            # Rectangle vert pour chaque visage détecté
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "VISAGE DETECTE", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            try:
                # Préparation du ROI pour LBPH (taille 200x200 en niveaux de gris)
                roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

                # Prédiction LBPH : label et "confidence"
                label_pred, conf = recognizer.predict(roi)

                if conf < THRESHOLD and label_pred in labels_to_name:
                    name = labels_to_name[label_pred]
                    
                    # Mode test sans base de données pour éviter les erreurs
                    try:
                        exists = person_exists(name)
                    except:
                        exists = True  # Mode test
                        print(f"[INFO] Mode test - pas de vérification DB pour {name}")

                    status = f"{name} (conf={conf:.1f})"
                    status += " — Succès ✅" if exists else " — Non trouvé ❌"

                    if name == TARGET_NAME and exists:
                        is_target_present = True

                    color = (0, 255, 0) if exists else (0, 0, 255)
                else:
                    status = f"Inconnu (conf={conf:.1f})"
                    color = (0, 0, 255)

                # Mettre à jour le texte du visage
                cv2.putText(frame, status, (x, y-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                            
            except Exception as e:
                print(f"[WARN] Erreur reconnaissance: {e}")
                cv2.putText(frame, "ERREUR RECONNAISSANCE", (x, y-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Bandeau global "toi présent/absent"
        if is_target_present:
            global_text = "TOI: PRESENT ✅"

        cv2.putText(frame, global_text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 0) if is_target_present else (0, 0, 255), 3, cv2.LINE_AA)

        # Ajouter des informations sur le frame
        height, width = frame.shape[:2]
        cv2.putText(frame, f"Resolution: {width}x{height}", (20, height-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Appuyez sur 'q' pour quitter", (20, height-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # FORCER l'affichage de la fenêtre avec gestion d'erreur
        window_name = "🎯 Camera Frontale - Reconnaissance Faciale"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            cv2.imshow(window_name, frame)
        except Exception as e:
            print(f"[WARN] Erreur affichage: {e}")
            # Fallback sans émojis
            cv2.imshow("Camera Frontale - Reconnaissance", frame)

        # Gestion clavier
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC ou q → quitter
            print("[INFO] Arrêt demandé par l'utilisateur")
            break
        if key == ord('c'):        # c → rouvrir/re-sélectionner la caméra
            print("[INFO] Ré-ouverture de la caméra...")
            cap.release()
            cap = open_fixed_cam()
            if cap is None:
                print("[ERREUR] Impossible de rouvrir la caméra.")
                diagnose_cameras()
                break
        if key == ord('d'):        # d → diagnostic des caméras
            print("[INFO] Diagnostic des caméras...")
            diagnose_cameras()

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
