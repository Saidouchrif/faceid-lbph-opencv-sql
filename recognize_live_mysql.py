# recognize_live_mysql.py
import cv2, json, time
from db_utils import person_exists

DATA_DIR = "data"
MODEL_PATH = f"{DATA_DIR}/model.yml"
LABELS_PATH = f"{DATA_DIR}/labels.json"
TARGET_NAME = "Ayoub"
THRESHOLD = 70.0
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_model_and_labels():
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels_to_name = {int(k): v for k, v in json.load(f).items()}
    return rec, labels_to_name

# --- NEW: فتح الكاميرا بذكاء (Windows: جرب DSHOW ثم MSMF) ---
def open_cam_try():
    for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        for idx in (0, 1, 2, 3):
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                # إعداد اختياري
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                # warmup سريع
                ok, frame = cap.read()
                if ok:
                    print(f"[INFO] Camera opened: index={idx}, api={api}")
                    return cap, idx, api
                cap.release()
    return None, None, None

# --- NEW: قراءة آمنة مع مهلة قصيرة ---
def safe_read(cap, max_tries=20):
    tries = 0
    while tries < max_tries:
        ok, frame = cap.read()
        if ok:
            return True, frame
        tries += 1
        time.sleep(0.02)  # 20ms
    return False, None

def main():
    try:
        recognizer, labels_to_name = load_model_and_labels()
    except Exception as e:
        print("[ERREUR] Modèle/labels introuvables. Entraîne d'abord avec enroll/import.")
        print(e); return

    cap, cam_idx, cam_api = open_cam_try()
    if cap is None:
        print("[ERREUR] Aucune caméra disponible (permissions? occupée par autre app?)")
        return

    print("[INFO] Appuie sur 'c' لتغيير الكاميرا، 'q' للخروج.")
    is_target_present = False
    global_text = "TOI: ABSENT ❌"

    while True:
        ok, frame = safe_read(cap)
        if not ok:
            print("[WARN] Lecture échouée. On réessaie avec une autre caméra/backend...")
            cap.release()
            cap, cam_idx, cam_api = open_cam_try()
            if cap is None:
                print("[ERREUR] Impossible de rouvrir la caméra."); break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        is_target_present = False
        global_text = "TOI: ABSENT ❌"

        for (x, y, w, h) in faces:
            roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label_pred, conf = recognizer.predict(roi)

            if conf < THRESHOLD and label_pred in labels_to_name:
                name = labels_to_name[label_pred]
                exists = person_exists(name)
                status = f"{name} (conf={conf:.1f})" + (" — Succès: trouvé en base ✅" if exists else " — Non trouvé en base ❌")
                if name == TARGET_NAME and exists: is_target_present = True
                color = (0,255,0) if exists else (0,0,255)
            else:
                status = f"Inconnu (conf={conf:.1f}) — Non trouvé en base ❌"
                color = (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        if is_target_present: global_text = "TOI: PRESENT ✅"
        cv2.putText(frame, global_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,255,0) if is_target_present else (0,0,255), 3, cv2.LINE_AA)

        cv2.imshow("Reconnaissance (LBPH + MySQL)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')): 
            break
        if key == ord('c'): 
            print("[INFO] Switching camera...")
            cap.release()
            cap, cam_idx, cam_api = open_cam_try()
            if cap is None:
                print("[ERREUR] Impossible de rouvrir la caméra."); break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
