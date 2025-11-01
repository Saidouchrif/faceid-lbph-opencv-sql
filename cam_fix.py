# cam_fix.py — Windows webcam fixer/diag
import cv2, time

APIS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
IDX  = [0,1,2,3]
RES  = [(640,480),(1280,720)]
FOUR = [None, 'MJPG', 'YUY2']  # MJPG مهم بزاف فـ Windows

def try_combo(i, api, res, four):
    cap = cv2.VideoCapture(i, api)
    if not cap.isOpened():
        return (False, 0, None)
    w,h = res
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if four:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*four))
    ok_count = 0
    t0 = time.time()
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_count += 1
        if time.time() - t0 > 3:
            break
    # خزن فريم باش نعرضوه لحظة
    last = frame if ok_count else None
    cap.release()
    return (ok_count>0, ok_count, last)

best = None
for api in APIS:
    for i in IDX:
        for res in RES:
            for four in FOUR:
                ok, cnt, last = try_combo(i, api, res, four)
                print(f"idx={i}, api={api}, res={res}, fourcc={four} -> frames_ok={cnt}")
                if ok and (best is None or cnt > best[0]):
                    best = (cnt, i, api, res, four, last)

print("\n=== RESULT ===")
if best:
    cnt,i,api,res,four,last = best
    print(f"USE -> index={i}, api={api}, res={res}, fourcc={four}, frames_ok={cnt}")
    if last is not None:
        cv2.imshow(f"OK index={i} api={api} res={res} four={four}", last)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()
else:
    print("Aucune combinaison ناجحة. سدّ أي برنامج شاد الكاميرا، وتأكد من صلاحيات Windows.")
