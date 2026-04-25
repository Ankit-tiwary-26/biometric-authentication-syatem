import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time

st.set_page_config(page_title="Biometric Security System", layout="centered")

# ---------- UI ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #020617, #0f172a); color:white;}
h1 {text-align:center; color:cyan;}
</style>
""", unsafe_allow_html=True)

st.title("🔐 AI Biometric Security System")

DB_FILE = "database.json"

def load_db():
    if not os.path.exists(DB_FILE):
        return {}
    return json.load(open(DB_FILE))

def save_db(db):
    json.dump(db, open(DB_FILE, "w"))

db = load_db()

# ---------- FACE MATCH ----------
def compare_faces(img1, img2):
    img1 = cv2.resize(img1, (100,100))
    img2 = cv2.resize(img2, (100,100))
    diff = np.mean((img1 - img2) ** 2)
    return diff < 2000, diff

# ---------- MEDIAPIPE ----------
mp_mesh = mp.solutions.face_mesh
mp_face = mp.solutions.face_detection

# ---------- AUTH SYSTEM ----------
def run_authentication(username):
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()

    blink = False
    head_move = False
    lip_move = False
    single_face = False
    spoof_ok = True

    initial_x = None
    mouth_history = []

    with mp_mesh.FaceMesh(refine_landmarks=True) as mesh, \
         mp_face.FaceDetection() as face_det:

        start = time.time()

        while time.time() - start < 8:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---- FACE COUNT ----
            faces = face_det.process(rgb)
            if faces.detections and len(faces.detections) == 1:
                single_face = True
            else:
                single_face = False

            # ---- SPOOF DETECTION ----
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if np.var(gray) < 300:
                spoof_ok = False

            # ---- LANDMARKS ----
            results = mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark

                    # 👁️ Blink detection
                    eye = lm[159].y - lm[145].y
                    if eye < 0.01:
                        blink = True

                    # 👄 Lip movement (FIXED)
                    mouth = lm[13].y - lm[14].y
                    mouth_history.append(mouth)

                    if len(mouth_history) > 15:
                        mouth_history.pop(0)

                    if len(mouth_history) >= 10:
                        variation = max(mouth_history) - min(mouth_history)
                        if variation > 0.02:
                            lip_move = True

                    # ↔️ Head movement
                    x = lm[1].x
                    if initial_x is None:
                        initial_x = x
                    elif abs(x - initial_x) > 0.05:
                        head_move = True

            # ---- UI ----
            h, w, _ = frame.shape
            cv2.rectangle(frame, (50,50), (w-50,h-50), (0,255,255), 2)

            cv2.putText(frame, "Blink + Turn Head + Say Name", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            frame_placeholder.image(frame, channels="BGR")

        ret, final_frame = cap.read()
        cap.release()

    # ---- FACE MATCH ----
    stored = cv2.imread(db[username]["face"])
    verified, diff = compare_faces(stored, final_frame)

    return {
        "blink": blink,
        "head": head_move,
        "lip": lip_move,
        "face_ok": single_face,
        "spoof": spoof_ok,
        "face_match": verified,
        "distance": diff
    }

# ---------- MENU ----------
menu = st.sidebar.radio("Menu", ["Register", "Login"])

# ---------- REGISTER ----------
if menu == "Register":
    st.subheader("📝 Register")

    username = st.text_input("Enter Name")
    img = st.camera_input("Capture Face")

    if st.button("Save User"):
        if username and img:
            path = f"{username}.jpg"
            with open(path, "wb") as f:
                f.write(img.getbuffer())

            db[username] = {"face": path}
            save_db(db)

            st.success("User Registered")
        else:
            st.warning("Complete all fields")

# ---------- LOGIN ----------
if menu == "Login":
    st.subheader("🔑 Login")

    username = st.text_input("Enter Name")

    if st.button("Start Secure Login"):
        if username not in db:
            st.error("User not found")
        else:
            st.warning("👉 Look at camera, BLINK, turn head, and SAY your name clearly")

            result = run_authentication(username)

            score = 0

            if result["face_ok"]:
                score += 0.2
            else:
                st.error("❌ Multiple faces detected")

            if result["blink"]:
                score += 0.2
            else:
                st.error("❌ No blink detected")

            if result["head"]:
                score += 0.2
            else:
                st.error("❌ No head movement")

            if result["lip"]:
                score += 0.2
                st.success("👄 Speech detected")
            else:
                st.error("❌ No speech detected")

            if result["spoof"]:
                score += 0.1
            else:
                st.error("❌ Spoof detected")

            if result["face_match"]:
                score += 0.1
            else:
                st.error("❌ Face mismatch")

            st.write(f"🔐 Trust Score: {score:.2f}")

            if score > 0.7:
                st.success("🔓 ACCESS GRANTED")
            else:
                st.error("🚫 ACCESS DENIED")