"""
BioGuard Pro — Multi-Factor Biometric Authentication
======================================================
Face + Voice + PIN authentication system with a gated security flow,
role-based dashboard, and full audit trail.

DEPLOYMENT NOTE (read this first):
  The original build called cv2.VideoCapture(0) and sounddevice.rec()
  directly. Those APIs open hardware attached to whatever machine is
  running the Python process. On your laptop that's your own webcam/mic,
  so it looked fine — but on Streamlit Community Cloud / any server
  deployment, that machine has no camera or microphone, so login would
  silently fail or crash for every real user.

  This version captures media entirely in the *user's browser* via
  Streamlit's native widgets (st.camera_input / st.audio_input) and
  ships the resulting bytes to the server for analysis. That's the
  only way biometric capture works once this is actually deployed.

Author: AIML Engineering — BioGuard Pro v4.0 (Cloud Edition)
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import json, os, time, hashlib, uuid, io
from datetime import datetime, timedelta

# ── Optional MongoDB ────────────────────────────────────────────────────────
try:
    from pymongo import MongoClient
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

# ── Voice recognition (speech-to-text) ──────────────────────────────────────
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ── Audio / MFCC for voiceprint ─────────────────────────────────────────────
try:
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioGuard Pro – Voice + Biometric",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
:root{--bg0:#05080f;--bg1:#0b1120;--bg2:#111827;--accent:#00e5ff;--accent2:#00ff9d;--warn:#ffd740;--danger:#ff4757;--text:#e2e8f0;--muted:#64748b;}
html,body,.stApp{background:var(--bg0)!important;font-family:'IBM Plex Sans',sans-serif;color:var(--text);}
[data-testid="stSidebar"]{background:var(--bg1)!important;border-right:1px solid #1e293b;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3{font-family:'IBM Plex Mono',monospace!important;color:var(--accent)!important;letter-spacing:-0.5px;}
h2{color:var(--text)!important;font-size:1.1rem!important;}
[data-testid="stMetric"]{background:var(--bg2)!important;border:1px solid #1e293b;border-radius:8px;padding:12px 16px!important;}
[data-testid="stMetric"] label{color:var(--muted)!important;font-size:0.75rem!important;}
[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--accent)!important;}
.stButton>button{background:transparent!important;border:1px solid var(--accent)!important;color:var(--accent)!important;font-family:'IBM Plex Mono',monospace!important;border-radius:4px!important;transition:all 0.2s;width:100%;}
.stButton>button:hover{background:var(--accent)!important;color:var(--bg0)!important;}
.stTextInput input,.stSelectbox select,.stTextArea textarea{background:var(--bg2)!important;border:1px solid #1e293b!important;color:var(--text)!important;border-radius:4px!important;}
.ok{background:#052e16;border-left:3px solid var(--accent2);padding:8px 12px;border-radius:4px;margin:4px 0;font-size:0.85rem;}
.fail{background:#2d0b0b;border-left:3px solid var(--danger);padding:8px 12px;border-radius:4px;margin:4px 0;font-size:0.85rem;}
.warn{background:#2d2000;border-left:3px solid var(--warn);padding:8px 12px;border-radius:4px;margin:4px 0;font-size:0.85rem;}
.score-bg{background:#1e293b;border-radius:4px;height:10px;margin:8px 0;}
.score-fg{border-radius:4px;height:10px;}
.audit-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1e293b;font-size:0.8rem;}
hr{border-color:#1e293b!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--bg1)!important;border-radius:8px;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-family:'IBM Plex Mono',monospace!important;}
.stTabs [aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important;}
.streamlit-expanderHeader{color:var(--accent)!important;font-family:'IBM Plex Mono',monospace!important;}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.7rem;font-family:'IBM Plex Mono',monospace;border:1px solid #1e293b;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DB_FILE       = "bioguard_db.json"
AUDIT_FILE    = "audit_log.json"
SETTINGS_FILE = "settings.json"
FACES_DIR     = "faces"
VOICES_DIR    = "voiceprints"
os.makedirs(FACES_DIR, exist_ok=True)
os.makedirs(VOICES_DIR, exist_ok=True)

ROLES = ["viewer", "operator", "admin", "superadmin"]
ROLE_COLORS = {"viewer":"#64748b","operator":"#00e5ff","admin":"#ffd740","superadmin":"#ff4757"}

DEFAULT_SETTINGS = {
    "trust_threshold":   0.72,
    "lockout_attempts":  3,
    "lockout_minutes":   5,
    "voice_threshold":   35.0,
    "require_pin":       True,
    "require_voice":     True,
    "antispoof_mandatory": True,
}

# ─────────────────────────────────────────────────────────────────────────────
#  SETTINGS (admin-configurable, persisted to disk)
# ─────────────────────────────────────────────────────────────────────────────
def load_settings() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE) as f:
            s = json.load(f)
        merged = dict(DEFAULT_SETTINGS)
        merged.update(s)
        return merged
    except Exception:
        return dict(DEFAULT_SETTINGS)

def save_settings(settings: dict):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
#  MONGODB LAYER  (falls back to JSON if not configured)
# ─────────────────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "")   # set in environment / Streamlit secrets

def get_mongo():
    """Return (users_col, audit_col) or (None, None) if not configured."""
    if not MONGO_AVAILABLE or not MONGO_URI:
        return None, None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        db = client["bioguard"]
        return db["users"], db["audit"]
    except Exception:
        return None, None

def load_db() -> dict:
    users_col, _ = get_mongo()
    if users_col is not None:
        return {u["_id"]: u for u in users_col.find()}
    if not os.path.exists(DB_FILE):
        return {}
    with open(DB_FILE) as f:
        return json.load(f)

def save_db(db: dict):
    users_col, _ = get_mongo()
    if users_col is not None:
        for uid, udata in db.items():
            users_col.replace_one({"_id": uid}, {"_id": uid, **udata}, upsert=True)
        return
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def load_audit() -> list:
    _, audit_col = get_mongo()
    if audit_col is not None:
        return list(audit_col.find({}, {"_id": 0}).sort("ts", -1).limit(500))
    if not os.path.exists(AUDIT_FILE):
        return []
    with open(AUDIT_FILE) as f:
        return json.load(f)

def save_audit(log: list):
    _, audit_col = get_mongo()
    if audit_col is not None:
        return   # individual inserts handled in add_audit_event
    with open(AUDIT_FILE, "w") as f:
        json.dump(log[-500:], f, indent=2)

def add_audit_event(username, event, score=None, ip="browser", success=None):
    entry = {
        "ts":      datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user":    username,
        "event":   event,
        "score":   round(score, 3) if score is not None else None,
        "ip":      ip,
        "success": success,
        "id":      str(uuid.uuid4())[:8],
    }
    _, audit_col = get_mongo()
    if audit_col is not None:
        audit_col.insert_one(entry)
        return
    log = load_audit()
    log.append(entry)
    save_audit(log)

def hash_pin(pin: str) -> str:
    return hashlib.sha256(pin.encode()).hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
#  MEDIAPIPE  (static-image mode — we work with photos, not a live stream)
# ─────────────────────────────────────────────────────────────────────────────
mp_face = mp.solutions.face_detection

def decode_camera_image(camera_file) -> np.ndarray | None:
    """Turn a Streamlit camera_input UploadedFile into a BGR numpy image."""
    if camera_file is None:
        return None
    file_bytes = np.frombuffer(camera_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

def analyze_photo(frame: np.ndarray) -> dict:
    """Run face-count + anti-spoof texture/colour checks on a single photo."""
    result = {"single_face": False, "texture_ok": False, "color_ok": False, "face_cx": None}
    if frame is None:
        return result
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    with mp_face.FaceDetection(min_detection_confidence=0.6, model_selection=1) as det:
        res = det.process(rgb)
    count = len(res.detections) if res.detections else 0
    result["single_face"] = (count == 1)
    if count == 1:
        bbox = res.detections[0].location_data.relative_bounding_box
        result["face_cx"] = bbox.xmin + bbox.width / 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result["texture_ok"] = bool(np.var(gray) >= 250)   # printed photo / screen replay ≈ flat texture

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result["color_ok"] = bool(np.mean(hsv[:, :, 1]) >= 20)   # low saturation ≈ grayscale print / screen glare
    return result

def compare_faces(img1: np.ndarray, img2: np.ndarray):
    if img1 is None or img2 is None:
        return False, 9999.0
    img1 = cv2.resize(img1, (100, 100)).astype(float)
    img2 = cv2.resize(img2, (100, 100)).astype(float)
    diff = float(np.mean((img1 - img2) ** 2))
    return diff < 1800, diff

def two_shot_liveness(frame1: np.ndarray, frame2: np.ndarray) -> dict:
    """
    Two-photo liveness check (replaces the old continuous webcam loop):
      Shot 1 = look straight at camera
      Shot 2 = turn head slightly
    A live person's face-centre X shifts noticeably between shots; a static
    photo held up to the camera won't move relative to the frame the same way.
    """
    a1 = analyze_photo(frame1)
    a2 = analyze_photo(frame2)
    head_move = False
    if a1["face_cx"] is not None and a2["face_cx"] is not None:
        head_move = abs(a1["face_cx"] - a2["face_cx"]) > 0.04
    return {
        "single_face": a1["single_face"] and a2["single_face"],
        "texture_ok":  a1["texture_ok"] and a2["texture_ok"],
        "color_ok":    a1["color_ok"] and a2["color_ok"],
        "head_move":   head_move,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  VOICE RECOGNITION ENGINE  (browser-captured audio, via st.audio_input)
# ─────────────────────────────────────────────────────────────────────────────
PASSPHRASE_KEYWORDS = ["open", "access", "unlock", "grant", "bioguard", "secure"]
VOICE_SAMPLE_RATE = 16000

def load_audio_bytes(raw_bytes: bytes):
    """Decode browser-recorded audio (wav/webm/ogg) into a mono float32 array."""
    if not AUDIO_AVAILABLE or not raw_bytes:
        return None
    try:
        audio, _ = librosa.load(io.BytesIO(raw_bytes), sr=VOICE_SAMPLE_RATE, mono=True)
        return audio
    except Exception:
        return None

def extract_mfcc(audio: np.ndarray) -> np.ndarray | None:
    if not AUDIO_AVAILABLE or audio is None or len(audio) == 0:
        return None
    try:
        mfcc = librosa.feature.mfcc(y=audio, sr=VOICE_SAMPLE_RATE, n_mfcc=40)
        return np.mean(mfcc, axis=1)
    except Exception:
        return None

def compare_voiceprints(enrolled: np.ndarray, live: np.ndarray, threshold: float = 35.0):
    """Euclidean distance between MFCC vectors. Lower = more similar voice."""
    if enrolled is None or live is None:
        return False, 9999.0
    dist = float(np.linalg.norm(enrolled - live))
    return dist < threshold, dist

def speech_to_text(raw_bytes: bytes) -> str:
    """Google Web Speech API (free, no key) on browser-recorded audio bytes."""
    if not SR_AVAILABLE or not raw_bytes:
        return ""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(io.BytesIO(raw_bytes)) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data).lower()
    except sr.UnknownValueError:
        return ""            # speech not understood
    except sr.RequestError:
        return "[offline]"   # no internet / API error
    except Exception:
        return ""

def check_passphrase(spoken_text: str, username: str) -> bool:
    if not spoken_text:
        return False
    text = spoken_text.lower()
    name_present = username.replace("_", " ").split()[0].lower() in text
    keyword_present = any(kw in text for kw in PASSPHRASE_KEYWORDS)
    return name_present or keyword_present

def run_voice_auth(username: str, raw_bytes: bytes, enroll_mode: bool = False,
                    voiceprint_threshold: float = 35.0) -> dict:
    """
    Complete voice authentication step, operating on already-captured browser
    audio bytes (no server-side microphone access — see module docstring).
    """
    result = {"passphrase_ok": False, "voiceprint_ok": False,
              "spoken_text": "", "voice_dist": 9999.0, "audio_ok": False}

    if not AUDIO_AVAILABLE:
        result["error"] = "librosa not installed on server"
        return result
    if not SR_AVAILABLE:
        result["error"] = "SpeechRecognition not installed on server"
        return result
    if not raw_bytes:
        result["error"] = "No audio captured — check microphone permission"
        return result

    audio = load_audio_bytes(raw_bytes)
    if audio is None or len(audio) == 0 or np.max(np.abs(audio)) < 0.005:
        result["error"] = "Audio too quiet / empty — please re-record"
        return result
    result["audio_ok"] = True

    spoken = speech_to_text(raw_bytes)
    result["spoken_text"]   = spoken
    result["passphrase_ok"] = check_passphrase(spoken, username)

    live_mfcc = extract_mfcc(audio)
    vp_path = os.path.join(VOICES_DIR, f"{username}_voiceprint.npy")

    if enroll_mode:
        np.save(vp_path, live_mfcc)
        result["voiceprint_ok"] = True
        result["enrolled"] = True
    elif os.path.exists(vp_path):
        stored_mfcc = np.load(vp_path)
        matched, dist = compare_voiceprints(stored_mfcc, live_mfcc, threshold=voiceprint_threshold)
        result["voiceprint_ok"] = matched
        result["voice_dist"] = round(dist, 2)
    else:
        result["voiceprint_ok"] = False
        result["error"] = "No voiceprint enrolled for this user"

    return result

# ─────────────────────────────────────────────────────────────────────────────
#  TRUST SCORE
# ─────────────────────────────────────────────────────────────────────────────
def compute_trust_score(liveness: dict, face_matched: bool, face_diff: float,
                         pin_ok: bool, voice: dict) -> float:
    """
    Weight breakdown (total = 1.0):
      Face pixel match          0.20
      PIN                       0.17
      Voice passphrase match    0.20
      Voiceprint (MFCC) match   0.15
      Head turn (liveness)      0.10
      Single face               0.08
      Anti-spoof texture        0.05
      Anti-spoof colour         0.05
    """
    score = 0.0
    if face_matched:
        score += max(0.0, 0.20 * (1 - min(face_diff, 2000) / 2000))
    score += 0.17 if pin_ok else 0.0
    score += 0.20 if voice.get("passphrase_ok") else 0.0
    score += 0.15 if voice.get("voiceprint_ok") else 0.0
    score += 0.10 if liveness.get("head_move")  else 0.0
    score += 0.08 if liveness.get("single_face") else 0.0
    score += 0.05 if liveness.get("texture_ok") else 0.0
    score += 0.05 if liveness.get("color_ok")   else 0.0
    return min(score, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = dict(authenticated=False, session_user=None, session_role=None,
                 session_start=None, trust_score=0.0, failed_attempts={})
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

settings = load_settings()

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
_users_col, _ = get_mongo()
USING_MONGO = _users_col is not None
with st.sidebar:
    st.markdown("## 🔐 BioGuard Pro")
    st.markdown('<p style="color:#64748b;font-size:0.7rem;font-family:\'IBM Plex Mono\'">AI Biometric + Voice · v4.0 Cloud Edition</p>', unsafe_allow_html=True)

    if USING_MONGO:
        st.markdown('<span style="color:#00ff9d;font-size:0.75rem">● MongoDB connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#ffd740;font-size:0.75rem">● JSON storage mode</span>', unsafe_allow_html=True)

    missing = [n for n, ok in [("librosa", AUDIO_AVAILABLE), ("SpeechRecognition", SR_AVAILABLE)] if not ok]
    if missing:
        st.markdown(f'<span style="color:#ff4757;font-size:0.72rem">⚠ Missing: {", ".join(missing)}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color:#00ff9d;font-size:0.75rem">● Voice engine ready</span>', unsafe_allow_html=True)
    st.divider()

    if st.session_state.authenticated:
        st.markdown(f"**Logged in as:** {st.session_state.session_user}")
        rc = ROLE_COLORS.get(st.session_state.session_role, "#64748b")
        st.markdown(f'<span style="color:{rc};font-weight:600;font-size:0.85rem">● {st.session_state.session_role.upper()}</span>', unsafe_allow_html=True)
        st.markdown(f"**Trust:** `{st.session_state.trust_score:.2f}`")
        elapsed = int((datetime.utcnow() - st.session_state.session_start).total_seconds())
        st.markdown(f"**Session:** `{elapsed//60}m {elapsed%60}s`")
        st.divider()
        if st.button("🚪 Logout"):
            add_audit_event(st.session_state.session_user, "LOGOUT", success=True)
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()
        menu = st.radio("Navigate", ["Dashboard", "Admin Panel"])
    else:
        menu = st.radio("Navigate", ["Login", "Register"])

# ─────────────────────────────────────────────────────────────────────────────
#  REGISTER
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.authenticated and menu == "Register":
    st.title("📝 Register New User")
    col1, col2 = st.columns([1, 1])

    with col1:
        username  = st.text_input("Username (unique)")
        pin       = st.text_input("4-digit PIN", max_chars=4, type="password")
        role      = st.selectbox("Role", ROLES)
        dept      = st.text_input("Department")
        img_input = st.camera_input("📸 Face photo")

        st.markdown("---")
        st.markdown("**🎙 Voiceprint Enrollment**")
        st.info("Say your name plus a keyword like **'open'** or **'bioguard'** — "
                "e.g. *\"Ankit, open bioguard\"*. This is the passphrase you'll repeat every login.")
        voice_clip = st.audio_input("🎤 Record your passphrase")

    with col2:
        st.markdown("### Enrollment status")
        if voice_clip is not None:
            st.audio(voice_clip)  # lets the user hear back what was captured
            if not username:
                st.markdown('<div class="fail">✗ Enter a username before enrolling voice</div>', unsafe_allow_html=True)
            elif not AUDIO_AVAILABLE or not SR_AVAILABLE:
                st.markdown('<div class="fail">✗ Voice engine not installed on server</div>', unsafe_allow_html=True)
            else:
                with st.status("🎙 Uploading and analyzing voice sample…", expanded=True) as status:
                    st.write("📡 Voice clip received from browser microphone")
                    time.sleep(0.3)
                    st.write("🔎 Extracting MFCC voiceprint features…")
                    vr = run_voice_auth(username, voice_clip.getvalue(), enroll_mode=True)
                    time.sleep(0.2)
                    if vr.get("enrolled"):
                        st.write(f"📝 Recognised speech: \"{vr.get('spoken_text','—') or '(unclear)'}\"")
                        status.update(label="✅ Voiceprint enrolled", state="complete")
                        st.session_state["_last_voiceprint_user"] = username
                    else:
                        status.update(label="❌ Enrollment failed", state="error")
                        st.write(vr.get("error", "Unknown error"))
        else:
            st.markdown('<div class="warn">○ Waiting for voice recording…</div>', unsafe_allow_html=True)

    if st.button("✅ Save User"):
        db = load_db()
        if not username:
            st.error("Username required")
        elif not pin or len(pin) != 4 or not pin.isdigit():
            st.error("PIN must be exactly 4 digits")
        elif username in db:
            st.error("Username already exists")
        elif img_input is None:
            st.error("Capture face photo first")
        else:
            face_path = os.path.join(FACES_DIR, f"{username}.jpg")
            with open(face_path, "wb") as f:
                f.write(img_input.getbuffer())
            db[username] = {
                "face": face_path, "pin_hash": hash_pin(pin),
                "role": role, "dept": dept,
                "created_at": datetime.utcnow().isoformat(),
                "active": True, "login_count": 0,
                "has_voiceprint": os.path.exists(os.path.join(VOICES_DIR, f"{username}_voiceprint.npy")),
            }
            save_db(db)
            add_audit_event(username, "REGISTER", success=True)
            st.success(f"✅ {username} registered!")
            if not db[username]["has_voiceprint"]:
                st.warning("No voiceprint enrolled — you can add one later, but voice login will fail until you do.")
            st.balloons()

# ─────────────────────────────────────────────────────────────────────────────
#  LOGIN — gated, step-by-step security flow
# ─────────────────────────────────────────────────────────────────────────────
elif not st.session_state.authenticated and menu == "Login":
    st.title("🔑 Secure Login — Face + Voice + PIN")

    col1, col2 = st.columns([1, 1])
    with col1:
        db = load_db()
        username = st.text_input("Username")
        pin      = st.text_input("PIN", max_chars=4, type="password")

        # ── Gate 0: lockout check (fails fast, no hardware needed) ──────────
        fa = st.session_state.failed_attempts.get(username, {"count": 0, "until": None})
        locked = False
        if fa["until"]:
            until = datetime.fromisoformat(fa["until"])
            if datetime.utcnow() < until:
                locked = True
                st.error(f"🔒 Account temporarily locked. Retry in {int((until-datetime.utcnow()).total_seconds())}s")

        if not locked:
            with st.expander("⚙️ Advanced thresholds (admin defaults shown)"):
                threshold    = st.slider("Trust threshold", 0.5, 0.95, settings["trust_threshold"], 0.05)
                vp_threshold = st.slider("Voice similarity threshold (lower = stricter)", 10.0, 60.0, settings["voice_threshold"], 5.0)

            st.markdown("---")
            st.markdown("**Step 1 — 📷 Face liveness (two photos)**")
            shot1 = st.camera_input("Look straight at the camera", key="shot1")
            shot2 = None
            if shot1 is not None:
                shot2 = st.camera_input("Now turn your head slightly", key="shot2")

            st.markdown("**Step 2 — 🎙 Voice passphrase**")
            st.caption('Say your name + a keyword, e.g. *"Ankit, open bioguard"*')
            voice_clip = st.audio_input("Record your passphrase", key="login_voice")
            if voice_clip is not None:
                st.audio(voice_clip)

            login_btn = st.button("🚀 Verify & Login", type="primary")

            if login_btn:
                # ── Gate 1: user exists & active ─────────────────────────
                if username not in db:
                    st.error("User not found")
                    add_audit_event(username, "LOGIN_UNKNOWN_USER", success=False)
                elif not db[username].get("active", True):
                    st.error("🚫 Account disabled by admin")
                    add_audit_event(username, "LOGIN_DISABLED_ACCOUNT", success=False)
                # ── Gate 2: PIN, if required, checked before touching biometrics ──
                elif settings["require_pin"] and hash_pin(pin) != db[username]["pin_hash"]:
                    st.error("🚫 Incorrect PIN")
                    add_audit_event(username, "LOGIN_BAD_PIN", success=False)
                elif shot1 is None or shot2 is None:
                    st.error("📷 Both face photos are required")
                elif settings["require_voice"] and voice_clip is None:
                    st.error("🎙 Voice passphrase is required")
                else:
                    pin_ok = (not settings["require_pin"]) or (hash_pin(pin) == db[username]["pin_hash"])

                    with col2:
                        st.markdown("### 🔍 Live Verification")

                        # ── Step A: face liveness + match ───────────────
                        with st.status("📷 Analyzing face liveness…", expanded=True) as s1:
                            st.write("🖼 Decoding photos…")
                            f1 = decode_camera_image(shot1)
                            f2 = decode_camera_image(shot2)
                            st.write("🧪 Running anti-spoof + head-turn checks…")
                            liveness = two_shot_liveness(f1, f2)
                            stored_face = cv2.imread(db[username]["face"])
                            face_matched, face_diff = compare_faces(stored_face, f2)
                            st.write(f"🧬 Face similarity distance: {face_diff:.0f} (lower is better)")
                            s1.update(label="✅ Face analysis complete" if liveness["single_face"] else "⚠️ Face analysis complete — issues found",
                                      state="complete")

                        # ── Hard gate: anti-spoof / multi-face is a block, not just a score hit ──
                        spoof_block = settings["antispoof_mandatory"] and not (liveness["texture_ok"] or liveness["color_ok"])
                        multiface_block = not liveness["single_face"]

                        if spoof_block or multiface_block:
                            reason = "Possible spoof (photo/screen) detected" if spoof_block else "Zero or multiple faces detected"
                            st.error(f"🚫 ACCESS DENIED — {reason}")
                            add_audit_event(username, "LOGIN_SPOOF_BLOCKED", success=False)
                            fa["count"] = fa.get("count", 0) + 1
                            st.session_state.failed_attempts[username] = fa
                        else:
                            # ── Step B: voice ────────────────────────────
                            if settings["require_voice"] and voice_clip is not None and AUDIO_AVAILABLE and SR_AVAILABLE:
                                with st.status("🎙 Uploading voice sample to server…", expanded=True) as s2:
                                    st.write("📡 Received audio from browser microphone")
                                    st.write("🗣 Running speech-to-text on passphrase…")
                                    st.write("🔎 Comparing MFCC voiceprint against enrolled sample…")
                                    voice_result = run_voice_auth(username, voice_clip.getvalue(),
                                                                   enroll_mode=False, voiceprint_threshold=vp_threshold)
                                    if voice_result.get("error"):
                                        s2.update(label=f"⚠️ Voice check issue: {voice_result['error']}", state="error")
                                    else:
                                        s2.update(label="✅ Voice analysis complete", state="complete")
                            else:
                                voice_result = {"passphrase_ok": not settings["require_voice"],
                                                 "voiceprint_ok": not settings["require_voice"],
                                                 "spoken_text": "[voice check skipped by policy]", "voice_dist": 9999}

                            trust = compute_trust_score(liveness, face_matched, face_diff, pin_ok, voice_result)

                            st.markdown("#### Auth Report")
                            rows = [
                                ("🔢 PIN",                pin_ok),
                                ("🧬 Face pixel match",   face_matched),
                                ("👤 Single face",        liveness["single_face"]),
                                ("↔ Head turn liveness",  liveness["head_move"]),
                                ("🖼 Texture anti-spoof",  liveness["texture_ok"]),
                                ("🎨 Colour anti-spoof",  liveness["color_ok"]),
                                ("🎙 Voice passphrase",   voice_result.get("passphrase_ok", False)),
                                ("🧬 Voiceprint match",   voice_result.get("voiceprint_ok", False)),
                            ]
                            for label, ok in rows:
                                cls = "ok" if ok else "fail"
                                icon = "✓" if ok else "✗"
                                st.markdown(f'<div class="{cls}">{icon} {label}</div>', unsafe_allow_html=True)

                            spoken = voice_result.get("spoken_text", "")
                            if spoken:
                                st.markdown(f'<div class="warn">🗣 Heard: <em>"{spoken}"</em></div>', unsafe_allow_html=True)
                            vdist = voice_result.get("voice_dist", 9999)
                            if vdist < 9000:
                                st.markdown(f'<div class="warn">📊 Voice distance: {vdist} (threshold {vp_threshold})</div>', unsafe_allow_html=True)

                            pct = int(trust * 100)
                            bar_color = "#00e5ff" if trust >= threshold else "#ff4757"
                            st.markdown(f"""
                            <div style="margin:12px 0">
                              <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px">
                                <span>Trust Score</span>
                                <span style="color:{bar_color};font-weight:600">{pct}%</span>
                              </div>
                              <div class="score-bg">
                                <div class="score-fg" style="width:{pct}%;background:{bar_color}"></div>
                              </div>
                              <div style="font-size:0.72rem;color:#64748b">Threshold: {int(threshold*100)}%</div>
                            </div>""", unsafe_allow_html=True)

                            # ── Final decision ───────────────────────────
                            if trust >= threshold:
                                st.success(f"🔓 ACCESS GRANTED — welcome, {username}!")
                                db[username]["login_count"] = db[username].get("login_count", 0) + 1
                                db[username]["last_login"]  = datetime.utcnow().isoformat()
                                save_db(db)
                                st.session_state.authenticated = True
                                st.session_state.session_user  = username
                                st.session_state.session_role  = db[username]["role"]
                                st.session_state.session_start = datetime.utcnow()
                                st.session_state.trust_score   = round(trust, 3)
                                st.session_state.failed_attempts.pop(username, None)
                                add_audit_event(username, "LOGIN_SUCCESS", score=trust, success=True)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("🚫 ACCESS DENIED — trust score too low")
                                fa["count"] = fa.get("count", 0) + 1
                                if fa["count"] >= settings["lockout_attempts"]:
                                    fa["until"] = (datetime.utcnow() + timedelta(minutes=settings["lockout_minutes"])).isoformat()
                                    st.warning(f"⚠️ {settings['lockout_attempts']} failed attempts – locked for {settings['lockout_minutes']} minutes")
                                st.session_state.failed_attempts[username] = fa
                                add_audit_event(username, "LOGIN_FAILED", score=trust, success=False)

# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.authenticated and menu == "Dashboard":
    db   = load_db()
    user = st.session_state.session_user
    role = st.session_state.session_role
    info = db.get(user, {})

    st.title(f"🏠 Dashboard — {user}")
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Role", role.upper())
    with m2: st.metric("Trust Score", f"{st.session_state.trust_score:.2f}")
    with m3: st.metric("Department", info.get("dept", "—"))
    with m4: st.metric("Login Count", info.get("login_count", 1))

    st.divider()
    tab1, tab2, tab3 = st.tabs(["🏠 Overview", "📋 Audit Log", "🧑 Profile"])

    with tab1:
        if role in ["operator", "admin", "superadmin"]:
            st.markdown("#### ✅ Systems Access")
            cols = st.columns(3)
            for i, s in enumerate(["CCTV Monitor", "Badge Scanner", "Alarm Control", "Server Room", "Network Console", "Safety Logs"]):
                with cols[i % 3]: st.markdown(f'<div class="ok">● {s}</div>', unsafe_allow_html=True)
        if role in ["admin", "superadmin"]:
            st.markdown("#### 🛠 Admin Systems")
            ac = st.columns(2)
            for i, s in enumerate(["User Management", "Threshold Config", "Audit Export", "Backup & Restore"]):
                with ac[i % 2]: st.markdown(f'<div class="warn">● {s}</div>', unsafe_allow_html=True)
        if role == "viewer":
            st.info("Read-only access. Contact admin for elevated privileges.")

    with tab2:
        log = load_audit()
        if isinstance(log, list) and log and isinstance(log[0], dict) and "_id" in log[0]:
            for e in log: e.pop("_id", None)
        visible = log[-50:][::-1] if role in ["admin", "superadmin"] \
                  else [e for e in log if e.get("user") == user][-20:][::-1]
        for e in visible:
            clr = "#00ff9d" if e.get("success") else "#ff4757"
            sc = f" | score={e['score']}" if e.get("score") is not None else ""
            st.markdown(f'<div class="audit-row"><span style="color:#64748b">{e["ts"]}</span><span>{e["user"]}</span><span>{e["event"]}{sc}</span><span style="color:{clr}">{"✓" if e.get("success") else "✗"}</span></div>', unsafe_allow_html=True)

    with tab3:
        for k, v in {"Username": user, "Role": role, "Department": info.get("dept", "—"),
                     "Registered": info.get("created_at", "")[:10],
                     "Last Login": info.get("last_login", "—")[:19] if info.get("last_login") else "—",
                     "Login Count": info.get("login_count", 0),
                     "Voiceprint": ("✅ Enrolled" if info.get("has_voiceprint") else "❌ Not enrolled")}.items():
            a, b = st.columns([1, 2])
            with a: st.markdown(f'<span style="color:#64748b;font-size:0.85rem">{k}</span>', unsafe_allow_html=True)
            with b: st.markdown(f'<span style="font-size:0.85rem">{v}</span>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ADMIN PANEL
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.authenticated and menu == "Admin Panel":
    role = st.session_state.session_role
    if role not in ["admin", "superadmin"]:
        st.error("🚫 Insufficient privileges")
        st.stop()

    st.title("🛡 Admin Panel")
    db = load_db()
    t1, t2, t3 = st.tabs(["👥 Users", "📊 Stats", "⚙️ Settings"])

    with t1:
        for uname, udata in db.items():
            with st.expander(f"{'🟢' if udata.get('active', True) else '🔴'} {uname}  [{udata.get('role', '?').upper()}]  — {udata.get('dept', '?')}"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Created:** {udata.get('created_at', '')[:10]}")
                    st.markdown(f"**Logins:** {udata.get('login_count', 0)}")
                    vp = "✅" if udata.get("has_voiceprint") else "❌"
                    st.markdown(f"**Voiceprint:** {vp}")
                with c2:
                    nr = st.selectbox("Change role", ROLES, index=ROLES.index(udata.get("role", "viewer")), key=f"r_{uname}")
                with c3:
                    if role == "superadmin":
                        lbl = "Disable" if udata.get("active", True) else "Enable"
                        if st.button(f"{lbl} account", key=f"t_{uname}"):
                            db[uname]["active"] = not udata.get("active", True)
                            save_db(db)
                            add_audit_event(st.session_state.session_user, f"ADMIN_TOGGLE_{uname}", success=True)
                            st.rerun()
                if st.button("Save role", key=f"s_{uname}"):
                    db[uname]["role"] = nr
                    save_db(db)
                    add_audit_event(st.session_state.session_user, f"ADMIN_ROLE_{uname}_TO_{nr}", success=True)
                    st.success("Role updated")

    with t2:
        log = load_audit()
        if isinstance(log, list) and log and "_id" in log[0]:
            [e.pop("_id", None) for e in log]
        total = len(log)
        succ  = sum(1 for e in log if e.get("success"))
        logins = [e for e in log if e.get("event") == "LOGIN_SUCCESS"]
        avg = round(sum(e["score"] for e in logins if e.get("score")) / max(len(logins), 1), 3)
        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Total Events", total)
        with s2: st.metric("Successes", succ)
        with s3: st.metric("Failures", total - succ)
        with s4: st.metric("Avg Login Score", avg)
        st.markdown("#### ⚠️ Recent Failures")
        for e in [e for e in log if not e.get("success") and "LOGIN" in e.get("event", "")][-8:][::-1]:
            st.markdown(f'<div class="fail">✗ {e["ts"][:19]} | {e["user"]} | {e["event"]}</div>', unsafe_allow_html=True)

    with t3:
        st.caption("These values become the defaults every user sees on the login screen, and the hard security gates enforced during verification.")
        s = load_settings()
        new_settings = dict(s)
        new_settings["trust_threshold"]  = st.slider("Default trust threshold", 0.5, 1.0, float(s["trust_threshold"]), 0.05)
        new_settings["lockout_attempts"] = st.slider("Lockout after N failures", 1, 10, int(s["lockout_attempts"]))
        new_settings["lockout_minutes"]  = st.slider("Lockout duration (min)", 1, 60, int(s["lockout_minutes"]))
        new_settings["voice_threshold"]  = st.slider("Default voice similarity threshold", 10.0, 60.0, float(s["voice_threshold"]), 5.0)
        new_settings["require_pin"]        = st.toggle("Require PIN", value=bool(s["require_pin"]))
        new_settings["require_voice"]      = st.toggle("Require voiceprint", value=bool(s["require_voice"]))
        new_settings["antispoof_mandatory"]= st.toggle("Anti-spoof mandatory (hard block, not just scored)", value=bool(s["antispoof_mandatory"]))
        if st.button("💾 Save Settings"):
            save_settings(new_settings)
            add_audit_event(st.session_state.session_user, "ADMIN_SETTINGS_UPDATED", success=True)
            st.success("Settings saved — new defaults apply immediately for all users.")
            st.rerun()