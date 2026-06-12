import streamlit as st
import cv2
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import re
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from rapidfuzz import process, distance


# CONFIG PAGE

st.set_page_config(
    page_title="ALPR-MA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# CSS

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@300;400;700&display=swap');

:root {
    --black:   #080808;
    --dark:    #111111;
    --card:    #161616;
    --border:  #2a2a2a;
    --blue:    #1c69d4;
    --blue-lt: #4a90d9;
    --white:   #f5f5f5;
    --muted:   #888888;
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif !important;
    background: var(--black) !important;
    color: var(--white) !important;
}

.stApp { background: var(--black) !important; }

header[data-testid="stHeader"] { display: none !important; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── NAVBAR ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    border-bottom: 1px solid var(--border);
    background: var(--black);
}

.navbar-brand {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--white);
}

.navbar-brand span { color: var(--blue); }

.navbar-links {
    display: flex;
    gap: 36px;
    list-style: none;
    margin: 0; padding: 0;
}

.navbar-links a {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    text-decoration: none;
}

.navbar-tag {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── HERO ── */
.hero {
    position: relative;
    min-height: 88vh;
    display: flex;
    align-items: stretch;
    overflow: hidden;
    background: linear-gradient(105deg, #000000 0%, #0a0a14 35%, #0d1628 100%);
}

.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(28,105,212,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(28,105,212,0.04) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
}

.hero-left {
    position: relative;
    z-index: 2;
    flex: 0 0 48%;
    display: flex;
    align-items: flex-start;
    padding: 72px 0 0 64px;
    background: linear-gradient(90deg, #000000 60%, transparent 100%);
}

.hero-right {
    flex: 1;
    position: relative;
    overflow: hidden;
}

.hero-right img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center center;
    display: block;
    mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,0.4) 20%, rgba(0,0,0,0.85) 45%, #000 100%);
    -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,0.4) 20%, rgba(0,0,0,0.85) 45%, #000 100%);
}

.hero-right::after {
    content: '';
    position: absolute;
    bottom: -80px; right: -40px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(28,105,212,0.18) 0%, transparent 65%);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 2;
    max-width: 560px;
}

.hero-eyebrow {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 400;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--blue-lt);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-eyebrow::before {
    content: '';
    display: inline-block;
    width: 32px; height: 1px;
    background: var(--blue);
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3.5rem, 7vw, 6rem);
    font-weight: 300;
    line-height: 1.0;
    color: var(--white);
    margin: 0 0 6px;
}

.hero-title em {
    font-style: italic;
    color: var(--blue-lt);
}

.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: var(--muted);
    line-height: 1.7;
    margin: 24px 0 40px;
}

.hero-stats {
    display: flex;
    gap: 0;
    border-top: 1px solid var(--border);
    padding-top: 32px;
}

.hero-stat {
    flex: 1;
    padding-right: 32px;
    border-right: 1px solid var(--border);
    margin-right: 32px;
}
.hero-stat:last-child { border-right: none; margin-right: 0; }

.hero-stat-val {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--white);
    line-height: 1;
    margin-bottom: 4px;
}
.hero-stat-val span { color: var(--blue); }

.hero-stat-lbl {
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── SECTION ── */
.section {
    padding: 64px 64px 24px;
}

.section-label {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    width: 80px; height: 1px;
    background: var(--border);
}

.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 300;
    color: var(--white);
    margin: 0 0 32px;
}

/* ── RÉSULTAT PLAQUE ── */
.result-block {
    background: var(--dark);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    padding: 40px 48px;
    margin: 40px 0 0;
}

.result-eyebrow {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: var(--blue-lt);
    margin-bottom: 16px;
}

/* FIX BIDI — force LTR pour la plaque */
.result-plate {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 3.8rem;
    font-weight: 700;
    color: var(--white);
    letter-spacing: 0.18em;
    line-height: 1;
    margin-bottom: 20px;
    text-shadow: 0 0 60px rgba(28,105,212,0.3);
    direction: ltr;
    unicode-bidi: bidi-override;
}

.badge-valid {
    display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid #22c55e; color: #4ade80;
    padding: 5px 16px; font-size: 0.68rem;
    font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase;
}

.badge-invalid {
    display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid #f97316; color: #fb923c;
    padding: 5px 16px; font-size: 0.68rem;
    font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase;
}

/* ── MÉTRIQUES ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    margin: 1px 0 32px;
}

.metric-item {
    background: var(--card);
    padding: 28px 32px;
}

.metric-item .mv {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--white);
    line-height: 1;
    margin-bottom: 6px;
}

.metric-item .mv.blue  { color: var(--blue-lt); }
.metric-item .mv.green { color: #4ade80; }
.metric-item .mv.orange{ color: #fb923c; }

.metric-item .ml {
    font-size: 0.67rem; font-weight: 600;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--muted);
}

/* ── IMAGES ── */
[data-testid="stImage"] img {
    border-radius: 0 !important;
    border: 1px solid var(--border) !important;
}

.img-label {
    font-size: 0.65rem; font-weight: 600;
    letter-spacing: 0.2em; text-transform: uppercase;
    color: var(--muted);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--dark) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--white) !important; }

.sb-section {
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--muted) !important;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px; margin-bottom: 16px;
}

.sb-kpi {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 10px 0; border-bottom: 1px solid #1e1e1e;
}
.sb-kpi .sk-lbl { font-size: 0.75rem; color: var(--muted) !important; }
.sb-kpi .sk-val {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.1rem; font-weight: 700; color: var(--blue-lt) !important;
}

/* ── UPLOAD ── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
}

/* ── BOUTONS ── */
.stButton > button {
    background: var(--blue) !important;
    color: #fff !important; border: none !important;
    border-radius: 2px !important;
    font-size: 0.78rem !important; font-weight: 600 !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    padding: 14px 32px !important;
}

p, label { color: var(--muted) !important; }
h1,h2,h3 { color: var(--white) !important; }
</style>
""", unsafe_allow_html=True)


# DICTIONNAIRE

DICT_PATH = Path(__file__).parent / "data_dictionary.json"

@st.cache_resource
def load_dict():
    with open(DICT_PATH, encoding='utf-8') as f:
        return json.load(f)

dd          = load_dict()
arabic_map  = dd['arabic_mapping']
NUM_CLASSES = dd['num_classes_without_blank']
BLANK_IDX   = dd['blank_token']['index']
IDX_TO_ARABIC = {int(k): arabic_map.get(v, v) for k, v in dd['idx2char'].items()}


# CRNN

class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES + 1, hidden_size=256, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2,1),(2,1)),
            nn.Conv2d(512, 512, (2,1), padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.bilstm = nn.LSTM(512, hidden_size, num_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x).squeeze(2).permute(2, 0, 1)
        out, _ = self.bilstm(feat)
        return F.log_softmax(self.fc(out), dim=2)


# POST-TRAITEMENT — CORRIGÉ

class MoroccanPostProcessor:
    def __init__(self, arabic_dict):
        self.allowed_letters = [v for k, v in arabic_dict.items() if v not in "0123456789"]
        if 'ش' not in self.allowed_letters:
            self.allowed_letters.append('ش')
        self.latin_to_num = {'O':'0','D':'0','I':'1','L':'1','Z':'2','S':'5','B':'8'}

    def fix_digits(self, text):
        return "".join(self.latin_to_num.get(c, c) for c in text) if text else ""

    def process(self, raw_text):
        if not raw_text or raw_text == "?":
            return raw_text, False
        text = raw_text.strip().upper().replace(" ", "")
        match = re.match(r'^([0-9]+)([^0-9]+)([0-9]*)$', text)
        if match:
            p1, lp, p3 = match.group(1), match.group(2), match.group(3)
            if lp == "WW":
                letter = "WW"
            else:
                res = process.extractOne(lp, self.allowed_letters, scorer=distance.Levenshtein.distance)
                letter = res[0] if res else lp
            p3c = self.fix_digits(p3)
            if p3c:
                return f"{p1} | {letter} | {p3c}", True
            else:
                return f"{p1} | {letter}", True
        return raw_text, False


# MODÈLES

val_transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

@st.cache_resource
def load_models():
    base = Path(__file__).parent
    crnn_path = base / "crnn_weights_v2.pth"
    if crnn_path.is_dir():
        raise ValueError("crnn_weights_v2.pth est un dossier — téléchargez le vrai fichier depuis Kaggle Output")
    detector = YOLO(str(base / "yolo26_best.pt"))
    ocr = CRNN(num_classes=NUM_CLASSES + 1)
    state = torch.load(str(crnn_path), map_location='cpu', weights_only=False, mmap=True)
    ocr.load_state_dict(state.get('model_state_dict', state))
    ocr.eval()
    return detector, ocr, MoroccanPostProcessor(IDX_TO_ARABIC)


# PIPELINE

def greedy_decode(log_probs):
    indices = log_probs.argmax(dim=2).permute(1, 0)
    results = []
    for seq in indices:
        tokens, prev = [], BLANK_IDX
        for idx in seq.tolist():
            if idx != BLANK_IDX and idx != prev and idx < NUM_CLASSES:
                tokens.append(IDX_TO_ARABIC.get(idx, '?'))
            prev = idx
        results.append(''.join(tokens))
    return results

def run_pipeline(image_pil, detector, ocr, post, conf_thr=0.3):
    t0 = time.time()
    img_np  = np.array(image_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    res = detector.predict(source=img_bgr, conf=conf_thr, verbose=False)
    if not res or len(res[0].boxes) == 0:
        return None, None, None, None, None, None
    best = res[0].boxes.conf.argmax()
    box  = res[0].boxes.xyxy[best].cpu().numpy()
    conf = float(res[0].boxes.conf[best])
    x1,y1,x2,y2 = map(int, box)
    crop   = cv2.resize(img_bgr[y1:y2,x1:x2], (128,32))
    gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    tensor = val_transform(Image.fromarray(gray)).unsqueeze(0)
    with torch.no_grad():
        lp = ocr(tensor)
    raw        = greedy_decode(lp)[0]
    text, valid = post.process(raw)
    annotated  = img_np.copy()
    cv2.rectangle(annotated, (x1,y1), (x2,y2), (28,105,212) if valid else (249,115,22), 3)
    return text, valid, conf, gray, annotated, round((time.time()-t0)*1000, 1)


# NAVBAR

st.markdown("""
<div class="navbar">
    <div class="navbar-brand">ALPR<span>·</span>MA</div>
    <ul class="navbar-links">
        <li><a href="#">Détection</a></li>
        <li><a href="#">Architecture</a></li>
        <li><a href="#">Métriques</a></li>
        <li><a href="#">Équipe</a></li>
    </ul>
    <div class="navbar-tag">ENSA Khouribga · IID1 · 2025/2026</div>
</div>
""", unsafe_allow_html=True)


# HERO — chemin corrigé

import base64

HERO_IMAGE_PATH = Path("VOITURE.png")

def get_hero_b64(path):
    if path.exists():
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext  = path.suffix.lower().lstrip(".")
        mime = {"jpg":"jpeg","jpeg":"jpeg","png":"png","webp":"webp"}.get(ext, "jpeg")
        return f"data:image/{mime};base64,{data}"
    return None

hero_src = get_hero_b64(HERO_IMAGE_PATH)

if hero_src:
    hero_right = ('<div class="hero-right">'
                  '<img src="' + hero_src + '" alt="Vehicule marocain"/>'
                  '</div>')
else:
    hero_right = ('<div class="hero-right" style="display:flex;align-items:center;'
                  'justify-content:center;background:linear-gradient(135deg,#0a0a14,#0d1628);'
                  'border-left:1px solid #1e1e1e;">'
                  '<div style="text-align:center;opacity:0.25;">'
                  '<div style="font-size:5rem;">&#128664;</div>'
                  '<div style="font-family:sans-serif;font-size:0.68rem;'
                  'letter-spacing:0.3em;text-transform:uppercase;color:#888;margin-top:14px;">'
                  'Placez hero_car.jpg dans le dossier'
                  '</div></div></div>')

hero_html = (
    '<div class="hero">'
    '<div class="hero-left">'
    '<div class="hero-content">'
    '<div class="hero-eyebrow">Intelligence Artificielle &middot; Vision par Ordinateur</div>'
    '<div class="hero-title">Reconnaissance<br><em>automatique</em><br>des plaques</div>'
    '<p class="hero-sub">Pipeline YOLO + CRNN entraîné sur 6 714 images marocaines. '
    'Détection en temps réel, lecture des caractères arabes et chiffres.</p>'
    '<div class="hero-stats">'
    '<div class="hero-stat">'
    '<div class="hero-stat-val"><span>96</span>.4%</div>'
    '<div class="hero-stat-lbl">mAP@0.5 &middot; YOLO</div>'
    '</div>'
    '<div class="hero-stat">'
    '<div class="hero-stat-val"><span>98</span>.59%</div>'
    '<div class="hero-stat-lbl">Accuracy &middot; CRNN</div>'
    '</div>'
    '<div class="hero-stat">'
    '<div class="hero-stat-val">&le; <span>30</span>ms</div>'
    '<div class="hero-stat-lbl">Temps de traitement</div>'
    '</div>'
    '</div>'
    '</div>'
    '</div>'
    + hero_right +
    '</div>'
)

st.markdown(hero_html, unsafe_allow_html=True)


# SIDEBAR

with st.sidebar:
    st.markdown('<div class="sb-section">Paramètres</div>', unsafe_allow_html=True)
    conf_threshold = st.slider("Seuil de confiance YOLO", 0.1, 0.9, 0.3, 0.05)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Performances</div>', unsafe_allow_html=True)
    for lbl, val in [("mAP@0.5 (YOLO)","96.4%"),("Accuracy (CRNN)","98.59%"),
                     ("TER (CRNN)","0.0021"),("Plaques correctes","419 / 425")]:
        st.markdown(f'<div class="sb-kpi"><span class="sk-lbl">{lbl}</span><span class="sk-val">{val}</span></div>',
                    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sb-section">Équipe</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.78rem;line-height:2.2;color:#666;">MOUJI Rahma<br>M\'BARKI Mariam<br>MASBAH Hajar<br>LARHZALI Aya</div>',
                unsafe_allow_html=True)


# SECTION ANALYSE

st.markdown("""
<div class="section">
    <div class="section-label">Analyse · Pipeline ALPR</div>
    <div class="section-title">Déposez une image de véhicule</div>
</div>
""", unsafe_allow_html=True)

_, col_up, _ = st.columns([1, 3, 1])
with col_up:
    with st.spinner("Chargement des modèles IA..."):
        try:
            detector, ocr_model, post_processor = load_models()
        except Exception as e:
            st.error(f"❌ {e}")
            st.stop()
    uploaded = st.file_uploader(
        "Formats acceptés : JPG · JPEG · PNG",
        type=["jpg","jpeg","png"],
        label_visibility="visible"
    )

if uploaded:
    image_pil = Image.open(uploaded)
    with st.spinner("Analyse en cours..."):
        try:
            text, valid, conf, gray, annotated, ms = run_pipeline(
                image_pil, detector, ocr_model, post_processor, conf_threshold
            )
            display_text = text

            st.markdown('<div style="padding:0 64px;">', unsafe_allow_html=True)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown('<div class="img-label">Image originale</div>', unsafe_allow_html=True)
                st.image(image_pil, use_column_width=True)
            with c2:
                st.markdown('<div class="img-label">Détection YOLO</div>', unsafe_allow_html=True)
                if annotated is not None:
                    st.image(annotated, use_column_width=True)
                else:
                    st.warning("Aucune plaque détectée.")
            st.markdown('</div>', unsafe_allow_html=True)

            if text:
                badge = ('<span class="badge-valid">✓ Format légal valide</span>'
                         if valid else '<span class="badge-invalid">⚠ Format non validé</span>')
                st.markdown(f"""
                <div style="padding:0 64px;">
                <div class="result-block">
                    <div class="result-eyebrow">Plaque reconnue</div>
                    <div class="result-plate">{text}</div>
                    {badge}
                </div>""", unsafe_allow_html=True)

                mv_color = "green" if valid else "orange"
                mv_text  = "Valide" if valid else "Non validé"
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-item">
                        <div class="mv blue">{conf:.1%}</div>
                        <div class="ml">Confiance YOLO</div>
                    </div>
                    <div class="metric-item">
                        <div class="mv">{ms} ms</div>
                        <div class="ml">Temps de traitement</div>
                    </div>
                    <div class="metric-item">
                        <div class="mv {mv_color}">{mv_text}</div>
                        <div class="ml">Statut format</div>
                    </div>
                </div>
                </div>""", unsafe_allow_html=True)

                with st.expander("Étapes intermédiaires — plaque recadrée"):
                    if gray is not None:
                        st.image(gray, caption="Plaque 128×32 · niveaux de gris", width=320)

                with st.expander("Résultat JSON complet"):
                    st.code(json.dumps({
                        "plate_text": display_text.replace(" | ", " \u200e|\u200e "),
                        "valid_format": valid,
                        "confidence": round(conf, 3),
                        "processing_time_ms": ms,
                        "status": "success"
                    }, ensure_ascii=False, indent=2), language="json")
            else:
                st.error("❌ Aucune plaque détectée dans cette image.")

        except Exception as e:
            st.error(f"❌ Erreur : {e}")
