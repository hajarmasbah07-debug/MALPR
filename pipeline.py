# ═══════════════════════════════════════════════════════════════
# CELLULE 1 — Installation (Mise à jour : ajout rapidfuzz)
# ═══════════════════════════════════════════════════════════════
import subprocess
# Ajout de rapidfuzz pour le module de post-traitement
# Ajout de rapidfuzz ET onnxruntime pour l'inférence
subprocess.run(["pip", "install", "ultralytics", "kaggle", "rapidfuzz", "onnxruntime", "-q"])

import os, cv2, json, time, glob, torch, warnings, re
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import onnxruntime as ort
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from rapidfuzz import process, distance # Bibliothèques pour Levenshtein

print(f"✅ torch       : {torch.__version__}")
print(f"✅ GPU dispo   : {torch.cuda.is_available()}")


# ═══════════════════════════════════════════════════════════════
# CELLULE 2 — Télécharger le dataset de test
# ═══════════════════════════════════════════════════════════════

TEST_DIR = "/kaggle/working/moroccan_plates_test"
os.makedirs(TEST_DIR, exist_ok=True)

subprocess.run([
    "kaggle", "datasets", "download",
    "-d", "elmehditaf96/moroccan-vehicle-registration-plates",
    "-p", TEST_DIR, "--unzip"
], check=True)

# Correction des chemins glob
images_test = (
    glob.glob(f"{TEST_DIR}/**/*.jpg",  recursive=True) +
    glob.glob(f"{TEST_DIR}/**/*.png",  recursive=True) +
    glob.glob(f"{TEST_DIR}/**/*.jpeg", recursive=True)
)

print(f"✅ {len(images_test)} images de test trouvées")

# ═══════════════════════════════════════════════════════════════
# CELLULE 3 — Détection automatique des chemins
# ═══════════════════════════════════════════════════════════════
def find_file(pattern, search_root="/kaggle/input"):
    matches = [m for m in glob.glob(f"{search_root}/**/{pattern}", recursive=True)
               if os.path.isfile(m)]
    return matches[0] if matches else None

YOLO_PATH = find_file("yolo26_best.pt")
CRNN_PATH = find_file("crnn_weights_v2.pth")   
DICT_PATH = find_file("data_dictionary (1)*.json")  

with open(DICT_PATH, encoding='utf-8') as f:
    dd = json.load(f)

char2idx   = dd['char2idx']          
idx2char   = {int(k): v for k, v in dd['idx2char'].items()}  
arabic_map = dd['arabic_mapping']    
NUM_CLASSES = dd['num_classes_without_blank']   
BLANK_IDX   = dd['blank_token']['index']        
T_STEPS     = dd['model']['sequence_length']    

IDX_TO_ARABIC = {
    int(k): arabic_map.get(v, v)
    for k, v in dd['idx2char'].items()
}

print(f"\n✅ Alphabet chargé ({NUM_CLASSES} classes)")


# ═══════════════════════════════════════════════════════════════
# CELLULE 4 — val_transform
# ═══════════════════════════════════════════════════════════════
val_transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# ═══════════════════════════════════════════════════════════════
# CELLULE 5 — Architecture CRNN
# ═══════════════════════════════════════════════════════════════
class CRNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES + 1, hidden_size=256, num_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, (2, 1), padding=0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.bilstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        feat = self.cnn(x).squeeze(2).permute(2, 0, 1)
        out, _ = self.bilstm(feat)
        return F.log_softmax(self.fc(out), dim=2)


# ═══════════════════════════════════════════════════════════════
# CELLULE 6 — Module 2 : rectifier
# ═══════════════════════════════════════════════════════════════
def get_rectified_plate(image_bgr, box_coords):
    try:
        x1, y1, x2, y2 = map(int, box_coords)
        plate_crop = image_bgr[y1:y2, x1:x2]
        if plate_crop is None or plate_crop.size == 0: return None
        rectified  = cv2.resize(plate_crop, (128, 32))
        return cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    except: return None

def prepare_tensor(gray_plate):
    pil_img = Image.fromarray(gray_plate)
    return val_transform(pil_img).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════
# CELLULE 7 — Décodage CTC
# ═══════════════════════════════════════════════════════════════
def greedy_decode_arabic(log_probs):
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



# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# CELLULE 7.5 — Module 4 : Post-traitement (Standard Marocain Final)
# ═══════════════════════════════════════════════════════════════
import re
from rapidfuzz import process, distance

class MoroccanPostProcessor:
    def __init__(self, arabic_dict):
        # Dictionnaire des lettres arabes autorisées (Alif, Ba, etc.)
        self.allowed_letters = [v for k, v in arabic_dict.items() if v not in "0123456789"]
        if 'ش' not in self.allowed_letters: self.allowed_letters.append('ش')
        
        # Correction des erreurs de lecture classiques
        self.latin_to_num = {'O': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}

    def fix_digits(self, text):
        """Force la conversion des lettres ambigues en chiffres"""
        if not text: return ""
        res = ""
        for c in text:
            if c in self.latin_to_num: res += self.latin_to_num[c]
            else: res += c
        return res

    def process(self, raw_text):
        if not raw_text or raw_text == "?": return raw_text, False
        
        # Nettoyage et mise en majuscule
        text = raw_text.strip().upper().replace(" ", "")
        
        # REGEX ANTI-FUSION :
        # Group 1 : ([0-9]+) -> QUE des chiffres (Séquence)
        # Group 2 : ([^0-9]+) -> QUE des lettres (Alif, Ba, WW, etc.)
        # Group 3 : ([0-9]*) -> QUE des chiffres (Province - Optionnel)
        pattern = r'^([0-9]+)([^0-9]+)([0-9]*)$'
        match = re.match(pattern, text)
        
        if match:
            # Séparation propre des blocs
            part1 = match.group(1) 
            letter_part = match.group(2)
            part3 = match.group(3)
            
            # Traitement de la lettre (WW reste WW, le reste est recalé sur l'arabe)
            if letter_part == "WW":
                letter = "WW"
            else:
                # On utilise Levenshtein pour trouver la lettre arabe la plus proche
                res_lev = process.extractOne(letter_part, self.allowed_letters, scorer=distance.Levenshtein.distance)
                letter = res_lev[0] if res_lev else letter_part

            # On corrige les chiffres de la province (au cas où il resterait un 'O')
            p3_clean = self.fix_digits(part3)

            # --- SORTIE SANS FUSION ---
            if p3_clean:
                # Cas 3 parties : "12345 | أ | 06"
                return f"{part1} | {letter} | {p3_clean}", True
            else:
                # Cas 2 parties : "596395 | WW" ou "156160 | ش"
                return f"{part1} | {letter}", True
            
        # Si le texte ne ressemble pas du tout à une plaque marocaine
        return raw_text, False

print("✅ Module 4 finalisé : Protection contre la fusion des blocs activée.")

# ═══════════════════════════════════════════════════════════════
# CELLULE 8 — Pipeline complet
# ═══════════════════════════════════════════════════════════════
class LicensePlatePipeline:
    def __init__(self, yolo_path, crnn_path, conf_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.detector = YOLO(yolo_path)
        self.ocr = CRNN(num_classes=NUM_CLASSES + 1)
        state = torch.load(crnn_path, map_location='cpu')
        self.ocr.load_state_dict(state.get('model_state_dict', state))
        self.ocr.eval()
        self.post_processor = MoroccanPostProcessor(IDX_TO_ARABIC)

    def run(self, image_path):
        t0 = time.time()
        image = cv2.imread(str(image_path))
        if image is None: return {"status": "error"}
        
        results = self.detector.predict(source=image, conf=self.conf_threshold, verbose=False)
        if not results or len(results[0].boxes) == 0: return {"status": "no_plate"}

        best_idx = results[0].boxes.conf.argmax()
        box = results[0].boxes.xyxy[best_idx].cpu().numpy()
        conf = float(results[0].boxes.conf[best_idx])

        gray_crop = get_rectified_plate(image, box)
        if gray_crop is None: return {"status": "error"}
        
        with torch.no_grad():
            log_probs = self.ocr(prepare_tensor(gray_crop))
        raw_text = greedy_decode_arabic(log_probs)[0]

        final_text, valid = self.post_processor.process(raw_text)

        return {
            "plate_text": final_text, "is_valid": valid, "bbox": box,
            "confidence": round(conf, 3), "plate_crop": gray_crop,
            "processing_ms": round((time.time() - t0) * 1000, 2), "status": "success"
        }

    def draw_result(self, image, result):
        out = image.copy()
        if result["status"] != "success": return out
        x1, y1, x2, y2 = map(int, result["bbox"])
        color = (0, 255, 0) if result["is_valid"] else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        cv2.putText(out, result['plate_text'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return out


# ═══════════════════════════════════════════════════════════════
# CELLULE 9 — Instanciation et Test final
# ═══════════════════════════════════════════════════════════════
pipeline = LicensePlatePipeline(YOLO_PATH, CRNN_PATH)

# Test sur les 5 premières images
for img_p in images_test[:5]:
    res = pipeline.run(img_p)
    print(f"🔍 Plaque lue : {res['plate_text']} | Valide : {res.get('is_valid')}")


