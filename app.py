# -*- coding: utf-8 -*-
"""
EyeShield AI: ระบบ AI ตรวจคัดกรองความเสี่ยงโรคดวงตาเบื้องต้นจากภาพถ่าย
-----------------------------------------------------------------------
ไฟล์เดียวจบ (All-in-One):  streamlit run app.py

โครงสร้างโมเดล : MobileNetV2 (torchvision) + classifier head 3 คลาส
ลำดับคลาส      : [normal, dry_eye, cataract]   <-- ต้องตรงกับตอนเทรน
น้ำหนักโมเดล   : eyeshield_model.pth (วางไว้โฟลเดอร์เดียวกับ app.py)

วิธีเซฟน้ำหนักจากโค้ดเทรนของคุณให้เข้ากับไฟล์นี้:
    model = EyeShieldMobileNetV2(num_classes=3)   # หรือ torchvision mobilenet_v2 ที่แทน classifier[1] เป็น Linear(1280, 3)
    ... train ...
    torch.save(model.state_dict(), "eyeshield_model.pth")

หากไม่พบไฟล์น้ำหนัก ระบบจะสลับเป็นโหมดสำรอง (Heuristic Feature Analysis) อัตโนมัติ
โดยจะแสดงสถานะให้ผู้ใช้ทราบชัดเจนว่า "ไม่ใช่ผลจากโมเดล Deep Learning ที่ผ่านการฝึก"
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus
from xml.sax.saxutils import escape

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Optional heavy dependencies (guarded so the app never crashes on import)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    TORCH_OK = True
    TORCH_ERR = ""
except Exception as _exc:  # pragma: no cover
    TORCH_OK = False
    TORCH_ERR = str(_exc)

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    REPORTLAB_OK = True
except Exception:  # pragma: no cover
    REPORTLAB_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_PATH = APP_DIR / "eyeshield_model.pth"
CLASS_KEYS = ["normal", "dry_eye", "cataract"]  # ต้องเรียงตรงกับ label ตอนเทรน
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RISK_COLORS = {"green": "#1e9e5a", "yellow": "#e0a100", "red": "#d64545"}

# ---------------------------------------------------------------------------
# Localization (TH / EN)
# ---------------------------------------------------------------------------
TEXT = {
    "th": {
        "subtitle": "ระบบ AI ตรวจคัดกรองความเสี่ยงโรคดวงตาเบื้องต้นจากภาพถ่าย",
        "lang_label": "🌐 ภาษา / Language",
        "disc_title": "⚕️ ข้อความปฏิเสธความรับผิดชอบทางการแพทย์",
        "disc_body": (
            "EyeShield AI เป็นเครื่องมือ **คัดกรองเบื้องต้น** เพื่อให้ข้อมูลเท่านั้น "
            "**ไม่ใช่การวินิจฉัยโรค** และไม่สามารถทดแทนการตรวจโดยจักษุแพทย์ได้\n\n"
            "หากมีอาการผิดปกติ เช่น ตามัวลงเฉียบพลัน ปวดตารุนแรง เห็นแสงวาบ "
            "หรือเห็นเงาดำบังตา ให้พบแพทย์ทันที"
        ),
        "sys_status": "สถานะระบบ AI",
        "mode_model_badge": "🧠 Deep Learning (MobileNetV2) — โหลดน้ำหนักโมเดลสำเร็จ",
        "mode_heur_badge": "🛟 โหมดสำรอง: Heuristic Feature Analysis",
        "fallback_notice": (
            "ไม่พบหรือโหลดไฟล์ `eyeshield_model.pth` ไม่สำเร็จ ระบบจึงใช้ Heuristic Feature Analysis "
            "(วิเคราะห์สี/ความสว่างของภาพเบื้องต้น) ซึ่ง **ไม่ใช่ผลจากโมเดล Deep Learning ที่ผ่านการฝึก** "
            "ความแม่นยำจำกัด ใช้เพื่อสาธิตระบบเท่านั้น"
        ),
        "input_mode": "เลือกวิธีนำเข้าภาพ",
        "mode_upload": "📁 อัปโหลดไฟล์ (JPG/PNG)",
        "mode_camera": "📷 ถ่ายภาพสดผ่านกล้อง",
        "upload_label": "อัปโหลดภาพถ่ายดวงตา (JPG / PNG)",
        "camera_label": "ถ่ายภาพดวงตาด้วยกล้อง",
        "tips": (
            "**เคล็ดลับการถ่ายภาพ:** ถ่ายระยะใกล้ในที่สว่างพอ ไม่ใช้แฟลชแรง โฟกัสให้คม "
            "และให้เห็นตาขาว ม่านตา และรูม่านตาชัดเจน"
        ),
        "no_image": "กรุณาอัปโหลดภาพหรือถ่ายภาพดวงตาเพื่อเริ่มการวิเคราะห์",
        "bad_image": "ไม่สามารถเปิดไฟล์ภาพนี้ได้ กรุณาลองไฟล์อื่น",
        "img_caption": "ภาพที่นำเข้า",
        "analyzing": "AI กำลังวิเคราะห์ภาพ...",
        "results_header": "📊 ผลการคัดกรอง",
        "risk_label": "ระดับความเสี่ยง",
        "finding_label": "ผลที่พบมากที่สุด",
        "conf_label": "AI Confidence Score",
        "conf_label_heur": "Heuristic Score",
        "risk_green": "🟢 ปกติ",
        "risk_yellow": "🟡 ความเสี่ยงต่ำ-ปานกลาง",
        "risk_red": "🔴 ความเสี่ยงสูง",
        "risk_green_plain": "ปกติ",
        "risk_yellow_plain": "ความเสี่ยงต่ำ-ปานกลาง",
        "risk_red_plain": "ความเสี่ยงสูง",
        "classes": {
            "normal": "ปกติ (Normal)",
            "dry_eye": "ตาแห้ง/ระคายเคือง (Dry Eye/Irritation)",
            "cataract": "สงสัยต้อกระจก (Suspected Cataract)",
        },
        "prob_header": "การจัดอันดับความน่าจะเป็นของแต่ละคลาส",
        "advice_header": "💡 คำแนะนำเบื้องต้น",
        "advice": {
            "green": "ไม่พบสัญญาณผิดปกติเด่นชัดจากภาพนี้ ควรดูแลสุขภาพตาตามปกติ พักสายตาจากหน้าจอ "
                     "และตรวจตาประจำปี หากมีอาการผิดปกติให้พบแพทย์",
            "yellow": "พบลักษณะที่อาจเกี่ยวข้องกับความผิดปกติเล็กน้อยถึงปานกลาง แนะนำให้พักสายตา "
                      "หลีกเลี่ยงการขยี้ตา ใช้น้ำตาเทียมตามคำแนะนำเภสัชกร และพิจารณาพบจักษุแพทย์หากอาการไม่ดีขึ้น",
            "red": "พบลักษณะที่ควรได้รับการตรวจเพิ่มเติม แนะนำให้นัดพบจักษุแพทย์โดยเร็วเพื่อตรวจยืนยัน "
                   "โดยเฉพาะหากการมองเห็นแย่ลง",
        },
        "quality_header": "⚠️ ข้อสังเกตด้านคุณภาพภาพ",
        "q_blurry": "ภาพอาจไม่คมชัด ผลอาจคลาดเคลื่อน ลองถ่ายใหม่ให้โฟกัสคมขึ้น",
        "q_dark": "ภาพมืดเกินไป ลองถ่ายในที่ที่มีแสงมากขึ้น",
        "q_bright": "ภาพสว่างจ้า/แสงสะท้อนมากเกินไป ลองลดแสงแฟลชหรือเปลี่ยนมุม",
        "q_small": "ความละเอียดของภาพค่อนข้างต่ำ (ต่ำกว่า 224 พิกเซล) ผลอาจไม่แม่นยำ",
        "pdf_btn": "📄 ดาวน์โหลดรายงาน PDF",
        "pdf_unavailable": "ไม่พบไลบรารี reportlab จึงสร้าง PDF ไม่ได้ (pip install reportlab)",
        "pdf_font_missing": (
            "ไม่พบฟอนต์ภาษาไทยในเครื่อง จึงสร้างรายงาน PDF เป็นภาษาอังกฤษแทน "
            "(วางไฟล์ THSarabunNew.ttf ไว้ในโฟลเดอร์ fonts/ หรือเพิ่ม fonts-thai-tlwg ใน packages.txt)"
        ),
        "clinic_header": "🏥 ค้นหาคลินิก/โรงพยาบาลตาใกล้เคียง",
        "area_input": "พื้นที่ (ไม่บังคับ เช่น อำเภอ/จังหวัด)",
        "clinic_btn": "🗺️ เปิด Google Maps ค้นหาคลินิกตา",
        "clinic_query": "จักษุแพทย์ คลินิกตา โรงพยาบาลตา",
        "clinic_near_me": "ใกล้ฉัน",
        # PDF-only
        "pdf_title": "รายงานผลการคัดกรองดวงตาเบื้องต้น",
        "pdf_report_id": "รหัสรายงาน",
        "pdf_date": "วันที่/เวลา",
        "pdf_mode": "วิธีประมวลผล",
        "pdf_mode_model": "Deep Learning (MobileNetV2)",
        "pdf_mode_heur": "Heuristic Feature Analysis (โหมดสำรอง ไม่ใช่โมเดลที่ผ่านการฝึก)",
        "pdf_summary": "สรุปผล",
        "pdf_disclaimer": (
            "รายงานนี้สร้างโดยระบบคัดกรองเบื้องต้นด้วย AI เพื่อให้ข้อมูลเท่านั้น ไม่ใช่การวินิจฉัยโรค "
            "และไม่สามารถใช้แทนความเห็นของแพทย์ได้ กรุณาปรึกษาจักษุแพทย์เพื่อการตรวจและวินิจฉัยที่ถูกต้อง"
        ),
        "pdf_footer": "EyeShield AI - รายงานคัดกรองเบื้องต้น (ไม่ใช่การวินิจฉัยทางการแพทย์)",
        "pdf_class": "คลาส",
        "pdf_prob": "ความน่าจะเป็น",
    },
    "en": {
        "subtitle": "AI-powered preliminary eye disease risk screening from photos",
        "lang_label": "🌐 ภาษา / Language",
        "disc_title": "⚕️ Medical Disclaimer",
        "disc_body": (
            "EyeShield AI is a **preliminary screening** tool for informational purposes only. "
            "It is **not a medical diagnosis** and cannot replace an examination by an ophthalmologist.\n\n"
            "If you have sudden vision loss, severe eye pain, flashes of light, or a dark shadow "
            "in your vision, seek medical care immediately."
        ),
        "sys_status": "AI Engine Status",
        "mode_model_badge": "🧠 Deep Learning (MobileNetV2) - weights loaded",
        "mode_heur_badge": "🛟 Fallback mode: Heuristic Feature Analysis",
        "fallback_notice": (
            "`eyeshield_model.pth` was not found or failed to load, so the app is using Heuristic Feature "
            "Analysis (basic colour/brightness cues). This is **not the output of a trained deep learning "
            "model** and has limited accuracy - for demonstration only."
        ),
        "input_mode": "Choose image input",
        "mode_upload": "📁 Upload file (JPG/PNG)",
        "mode_camera": "📷 Live camera capture",
        "upload_label": "Upload an eye photo (JPG / PNG)",
        "camera_label": "Take an eye photo with your camera",
        "tips": (
            "**Photo tips:** shoot close-up in good light, avoid harsh flash, keep it in focus, "
            "and make sure the sclera, iris and pupil are clearly visible."
        ),
        "no_image": "Please upload or capture an eye photo to start the analysis.",
        "bad_image": "This image file could not be opened. Please try another one.",
        "img_caption": "Input image",
        "analyzing": "AI is analysing the image...",
        "results_header": "📊 Screening Result",
        "risk_label": "Risk Level",
        "finding_label": "Top Finding",
        "conf_label": "AI Confidence Score",
        "conf_label_heur": "Heuristic Score",
        "risk_green": "🟢 Normal",
        "risk_yellow": "🟡 Low-Moderate Risk",
        "risk_red": "🔴 High Risk",
        "risk_green_plain": "Normal",
        "risk_yellow_plain": "Low-Moderate Risk",
        "risk_red_plain": "High Risk",
        "classes": {
            "normal": "Normal",
            "dry_eye": "Dry Eye / Irritation",
            "cataract": "Suspected Cataract",
        },
        "prob_header": "Class probability ranking",
        "advice_header": "💡 General advice",
        "advice": {
            "green": "No prominent abnormal signs were found in this image. Keep up normal eye care, rest your "
                     "eyes from screens, and have regular check-ups. See a doctor if symptoms appear.",
            "yellow": "Features possibly related to a mild-to-moderate problem were found. Rest your eyes, avoid "
                      "rubbing them, consider artificial tears as advised by a pharmacist, and see an "
                      "ophthalmologist if symptoms persist.",
            "red": "Features that warrant further examination were found. We recommend seeing an "
                   "ophthalmologist soon for confirmation, especially if your vision is getting worse.",
        },
        "quality_header": "⚠️ Image quality notes",
        "q_blurry": "The image may be blurry, which can affect accuracy. Try again with sharper focus.",
        "q_dark": "The image is too dark. Try again with more light.",
        "q_bright": "The image is overexposed or has strong glare. Reduce flash or change the angle.",
        "q_small": "The image resolution is quite low (under 224 px), which may reduce accuracy.",
        "pdf_btn": "📄 Download PDF report",
        "pdf_unavailable": "reportlab is not installed, so a PDF cannot be created (pip install reportlab).",
        "pdf_font_missing": "",
        "clinic_header": "🏥 Find nearby eye clinics / hospitals",
        "area_input": "Area (optional, e.g. district / city)",
        "clinic_btn": "🗺️ Open Google Maps for eye clinics",
        "clinic_query": "eye clinic ophthalmologist eye hospital",
        "clinic_near_me": "near me",
        # PDF-only
        "pdf_title": "Preliminary Eye Screening Report",
        "pdf_report_id": "Report ID",
        "pdf_date": "Date / Time",
        "pdf_mode": "Processing method",
        "pdf_mode_model": "Deep Learning (MobileNetV2)",
        "pdf_mode_heur": "Heuristic Feature Analysis (fallback, not a trained model)",
        "pdf_summary": "Summary",
        "pdf_disclaimer": (
            "This report was generated by an AI-based preliminary screening tool for informational purposes "
            "only. It is not a medical diagnosis and cannot replace the opinion of a physician. Please consult "
            "an ophthalmologist for a proper examination and diagnosis."
        ),
        "pdf_footer": "EyeShield AI - Preliminary screening report (not a medical diagnosis)",
        "pdf_class": "Class",
        "pdf_prob": "Probability",
    },
}


# ---------------------------------------------------------------------------
# AI Model Architecture (MobileNetV2, 3 classes)
# ---------------------------------------------------------------------------
if TORCH_OK:

    class EyeShieldMobileNetV2(nn.Module):
        """MobileNetV2 backbone + custom 3-class head.

        ชื่อ key ของ state_dict (features.*, classifier.1.*) ตรงกับ torchvision mobilenet_v2
        ที่ถูกแทน classifier[1] ด้วย Linear(1280, 3) จึงโหลดน้ำหนักที่เทรนแบบมาตรฐานได้ทันที
        """

        def __init__(self, num_classes: int = 3, dropout: float = 0.2):
            super().__init__()
            backbone = models.mobilenet_v2(weights=None)  # โหลดน้ำหนักเองจาก .pth
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(backbone.last_channel, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x).flatten(1)
            return self.classifier(x)

    def build_transform():
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    """คืนค่า (model | None, status, detail) — ไม่ raise error เพื่อให้แอปไม่ crash"""
    if not TORCH_OK:
        return None, "torch_missing", TORCH_ERR
    if not Path(path).exists():
        return None, "no_weights", ""
    try:
        model = EyeShieldMobileNetV2(num_classes=len(CLASS_KEYS))
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            for k in ("model_state_dict", "state_dict", "model"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    ckpt = ckpt[k]
                    break
        state = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        return model, "ok", ""
    except Exception as exc:
        return None, "load_error", str(exc)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def load_image(data: bytes):
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")
        if max(img.size) > 1600:
            img.thumbnail((1600, 1600))
        return img
    except Exception:
        return None


def image_quality(img: Image.Image):
    """ตรวจคุณภาพภาพแบบคร่าวๆ — คืนลิสต์ของ flag"""
    flags = []
    if min(img.size) < 224:
        flags.append("small")
    small = img.resize((256, 256))
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    if float(lap.var()) < 0.0003:
        flags.append("blurry")
    mean_v = float(gray.mean())
    if mean_v < 0.20:
        flags.append("dark")
    elif mean_v > 0.85:
        flags.append("bright")
    return flags


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@dataclass
class Result:
    probs: dict
    ranking: list
    top_key: str
    confidence: float
    mode: str  # "model" | "heuristic"
    risk: str  # "green" | "yellow" | "red"
    quality: list = field(default_factory=list)


def softmax_np(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def risk_level(top_key: str, conf: float) -> str:
    if top_key == "normal":
        return "green" if conf >= 0.60 else "yellow"
    if top_key == "cataract":
        return "red" if conf >= 0.60 else "yellow"
    return "yellow"  # dry_eye / irritation


def heuristic_probs(img: Image.Image):
    """Safe-Inference: วิเคราะห์ลักษณะภาพเบื้องต้น (ไม่ใช่ Deep Learning)

    - Redness : ความแดงของบริเวณสว่าง (ตาขาว) -> สัมพันธ์กับตาแดง/ระคายเคือง
    - Haze    : บริเวณกลางภาพ (รูม่านตา) สว่าง/ซีดขาว ความอิ่มสีต่ำ -> สัมพันธ์กับความขุ่น
    ค่าเกณฑ์เป็นค่าประมาณ ยังไม่ได้ calibrate กับข้อมูลคลินิก
    """
    small = img.resize((256, 256))
    arr = np.asarray(small, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
    s, v = hsv[..., 1], hsv[..., 2]

    red_excess = (r - g) / (r + g + b + 1e-6)
    bright = v > 0.35
    pool = red_excess[bright] if bright.sum() > 500 else red_excess.ravel()
    redness = float(np.clip((np.percentile(pool, 75) - 0.12) / 0.12, 0.0, 1.0))

    c = slice(96, 160)  # ศูนย์กลาง ~25% ของภาพ
    center_v, center_s = float(v[c, c].mean()), float(s[c, c].mean())
    haze = float(
        np.clip((center_v - 0.35) / 0.40, 0.0, 1.0) * np.clip(1.0 - center_s / 0.5, 0.0, 1.0)
    )

    logits = [1.2 + 1.6 * (1.0 - max(redness, haze)), 2.4 * redness, 2.4 * haze]
    return softmax_np(logits)


def run_inference(img: Image.Image, model) -> Result:
    quality = image_quality(img)
    if model is not None:
        x = build_transform()(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        mode = "model"
    else:
        probs = heuristic_probs(img)
        mode = "heuristic"

    prob_map = {k: float(p) for k, p in zip(CLASS_KEYS, probs)}
    ranking = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)
    top_key, conf = ranking[0]
    return Result(prob_map, ranking, top_key, conf, mode, risk_level(top_key, conf), quality)


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
THAI_FONT_CANDIDATES = [
    APP_DIR / "fonts" / "THSarabunNew.ttf",
    APP_DIR / "fonts" / "Sarabun-Regular.ttf",
    APP_DIR / "fonts" / "NotoSansThai-Regular.ttf",
    APP_DIR / "THSarabunNew.ttf",
    Path("/usr/share/fonts/truetype/tlwg/Garuda.ttf"),  # apt: fonts-thai-tlwg
    Path("/usr/share/fonts/truetype/tlwg/Loma.ttf"),
    Path("/usr/share/fonts/truetype/tlwg/Norasi.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"),
    Path("C:/Windows/Fonts/tahoma.ttf"),
    Path("C:/Windows/Fonts/LeelawUI.ttf"),
    Path("/System/Library/Fonts/Supplemental/Tahoma.ttf"),
    Path("/System/Library/Fonts/Supplemental/Ayuthaya.ttf"),
]
SARABUN_URL = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"


@st.cache_resource(show_spinner=False)
def get_pdf_font():
    """คืนชื่อฟอนต์ที่รองรับภาษาไทย หรือ None ถ้าไม่พบ"""
    if not REPORTLAB_OK:
        return None
    paths = list(THAI_FONT_CANDIDATES)
    cache_path = APP_DIR / ".font_cache" / "Sarabun-Regular.ttf"
    paths.append(cache_path)
    for p in paths:
        try:
            if p.exists():
                pdfmetrics.registerFont(TTFont("EyeShieldFont", str(p)))
                return "EyeShieldFont"
        except Exception:
            continue
    # ลองดาวน์โหลดฟอนต์ (ถ้ามีอินเทอร์เน็ต) — ล้มเหลวได้โดยไม่กระทบแอป
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SARABUN_URL, timeout=8) as resp:
            cache_path.write_bytes(resp.read())
        pdfmetrics.registerFont(TTFont("EyeShieldFont", str(cache_path)))
        return "EyeShieldFont"
    except Exception:
        return None


def build_pdf(img: Image.Image, res: Result, lang: str, report_id: str, stamp: str):
    """คืนค่า (pdf_bytes | None, ภาษาที่ใช้จริงใน PDF)"""
    if not REPORTLAB_OK:
        return None, lang

    thai_font = get_pdf_font()
    if lang == "th" and thai_font is None:
        lang = "en"  # ไม่มีฟอนต์ไทย -> ใช้อังกฤษเพื่อไม่ให้ตัวอักษรเป็นสี่เหลี่ยม
    font = thai_font or "Helvetica"
    T = TEXT[lang]
    wrap = "CJK" if lang == "th" else None  # ไทยไม่มีช่องว่างระหว่างคำ ใช้ CJK-wrap กันข้อความล้น
    lead = 17 if lang == "th" else 14

    def style(size, color=colors.HexColor("#12263f"), leading=None):
        return ParagraphStyle(
            "s%s" % size, fontName=font, fontSize=size, leading=leading or size + (6 if lang == "th" else 4),
            textColor=color, alignment=TA_LEFT, wordWrap=wrap,
        )

    body, small = style(11, leading=lead), style(9, colors.HexColor("#5b6b7f"))
    h1, h2 = style(20, colors.HexColor("#0b5ed7")), style(13, colors.HexColor("#0b5ed7"))

    def P(text, st_=body):
        return Paragraph(escape(str(text)), st_)

    risk_plain = T["risk_%s_plain" % res.risk]
    conf_label = T["conf_label"] if res.mode == "model" else T["conf_label_heur"]
    mode_label = T["pdf_mode_model"] if res.mode == "model" else T["pdf_mode_heur"]
    top_name = T["classes"][res.top_key]
    rc = colors.HexColor(RISK_COLORS[res.risk])

    grid = TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c9d8ee")),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eaf2ff")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )

    story = [P(T["pdf_title"], h1), P("EyeShield AI", small), Spacer(1, 10)]

    meta = Table(
        [
            [P(T["pdf_report_id"]), P(report_id)],
            [P(T["pdf_date"]), P(stamp)],
            [P(T["pdf_mode"]), P(mode_label)],
        ],
        colWidths=[4.2 * cm, 12.8 * cm],
    )
    meta.setStyle(grid)
    story += [meta, Spacer(1, 12)]

    thumb = img.copy()
    thumb.thumbnail((700, 700))
    ib = io.BytesIO()
    thumb.save(ib, format="JPEG", quality=85)
    ib.seek(0)
    w, h = thumb.size
    scale = min(7 * cm / w, 6 * cm / h)
    pic = RLImage(ib, width=w * scale, height=h * scale)
    pic.hAlign = "LEFT"
    story += [pic, Spacer(1, 12), P(T["pdf_summary"], h2), Spacer(1, 4)]

    summary = Table(
        [
            [P(T["risk_label"]), P(risk_plain, ParagraphStyle("r", parent=body, textColor=rc))],
            [P(T["finding_label"]), P(top_name)],
            [P(conf_label), P("%.1f %%" % (res.confidence * 100))],
        ],
        colWidths=[5 * cm, 12 * cm],
    )
    summary.setStyle(grid)
    story += [summary, Spacer(1, 12), P(T["prob_header"], h2), Spacer(1, 4)]

    rows = [[P(T["pdf_class"]), P(T["pdf_prob"])]]
    rows += [[P(T["classes"][k]), P("%.1f %%" % (p * 100))] for k, p in res.ranking]
    probs_tbl = Table(rows, colWidths=[11 * cm, 6 * cm])
    probs_tbl.setStyle(grid)
    story += [probs_tbl, Spacer(1, 12), P(T["advice_header"].replace("💡 ", ""), h2), Spacer(1, 4)]
    story.append(P(T["advice"][res.risk]))

    if res.quality:
        story += [Spacer(1, 10), P(T["quality_header"].replace("⚠️ ", ""), h2), Spacer(1, 4)]
        for q in res.quality:
            story.append(P("- " + T["q_" + q]))

    disc = Table([[P(T["pdf_disclaimer"], style(9.5))]], colWidths=[17 * cm])
    disc.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff6e0")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#e0a100")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story += [Spacer(1, 16), disc]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1 * cm, T["pdf_footer"])
        canvas.drawRightString(A4[0] - 2 * cm, 1 * cm, str(doc_.page))
        canvas.restoreState()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="EyeShield AI Report",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue(), lang


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
CSS = """
<style>
.stApp { background: #f5f9ff; }
section[data-testid="stSidebar"] { background: #eaf2ff; }
.es-banner { background: linear-gradient(90deg,#0b5ed7,#2b8cff); color:#fff; padding:20px 26px;
             border-radius:14px; margin-bottom:18px; }
.es-banner h1 { color:#fff; margin:0; font-size:2rem; }
.es-banner p  { color:#e6f0ff; margin:4px 0 0 0; font-size:1.02rem; }
.es-card { background:#fff; border-radius:12px; padding:14px 16px; box-shadow:0 1px 4px rgba(11,94,215,.12);
           color:#12263f; height:100%; }
.es-label { font-size:.82rem; color:#5b6b7f; margin-bottom:4px; }
.es-value { font-size:1.25rem; font-weight:700; line-height:1.3; }
</style>
"""


def metric_card(label: str, value: str, color: str = "#0b5ed7") -> str:
    return (
        '<div class="es-card" style="border-left:6px solid %s">'
        '<div class="es-label">%s</div><div class="es-value">%s</div></div>' % (color, label, value)
    )


def main():
    st.set_page_config(page_title="EyeShield AI", page_icon="👁️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    model, status, detail = load_model(str(MODEL_PATH))

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("## 👁️ EyeShield AI")
        lang_choice = st.radio(TEXT["th"]["lang_label"], ["TH", "EN"], horizontal=True)
        lang = lang_choice.lower()
        T = TEXT[lang]

        st.markdown("---")
        st.markdown("#### %s" % T["disc_title"])
        st.warning(T["disc_body"])

        st.markdown("---")
        st.markdown("#### %s" % T["sys_status"])
        if status == "ok":
            st.success(T["mode_model_badge"])
        else:
            st.info(T["mode_heur_badge"])
            if detail:
                with st.expander("details"):
                    st.code(detail)

    # ---------------- Header ----------------
    st.markdown(
        '<div class="es-banner"><h1>👁️ EyeShield AI</h1><p>%s</p></div>' % T["subtitle"],
        unsafe_allow_html=True,
    )
    if status != "ok":
        st.warning(T["fallback_notice"])

    # ---------------- Input ----------------
    source = st.radio(T["input_mode"], [T["mode_upload"], T["mode_camera"]], horizontal=True)
    if source == T["mode_upload"]:
        file = st.file_uploader(T["upload_label"], type=["jpg", "jpeg", "png"])
    else:
        file = st.camera_input(T["camera_label"])

    st.markdown(T["tips"])
    if file is None:
        st.info(T["no_image"])
        return

    data = file.getvalue()
    img = load_image(data)
    if img is None:
        st.error(T["bad_image"])
        return

    # ---------------- Inference ----------------
    with st.spinner(T["analyzing"]):
        res = run_inference(img, model)

    img_hash = hashlib.md5(data).hexdigest()
    if st.session_state.get("img_hash") != img_hash:
        st.session_state["img_hash"] = img_hash
        st.session_state["report_id"] = "ES-%s-%s" % (
            datetime.now().strftime("%Y%m%d"), uuid.uuid4().hex[:6].upper()
        )
        st.session_state["stamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------- Results ----------------
    st.markdown("### %s" % T["results_header"])
    col_img, col_res = st.columns([1, 1.5], gap="large")

    with col_img:
        st.image(img, caption=T["img_caption"], width=380)

    with col_res:
        conf_label = T["conf_label"] if res.mode == "model" else T["conf_label_heur"]
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            metric_card(T["risk_label"], T["risk_" + res.risk], RISK_COLORS[res.risk]),
            unsafe_allow_html=True,
        )
        c2.markdown(
            metric_card(T["finding_label"], T["classes"][res.top_key].split(" (")[0]),
            unsafe_allow_html=True,
        )
        c3.markdown(
            metric_card(conf_label, "%.1f%%" % (res.confidence * 100)),
            unsafe_allow_html=True,
        )

        st.markdown("")
        st.markdown("**%s**" % T["prob_header"])
        for key, p in res.ranking:
            st.write("%s — %.1f%%" % (T["classes"][key], p * 100))
            st.progress(float(min(max(p, 0.0), 1.0)))

        if res.quality:
            st.markdown("**%s**" % T["quality_header"])
            for q in res.quality:
                st.caption("• " + T["q_" + q])

        st.markdown("**%s**" % T["advice_header"])
        box = {"green": st.success, "yellow": st.warning, "red": st.error}[res.risk]
        box(T["advice"][res.risk])

    # ---------------- Actions ----------------
    st.markdown("---")
    a1, a2 = st.columns(2, gap="large")

    with a1:
        if not REPORTLAB_OK:
            st.warning(T["pdf_unavailable"])
        else:
            try:
                pdf_bytes, used_lang = build_pdf(
                    img, res, lang, st.session_state["report_id"], st.session_state["stamp"]
                )
                st.download_button(
                    T["pdf_btn"],
                    data=pdf_bytes,
                    file_name="EyeShield_Report_%s.pdf" % st.session_state["report_id"],
                    mime="application/pdf",
                )
                if used_lang != lang and T["pdf_font_missing"]:
                    st.caption(T["pdf_font_missing"])
            except Exception as exc:
                st.error("PDF error: %s" % exc)

    with a2:
        st.markdown("**%s**" % T["clinic_header"])
        area = st.text_input(T["area_input"], key="area_input")
        query = "%s %s" % (T["clinic_query"], area.strip() if area.strip() else T["clinic_near_me"])
        url = "https://www.google.com/maps/search/?api=1&query=" + quote_plus(query)
        if hasattr(st, "link_button"):
            st.link_button(T["clinic_btn"], url)
        else:
            st.markdown("[%s](%s)" % (T["clinic_btn"], url))


main()
