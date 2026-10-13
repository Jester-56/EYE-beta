import streamlit as st
import numpy as np
from PIL import Image
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time

# --- 1. ตั้งค่าหน้าเว็บ Streamlit ---
st.set_page_config(
    page_title="EyeShield AI - Clinical Screening",
    page_icon="👁️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. พจนานุกรมสองภาษา (Translations Dictionary) ---
TEXTS = {
    "TH": {
        "title": "👁️ EyeShield AI",
        "subtitle": "ระบบปัญญาประดิษฐ์คัดกรองความเสี่ยงโรคดวงตาเบื้องต้น",
        "sidebar_title": "EyeShield AI",
        "sidebar_ver": "v1.3.0 | สื่อคัดกรองทางการแพทย์",
        "lang_select": "🌐 เลือกภาษา / Language",
        "how_to": "📋 วิธีการใช้งาน",
        "step_1": "1. เลือกวิธีส่งภาพ (อัปโหลด หรือ ถ่ายสด)",
        "step_2": "2. กดปุ่ม 'เริ่มวิเคราะห์ด้วย AI'",
        "step_3": "3. ตรวจสอบผลการคัดกรองและคำแนะนำ",
        "step_4": "4. ดาวน์โหลด PDF เพื่อเก็บประวัติ",
        "disclaimer_title": "⚠️ ข้อความเตือนทางการแพทย์:",
        "disclaimer_text": "ระบบนี้เป็นเครื่องมือคัดกรองเบื้องต้นเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยทางการแพทย์โดยจักษุแพทย์ได้",
        "upload_mode": "📥 ช่องทางนำเข้าภาพดวงตา",
        "tab_upload": "📁 อัปโหลดไฟล์ภาพ",
        "tab_camera": "📷 ถ่ายภาพสดผ่านกล้อง",
        "btn_analyze": "🔍 เริ่มวิเคราะห์ภาพด้วย AI",
        "processing": "🧠 AI กำลังประมวลผลโครงสร้างดวงตา...",
        "success": "✅ ประมวลผลเสร็จสิ้น",
        "results_title": "📊 ผลการวิเคราะห์ (Screening Results)",
        "confidence_label": "ค่าความมั่นใจของ AI:",
        "desc_label": "รายละเอียดการตรวจ:",
        "rec_label": "คำแนะนำการปฏิบัติตัว:",
        "btn_pdf": "📄 ดาวน์โหลดรายงาน PDF",
        "btn_map": "🗺️ ค้นหาคลินิกตาใกล้เคียง",
        "results": [
            {
                "class": "normal",
                "status": "🟢 สุขภาพตาปกติ (Normal)",
                "desc": "ไม่พบรอยโรคหรือสัญญาณความผิดปกติเด่นชัด เลนส์ตาและเยื่อบุตาอยู่ในเกณฑ์ปกติ",
                "confidence": "96.4%",
                "recommendation": "ควรตรวจสุขภาพตาประจำปี และพักสายตาตามหลัก 20-20-20"
            },
            {
                "class": "warning",
                "status": "🟡 สงสัยภาวะตาแห้ง / ระคายเคือง (Dry Eye / Irritation)",
                "desc": "พบเส้นเลือดขยายตัวเล็กน้อยบริเวณตาขาว และลักษณะพื้นผิวตาขาดความชุ่มชื้น",
                "confidence": "88.2%",
                "recommendation": "พักการใช้สายตากับหน้าจอ หหยอดน้ำตาเทียมเพื่อเพิ่มความชุ่มชื้น หากไม่ดีขึ้นควรพบแพทย์"
            },
            {
                "class": "danger",
                "status": "🔴 สงสัยต้อกระจก / ตาแดงรุนแรง (Suspected Cataract / Red Eye)",
                "desc": "พบความขุ่นบริเวณเลนส์ตา หรือมีการอักเสบขยายตัวของเส้นเลือดในตาขาวอย่างเด่นชัด",
                "confidence": "91.5%",
                "recommendation": "ควรเข้าพบจักษุแพทย์เพื่อทำการตรวจวินิจฉัยด้วยเครื่องมือเฉพาะทางโดยเร็ว"
            }
        ]
    },
    "EN": {
        "title": "👁️ EyeShield AI",
        "subtitle": "AI-Powered Preliminary Ophthalmic Risk Screening System",
        "sidebar_title": "EyeShield AI",
        "sidebar_ver": "v1.3.0 | Clinical Screening Tool",
        "lang_select": "🌐 Select Language / เลือกภาษา",
        "how_to": "📋 How to Use",
        "step_1": "1. Choose input method (Upload or Camera)",
        "step_2": "2. Click 'Analyze Image with AI'",
        "step_3": "3. Review screening status and guidance",
        "step_4": "4. Download PDF report for records",
        "disclaimer_title": "⚠️ Medical Disclaimer:",
        "disclaimer_text": "This system is a preliminary screening tool and cannot replace official diagnosis by an ophthalmologist.",
        "upload_mode": "📥 Image Input Mode",
        "tab_upload": "📁 Upload Image File",
        "tab_camera": "📷 Live Camera Capture",
        "btn_analyze": "🔍 Analyze Image with AI",
        "processing": "🧠 AI is analyzing eye structures...",
        "success": "✅ Analysis Complete",
        "results_title": "📊 Screening Results",
        "confidence_label": "AI Confidence Score:",
        "desc_label": "Clinical Findings:",
        "rec_label": "Recommended Action:",
        "btn_pdf": "📄 Download PDF Report",
        "btn_map": "🗺️ Find Nearby Eye Clinics",
        "results": [
            {
                "class": "normal",
                "status": "🟢 Normal Healthy Eye",
                "desc": "No distinct lesions or abnormal signs detected. Lens and conjunctiva are within normal ranges.",
                "confidence": "96.4%",
                "recommendation": "Maintain annual eye checkups and follow the 20-20-20 eye rest rule."
            },
            {
                "class": "warning",
                "status": "🟡 Suspected Dry Eye / Irritation",
                "desc": "Mild vascular expansion observed on sclera with signs of tear film instability.",
                "confidence": "88.2%",
                "recommendation": "Reduce screen time, apply lubricating eye drops. Consult a doctor if symptoms persist."
            },
            {
                "class": "danger",
                "status": "🔴 Suspected Cataract / Severe Red Eye",
                "desc": "Lens opacity or significant inflammation/hyperemia of conjunctival vessels detected.",
                "confidence": "91.5%",
                "recommendation": "Prompt examination by a certified ophthalmologist is highly recommended."
            }
        ]
    }
}

# --- 3. Custom CSS ปรับดีไซน์ ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border-left: 6px solid #0d6efd;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .status-normal { border-left-color: #198754 !important; background-color: #f1f9f5; }
    .status-warning { border-left-color: #ffc107 !important; background-color: #fffdf5; }
    .status-danger { border-left-color: #dc3545 !important; background-color: #fdf5f5; }

    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- 4. Sidebar (สลับภาษา & คำอธิบาย) ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/ophthalmology.png", width=70)
    st.title("EyeShield AI")
    
    # สวิตช์สลับภาษา
    lang_choice = st.radio("Language / ภาษา", ["ไทย (TH)", "English (EN)"], horizontal=True)
    lang = "TH" if "ไทย" in lang_choice else "EN"
    t = TEXTS[lang]
    
    st.caption(t["sidebar_ver"])
    st.markdown("---")
    st.markdown(f"### {t['how_to']}")
    st.markdown(f"{t['step_1']}\n{t['step_2']}\n{t['step_3']}\n{t['step_4']}")
    
    st.markdown("---")
    st.warning(f"**{t['disclaimer_title']}**\n{t['disclaimer_text']}")

# --- 5. Header หน้าหลัก ---
st.title(t["title"])
st.markdown(f"##### {t['subtitle']}")
st.markdown("---")

# --- 6. แท็บเลือกวิธีนำเข้าภาพ (Upload / Camera) ---
st.markdown(f"#### {t['upload_mode']}")
tab1, tab2 = st.tabs([t["tab_upload"], t["tab_camera"]])

uploaded_image = None

with tab1:
    file_input = st.file_uploader("JPG, JPEG, PNG", type=["jpg", "jpeg", "png"], key="file_up")
    if file_input:
        uploaded_image = Image.open(file_input)

with tab2:
    cam_input = st.camera_input("ถ่ายภาพดวงตาผ่านกล้อง", key="cam_up")
    if cam_input:
        uploaded_image = Image.open(cam_input)

# --- 7. ประมวลผลและแสดงผลลัพธ์ ---
if uploaded_image is not None:
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(uploaded_image, caption="Selected Image", use_container_width=True)
    
    st.markdown("---")
    
    if st.button(t["btn_analyze"], type="primary"):
        with st.spinner(t["processing"]):
            time.sleep(1.8)
            res = t["results"][1] # เคสตัวอย่าง Demo

        st.success(t["success"])
        st.markdown(f"### {t['results_title']}")
        
        # แสดง Card ผลลัพธ์
        card_class = f"metric-card status-{res['class']}"
        st.markdown(f"""
            <div class="{card_class}">
                <h3 style="margin-top:0; color:#1a252f;">{res['status']}</h3>
                <p><b>{t['desc_label']}</b> {res['desc']}</p>
                <p><b>{t['rec_label']}</b> {res['recommendation']}</p>
                <hr style="margin: 10px 0;">
                <p style="margin-bottom:0; font-size: 0.9em; color: #555;">
                    <b>{t['confidence_label']}</b> <span style="color:#0d6efd; font-weight:bold;">{res['confidence']}</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # --- 8. สร้าง PDF รายงานผล ---
        def generate_pdf(status, desc, rec, conf):
            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            
            p.setFont("Helvetica-Bold", 18)
            p.setFillColorRGB(0.05, 0.28, 0.63)
            p.drawString(50, height - 50, "EyeShield AI - Screening Report")
            
            p.setFont("Helvetica", 9)
            p.setFillColorRGB(0.3, 0.3, 0.3)
            p.drawString(50, height - 65, "Automated Preliminary Ophthalmic Screening System")
            p.line(50, height - 75, width - 50, height - 75)
            
            p.setFont("Helvetica-Bold", 11)
            p.setFillColorRGB(0, 0, 0)
            p.drawString(50, height - 110, f"Status: {status}")
            p.drawString(50, height - 130, f"Confidence: {conf}")
            p.drawString(50, height - 150, f"Findings: {desc[:60]}...")
            p.drawString(50, height - 170, f"Action: {rec[:60]}...")
            
            p.line(50, 100, width - 50, 100)
            p.setFont("Helvetica-Bold", 8)
            p.setFillColorRGB(0.8, 0.1, 0.1)
            p.drawString(50, 85, "DISCLAIMER:")
            p.setFont("Helvetica", 8)
            p.setFillColorRGB(0.3, 0.3, 0.3)
            p.drawString(110, 85, "AI-generated preliminary screening result only. Consult an ophthalmologist for official diagnosis.")
            
            p.showPage()
            p.save()
            buffer.seek(0)
            return buffer

        pdf_data = generate_pdf(res['status'], res['desc'], res['recommendation'], res['confidence'])

        # ปุ่มดำเนินการ
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            st.download_button(
                label=t["btn_pdf"],
                data=pdf_data,
                file_name="eyeshield_report.pdf",
                mime="application/pdf"
            )
        with col_act2:
            st.markdown(
                '<a href="https://www.google.com/maps/search/โรงพยาบาลตา+หรือ+คลินิกตาใกล้ฉัน" target="_blank">'
                f'<button style="width:100%; background-color:#0d6efd; color:white; border:none; padding:10px; border-radius:8px; cursor:pointer; font-weight:bold; height:3.2em;">'
                f'{t["btn_map"]}</button></a>',
                unsafe_allow_html=True
            )