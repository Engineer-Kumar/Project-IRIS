import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import torch
import time
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering 
import easyocr
import pyttsx3
import json
import base64
import pygame

# ---- DEVICE ----
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- LOAD BLIP MODELS ----
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# ---- FUNCTION TO CONVERT LOCAL IMAGE TO BASE64 ----
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# ---- SPLASH SCREEN with Rotating Logo ----
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    st.set_page_config(page_title="IRIS", layout="wide")
    logo_path = os.path.join("assets", "iris_logo.png")
    img_base64 = get_base64_image(logo_path)
    st.markdown(f"""
        <style>
        body {{ background-color: #0f172a; color: white; font-family: 'Segoe UI', sans-serif; }}
        .splash-logo {{ text-align: center; margin-top: 100px; }}
        .splash-logo img {{ width: 150px; height: auto; object-fit: contain; animation: float 3s ease-in-out infinite; filter: drop-shadow(0 0 12px #3b82f6); }}
        @keyframes rotateLogo {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
        .splash-title {{ text-align: center; font-size: 48px; font-weight: bold; margin-top: 20px; color: #60a5fa; text-shadow: 2px 2px 8px #000; }}
        .splash-subtitle {{ text-align: center; font-size: 22px; color: #94a3b8; font-style: italic; }}
        .infinity-loader {{ display: flex; justify-content: center; align-items: center; font-size: 60px; color: #38bdf8; margin-top: 50px; animation: spin 3s linear infinite; }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        </style>
        <div class="splash-logo"><img src="data:image/png;base64,{img_base64}" alt="IRIS Logo"></div>
        <div class="splash-title">IRIS</div>
        <div class="splash-subtitle">Smart Vision Assistant</div>
        <div class="infinity-loader">∞</div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    st.session_state.splash_shown = True
    st.rerun()

# ---- PATHS ----
CUSTOM_MODEL_PATH = "models/bestpothole.pt"
COCO_MODEL_PATH = "models/coco/yolov8s.pt"
HISTORY_FILE = "detection_history.json"

# ---- LOAD MODELS ----
if not os.path.exists(CUSTOM_MODEL_PATH):
    st.error(f"Custom model not found at: {CUSTOM_MODEL_PATH}")
    st.stop()
if not os.path.exists(COCO_MODEL_PATH):
    st.error(f"COCO model not found at: {COCO_MODEL_PATH}")
    st.stop()

custom_model = YOLO(CUSTOM_MODEL_PATH)
coco_model = YOLO(COCO_MODEL_PATH)

# ---- OCR ----
reader = easyocr.Reader(['en'], gpu=False)

# ---- SPEECH AND SOUND ----
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        st.warning("🗣 Voice engine busy. Please wait or restart app.")

def play_beep():
    pygame.mixer.init()
    beep_sound = pygame.mixer.Sound("beep.wav")
    beep_sound.play()

# ---- DETECTION DISTANCE LOGIC ----
def is_pothole_close(xyxy, frame_height, threshold_distance_m=150):
    """
    Estimate distance based on bounding box vertical position (y2).
    The closer to the bottom of the frame, the closer the pothole.
    """
    y2 = xyxy[3]  # Bottom of the bounding box
    normalized = y2 / frame_height  # 1.0 = bottom, 0.0 = top
    estimated_distance_m = (1 - normalized) * 150  # Approximate 10m range
    return estimated_distance_m <= threshold_distance_m

# ---- CAPTION ----
def get_caption(pil_image):
    inputs = caption_processor(images=pil_image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    return caption_processor.decode(out[0], skip_special_tokens=True)

# ---- VQA ----
def answer_question(pil_image, question):
    inputs = vqa_processor(images=pil_image, text=question, return_tensors="pt").to(device)
    output_ids = vqa_model.generate(pixel_values=inputs["pixel_values"], input_ids=inputs["input_ids"])
    answer = vqa_processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# ---- PAGE UI ----
# ---- PAGE CONFIG ----

st.set_page_config(page_title="IRIS: Visual AI", layout="wide")

# ---- CUSTOM STYLES ----
st.markdown("""
<style>
/* --- Background & Layout with Animated Gradient --- */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    animation: gradientFlow 10s ease infinite;
    background-size: 400% 400%;
}
@keyframes gradientFlow {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* --- Font and Default Styling --- */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    font-size: 18px;
    color: white;
}

/* --- Header Logo Animation (ROTATING + PULSATING) --- */
.logo-box {
    text-align: center;
    margin-top: 1rem;
}
.logo-box img {
    width: 110px;
    height: 110px;
    border-radius: 50%;
    border: 4px solid #60a5fa;
    box-shadow: 0 0 10px #60a5fa;
    animation: rotateLogo 4s linear infinite, pulseLogo 4s ease-in-out infinite;
}
@keyframes rotateLogo {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
@keyframes pulseLogo {
    0%, 100% { box-shadow: 0 0 20px #60a5fa; }
    50% { box-shadow: 0 0 35px #3b82f6; }
}

/* --- Main Title Animation --- */
.iris-title {
    font-size: 52px;
    font-weight: bold;
    color: #ffffff;
    text-shadow: 2px 2px #000;
    text-align: center;
    animation: floatText 3s ease-in-out infinite;
}
@keyframes floatText {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}

/* --- Subtitle Styling --- */
.subtitle {
    text-align: center;
    font-size: 22px;
    color: #cbd5e1;
}
.tagline {
    text-align: center;
    font-style: italic;
    color: #94a3b8;
    margin-top: 0.5em;
    margin-bottom: 2em;
}

/* --- Tabs with Hover Glow --- */
.stTabs [data-baseweb="tab"] {
    background-color: #1e293b !important;
    color: white !important;
    font-size: 20px;
    font-weight: 600;
    border-radius: 10px 10px 0 0;
    padding: 1rem;
    margin-right: 8px;
    box-shadow: 0 0 5px #0ea5e9;
    transition: 0.3s;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #60a5fa !important;
    background-color: #0f172a !important;
    transform: scale(1.04);
    box-shadow: 0 0 15px #38bdf8;
}

/* --- File Uploader Neon Box --- */
.stFileUploader {
    background-color: rgba(0, 0, 0, 0.6) !important;
    padding: 1.5rem;
    border-radius: 16px;
    border: 2px dashed #60a5fa;
    box-shadow: 0 0 15px rgba(96, 165, 250, 0.6);
    margin-bottom: 2rem;
    transition: 0.4s;
}
.stFileUploader:hover {
    border-color: #3b82f6;
    box-shadow: 0 0 25px #38bdf8;
}

/* --- Centered Button Container --- */
div.stButton {
    display: flex;
    justify-content: center;
    align-items: center;
}

/* --- Button Styling --- */
.stButton>button {
    background: linear-gradient(to right, #3b82f6, #06b6d4);
    color: white;
    font-size: 22px;
    font-weight: bold;
    border: none;
    border-radius: 14px;
    padding: 1rem 2rem;
    margin: 1rem;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    transition: 0.4s ease;
}
.stButton>button:hover {
    background: linear-gradient(to right, #2563eb, #0891b2);
    transform: scale(1.08);
    box-shadow: 0 0 20px #3b82f6;
}

/* --- Radio Buttons (mode selection) --- */
.stRadio > div {
    flex-direction: row !important;
    justify-content: center !important;
    gap: 1.5rem;
    padding: 1rem 0;
}
.stRadio > div label {
    background-color: rgba(30, 41, 59, 0.85);
    padding: 1rem 2rem;
    border-radius: 12px;
    color: #e2e8f0;
    font-weight: bold;
    font-size: 20px;
    cursor: pointer;
    transition: 0.3s ease;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.4);
}
.stRadio > div label:hover {
    background-color: #334155;
    color: #60a5fa;
    transform: scale(1.08);
}
</style>
""", unsafe_allow_html=True)
# ---- Load base64 logo from assets ----
logo_base64 = get_base64_image("assets/iris_logo.png")

# ---- HEADER UI ----
st.markdown(f"""
    <style>
    .logo-box {{
        text-align: center;
        margin-top: 5px;
        margin-bottom: 10px;
    }}
    .logo-box img {{
        width: 150px;
        height: auto;
        filter: drop-shadow(0 0 10px #3b82f6);
        animation: float 3s ease-in-out infinite;
    }}
    @keyframes float {{
        0% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-10px); }}
        100% {{ transform: translateY(0); }}
    }}
    .iris-title {{
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #60a5fa;
        margin-top: 10px;
        text-shadow: 2px 2px 8px #000;
    }}
    </style>

    <div class="logo-box">
        <img src="data:image/png;base64,{logo_base64}" alt="IRIS Logo">
    </div>

    <div class="iris-title">
        IRIS: Visual AI for Smart Interaction System
    </div>

    <p style='text-align:center; font-size: 24px; color: #e2e8f0; font-weight: 500; margin-top: 8px;'>
        <b>SEE. SENSE. UNDERSTAND.</b><br>Going beyond detection to interaction.
    </p>
""", unsafe_allow_html=True)

# ---- HISTORY ----
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history[-5:], f, indent=2)

if "history" not in st.session_state:
    st.session_state.history = load_history()

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["🖼 Image", "🎞 Video", "📷 Live Camera"])


# ---- TAB 1: IMAGE ----
with tab1:
    uploaded_file = st.file_uploader("📷 Upload an Image for Analysis", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        results_custom = custom_model(image_np)[0]
        results_coco = coco_model(image_np)[0]

        annotated = image_np.copy()
        labels = []
        for r in [results_custom, results_coco]:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = r.names[cls]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                if label.lower() == "pothole" and is_pothole_close(xyxy, image_np.shape[0]):
                    speak_text("Pothole ahead. Be careful.")
                    play_beep()
                if label not in labels:
                    labels.append(label)
                cv2.rectangle(annotated, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                cv2.putText(annotated, label, xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        col1, col2 = st.columns(2)
        col1.image(image_np, caption="Original Image", use_container_width=True)
        col2.image(annotated, caption="Detected Image", use_container_width=True)

        caption = ""
        if "pothole" in labels:
            st.warning("🚨 Pothole detected ahead!")

        st.session_state.history.append({
            "Mode": "Image",
            "Detected": ", ".join(labels),
            "Caption": caption,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_history(st.session_state.history)

        if "pothole" in labels:
            st.markdown('<div class="alert">🚨 ALERT: POTHOLE DETECTED! BE CAREFUL!</div>', unsafe_allow_html=True)

        with st.expander("🔍 OCR Result"):
            ocr_result = reader.readtext(image_np)
            text = "\n".join([item[1] for item in ocr_result]) or "No text detected."
            st.markdown(f'<div style="background-color:#fef3c7;padding:1rem;border-radius:10px;color:black;">{text}</div>', unsafe_allow_html=True)

        with st.expander("🧠 Image Caption"):
            caption = get_caption(image)
            st.markdown(f'<div style="background-color:#fef3c7;padding:1rem;border-radius:10px;color:black;">{caption}</div>', unsafe_allow_html=True)
            if st.button("🔊 Speak Caption"):
                speak_text(caption)

        with st.expander("❓ Ask a Question About the Image"):
            question = st.text_input("Your Question")

            if question:
                with st.spinner("🤖 Thinking..."):
                    try:
                        answer = answer_question(image, question)
                        st.success(f"🗨 Answer: {answer}")
                        if st.button("🔊 Speak Answer"):
                            speak_text(answer)
                    except Exception as e:
                        st.error(f"⚠ Failed to generate answer: {e}")

        st.session_state.history.append({
            "Mode": "Image",
            "Detected": ", ".join(labels),
            "Caption": caption,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        save_history(st.session_state.history)

        with st.expander("📜 Detection History"):
            st.dataframe(st.session_state.history[-5:])

# ---- VIDEO TAB ----
with tab2:
    st.header("🎞 Upload a Video for Analysis")
    video_file = st.file_uploader("📼 Choose a video file...", type=["mp4", "mov", "avi"], key="video_uploader")

    if video_file:
        if st.button("▶ Start Video"):
            temp_video = tempfile.NamedTemporaryFile(delete=False)
            temp_video.write(video_file.read())
            temp_video_path = temp_video.name

            cap = cv2.VideoCapture(temp_video_path)
            stframe = st.empty()
            stop_video = st.button("⛔ Stop Video")

            # Get video FPS for smoother display
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            delay = 1 / frame_rate if frame_rate > 0 else 0.03

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run predictions from both models
                results_custom = custom_model.predict(frame_rgb, imgsz=360, conf=0.3)[0]
                results_coco = coco_model.predict(frame_rgb, imgsz=360, conf=0.3)[0]

                # Combine results
                for r in [results_custom, results_coco]:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        label = r.names[cls]
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)

                        if label.lower() == "pothole" and is_pothole_close(xyxy, frame.shape[0], threshold_distance_m=5):
                            play_beep()
                            cv2.putText(frame_rgb, "⚠ Pothole ahead! Be careful!", (30, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                        cv2.rectangle(frame_rgb, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                        cv2.putText(frame_rgb, label, xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                stframe.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(delay)

            cap.release()
            st.success("✅ Video processing complete.")



# ---- TAB 3: LIVE ----
with tab3:
    st.header("📷 Live Camera Detection")
    start_button = st.button("▶ Start Live Camera")

    if start_button:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        stop_signal = st.button("⛔ Stop Camera")

        while cap.isOpened() and not stop_signal:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_custom = custom_model.predict(frame_rgb, imgsz=360, conf=0.3)[0]
            results_coco = coco_model.predict(frame_rgb, imgsz=360, conf=0.3)[0]

            for r in [results_custom, results_coco]:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = r.names[cls]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)

                    if label.lower() == "pothole" and is_pothole_close(xyxy, frame.shape[0], threshold_distance_m=5):
                        speak_text("Pothole ahead. Be careful.")
                        play_beep()
                        cv2.putText(frame_rgb, "⚠ Pothole ahead! Be careful!", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    cv2.rectangle(frame_rgb, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                    cv2.putText(frame_rgb, label, xyxy[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            stframe.image(frame_rgb, channels="RGB", use_container_width=800)

        cap.release()
        st.success("✅ Live detection stopped.")