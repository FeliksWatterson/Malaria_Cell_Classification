import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# System Configuration
st.set_page_config(
    page_title="Malaria Classification System",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; background-color: #4b4bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# Model Initialization
@st.cache_resource
def load_core_model():
    # --- CẬP NHẬT QUAN TRỌNG ---
    # Tự động lấy đường dẫn thư mục chứa file app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Nối đường dẫn để tìm model.h5 nằm cùng thư mục
    model_path = os.path.join(current_dir, 'malaria_model.h5')
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        return None
    return load_model(model_path)

model = load_core_model()

# Core Processing Functions
def preprocess_input(image_arr):
    img = cv2.resize(image_arr, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def segment_and_classify(opencv_image, model):
    output_img = opencv_image.copy()
    
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    parasitized_count = 0
    uninfected_count = 0
    total_contours = len(contours)
    
    prog_bar = st.progress(0)
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 200: 
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        roi = opencv_image[y:y+h, x:x+w]
        
        try:
            processed_roi = preprocess_input(roi)
            prediction = model.predict(processed_roi, verbose=0)
            score = prediction[0][0]
            
            if score > 0.5:
                color = (255, 0, 0) # Red
                parasitized_count += 1
            else:
                color = (0, 0, 255 ) 
                uninfected_count += 1
                
            cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
            
        except Exception:
            pass
        
        if i % 10 == 0:
            prog_bar.progress((i + 1) / total_contours)
            
    prog_bar.empty()
    return output_img, parasitized_count, uninfected_count

# Main Layout
st.title("Malaria Cell Classification System")

analysis_mode = st.sidebar.selectbox(
    "Analysis Mode",
    ("Single Cell Inference", "Whole Slide Segmentation")
)

uploaded_file = st.file_uploader("Input Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    input_image = cv2.imdecode(file_bytes, 1)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(input_image, caption="Input Source", use_column_width=True)

    # Mode 1: Single Cell
    if analysis_mode == "Single Cell Inference":
        if st.button("Run Inference"):
            with st.spinner("Processing..."):
                input_tensor = preprocess_input(input_image)
                pred = model.predict(input_tensor)
                score = pred[0][0]
                
                with col2:
                    st.subheader("Inference Result")
                    if score > 0.5:
                        st.error(f"Class: Parasitized")
                        st.info(f"Confidence: {score:.4f}")
                    else:
                        st.success(f"Class: Uninfected")
                        st.info(f"Confidence: {(1-score):.4f}")

    # Mode 2: Whole Slide
    elif analysis_mode == "Whole Slide Segmentation":
        if st.button("Execute Segmentation"):
            with st.spinner("Segmenting & Classifying..."):
                result_img, p_count, u_count = segment_and_classify(input_image, model)
                
                with col2:
                    st.image(result_img, caption="Annotated Output", use_column_width=True)
                
                st.divider()
                st.subheader("Statistical Report")
                c1, c2, c3 = st.columns(3)
                c1.metric("Total ROI Detected", p_count + u_count)
                c2.metric("Parasitized", p_count)
                c3.metric("Uninfected", u_count)