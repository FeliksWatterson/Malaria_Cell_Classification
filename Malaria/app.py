import os
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from modules.model import build_model
from modules.preprocessor import manual_histogram_equalization, manual_resize

st.set_page_config(
    page_title="Hệ thống hỗ trợ chẩn đoán ký sinh trùng sốt rét",
    layout="wide"
)

css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_core_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "malaria_model.keras") 

    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        return None

    try:
        model = build_model()
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

model = load_core_model()

def preprocess_input(image_array):
    img_equalized = manual_histogram_equalization(image_array)
    img_resized = manual_resize(img_equalized, (128, 128))
    img_normalized = img_resized.astype("float32") / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def segment_and_classify(opencv_image, model, conf_threshold=0.5, min_area=200):
    output_img = opencv_image.copy()

    gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2) 
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    parasitized_count = 0
    uninfected_count = 0
    total_contours = len(contours)
    
    prog_bar = st.progress(0)

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = opencv_image[y:y+h, x:x+w]

        try:
            processed_roi = preprocess_input(roi)
            prediction = model.predict(processed_roi, verbose=0)
            score = prediction[0][0]

            if score > conf_threshold: 
                color = (255, 0, 0) 
                parasitized_count += 1
                label = f"M: {score*100:.1f}%"
            else:
                color = (0, 255, 0) 
                uninfected_count += 1
                label = f"N: {(1-score)*100:.1f}%"

            cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
            text_y = y - 8 if y - 8 > 15 else y + h + 20
            cv2.putText(output_img, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        except Exception:
            pass

        if i % 10 == 0:
            prog_bar.progress((i + 1) / total_contours)

    prog_bar.empty()
    return output_img, parasitized_count, uninfected_count

st.markdown(
    "<h2 style='text-align: center; margin-bottom: 0.5rem;'>Hệ thống hỗ trợ chẩn đoán ký sinh trùng sốt rét</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; color: #555;'>Hệ thống tự động phân tích và hỗ trợ tầm soát ký sinh trùng trên tiêu bản lam máu</p>",
    unsafe_allow_html=True,
)

tab_main, tab_detail, tab_info = st.tabs(
    ["Chẩn đoán hình ảnh", "Phân tích kỹ thuật", "Thông tin hệ thống"]
)

with tab_main:
    mode = st.radio(
        "Chế độ phân tích",
        ("Khảo sát tế bào đơn lẻ", "Quét toàn bộ tiêu bản rộng"),
        horizontal=True,
    )

    uploaded_files = st.file_uploader(
        "Tải lên ảnh (có thể chọn nhiều ảnh)",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files and model is not None:
        st.markdown("**Danh sách ảnh đã chọn:**")
        for f in uploaded_files:
            st.write(f"- {f.name}")

        if mode == "Khảo sát tế bào đơn lẻ":
            if st.button("Thực hiện phân tích lâm sàng"):
                results = []
                with st.spinner("Hệ thống đang quét và trích xuất đặc trưng..."):
                    for uploaded_file in uploaded_files:
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        input_image = cv2.imdecode(file_bytes, 1)
                        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

                        input_tensor = preprocess_input(input_image)
                        pred = model.predict(input_tensor)
                        score = pred[0][0]

                        if score > 0.5:
                            label = "DƯƠNG TÍNH (Phát hiện Ký sinh trùng)"
                            color = "error"
                            conf = float(score)
                        else:
                            label = "ÂM TÍNH (Tế bào hồng cầu bình thường)"
                            color = "success"
                            conf = float(1 - score)

                        display_image = cv2.resize(input_image, (250, 250))

                        results.append(
                            {
                                "name": uploaded_file.name,
                                "image": display_image,
                                "label": label,
                                "color": color,
                                "confidence": conf,
                            }
                        )

                num_cols = 3
                for i in range(0, len(results), num_cols):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i + j
                        if idx >= len(results):
                            break
                        r = results[idx]
                        with cols[j]:
                            st.image(r["image"], use_container_width=True)
                            diag_color = "#d32f2f" if r["color"] == "error" else "#2e7d32"
                            st.markdown(f"""
                            <div style='background-color: #ffffff; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); text-align: center; min-height: 120px; display: flex; flex-direction: column; justify-content: space-between; margin-top: -5px;'>
                                <div style='font-size: 0.8rem; color: #555; word-break: break-all;'>{r['name']}</div>
                                <div style='color: {diag_color}; font-weight: bold; font-size: 0.95rem; margin-top: 8px;'>{r['label']}</div>
                                <div style='font-size: 0.85rem; color: #555; margin-top: 4px;'>Độ tin cậy: <b>{r['confidence']*100:.1f}%</b></div>
                            </div>
                            """, unsafe_allow_html=True)

        elif mode == "Quét toàn bộ tiêu bản rộng":
            st.markdown("**Cài đặt thông số quét:**")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                conf_thresh = st.slider("Ngưỡng chẩn đoán Dương tính (%)", 50, 99, 50) / 100.0
            with col_opt2:
                min_cell_area = st.slider("Kích thước tế bào tối thiểu (Pixel)", 50, 1000, 200)

            if st.button("Thực hiện quét toàn bộ vi trường"):
                for uploaded_file in uploaded_files:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    input_image = cv2.imdecode(file_bytes, 1)
                    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

                    st.markdown(f"---\n**Ảnh:** `{uploaded_file.name}`")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(input_image, caption="Ảnh gốc", width=340)

                    with st.spinner("Đang phân đoạn & phân loại..."):
                        result_img, p_count, u_count = segment_and_classify(
                            input_image, model, conf_threshold=conf_thresh, min_area=min_cell_area
                        )

                    with col2:
                        st.image(result_img, caption="Ảnh đã chú thích", width=340)

                    st.divider()
                    with st.container():
                        st.subheader("Báo cáo thống kê")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Tổng số hồng cầu khảo sát", p_count + u_count)
                        c2.metric("Dương tính (Nhiễm Malaria)", p_count)
                        c3.metric("Âm tính (Khỏe mạnh)", u_count)

with tab_detail:
    st.markdown("#### Phân tích chi tiết từng bước thuật toán (Step-by-Step Pipeline)")
    detail_file = st.file_uploader(
        "Chọn một ảnh...",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False,
        key="detail_uploader",
    )

    if detail_file is not None and model is not None:
        file_bytes = np.asarray(bytearray(detail_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, 1)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        st.markdown("---")
        
        # Bước 1: Ảnh gốc
        st.markdown("##### Bước 1: Ảnh gốc đầu vào")
        st.image(input_image, width=400)
        
        # Bước 2: Chuyển xám và Lọc nhiễu
        st.markdown("##### Bước 2: Xử lý xám và Lọc nhiễu (Gaussian Blur)")
        gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        st.image(blurred, width=400, caption="Giúp loại bỏ nhiễu hạt, làm mịn ảnh trước khi phân đoạn")
        
        st.markdown("##### Bước 3: Phân đoạn ảnh (Otsu Thresholding & Morphology)")
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=2) 
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        st.image(thresh, width=400, caption="Tách biệt vùng chứa tế bào (màu trắng) khỏi nền (màu đen)")
        
        st.markdown("#####  Bước 4: Trích xuất và khoanh vùng")
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_img = input_image.copy()
        p_count = 0
        u_count = 0
        if len(contours) > 0:
            with st.spinner("Đang tải kết quả..."):
                for i, cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    if area < 200: 
                        continue
                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = input_image[y:y+h, x:x+w]
                    equalized = manual_histogram_equalization(roi)
                    resized = manual_resize(equalized, (128, 128))
                    normalized = resized.astype("float32") / 255.0
                    input_tensor = np.expand_dims(normalized, axis=0)
                    pred = model.predict(input_tensor, verbose=0)
                    score = float(pred[0][0])
                
                    if score > 0.5:
                        color = (255, 0, 0)
                        label = f"M: {score*100:.1f}%"
                        p_count += 1
                    else:
                        color = (0, 255, 0)
                        label = f"N: {(1-score)*100:.1f}%"
                        u_count += 1
                        
                    cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
                    text_y = y - 8 if y - 8 > 15 else y + h + 20
                    cv2.putText(output_img, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            st.image(output_img, caption="Kết quả khoanh vùng và phân loại cuối cùng", width=600)
            st.info(f"Phát hiện **{p_count}** tế bào nhiễm bệnh và **{u_count}** tế bào khỏe mạnh.")
            
        else:
            st.warning("Không tìm thấy tế bào nào có kích thước phù hợp trong ảnh!")

with tab_info:
    st.markdown("#### Giới thiệu hệ thống")
    st.write("- Ứng dụng hỗ trợ phân loại tế bào sốt rét trên ảnh lam máu.")
    st.write("- Mỗi tế bào được chuẩn hóa kích thước, cân bằng histogram và đưa vào mô hình CNN nhị phân (Nhiễm / Không nhiễm).")
    st.write("- Kết quả chỉ mang tính hỗ trợ, không thay thế chẩn đoán của bác sĩ chuyên khoa.")

    if model is not None:
        st.markdown("#### Thông tin mô hình")
        try:
            st.write(f"Số lượng tham số: {model.count_params():,}")
        except Exception:
            pass