import numpy as np
from tensorflow.keras.models import load_model
from modules.preprocessor import load_and_preprocess

model = load_model('Malaria Project\malaria_model.h5')
print("Model Loader Successfully")

def make_prediction(image_path):
    img = load_and_preprocess('Malaria Project\images\test.png')
    
    if img is None:
        print("No img detected")
        return

    img_batch = np.expand_dims(img, axis=0) 

    prediction = model.predict(img_batch)
    score = prediction[0][0]

    if score > 0.5:
        label = "Parasitized (NHIỄM BỆNH)"
        confidence = score
    else:
        label = "Uninfected (KHÔNG BỆNH)"
        confidence = 1 - score

    print(f"\n--- KẾT QUẢ ---")
    print(f"File: {image_path}")
    print(f"Dự đoán: {label}")
    print(f"Độ tin cậy: {confidence * 100:.2f}%")

make_prediction('test.png') 