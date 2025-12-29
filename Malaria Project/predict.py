import numpy as np
from tensorflow.keras.models import load_model
from modules.preprocessor import load_and_preprocess

model = load_model('malaria_model.h5')
print("Model Loader Successfully")

def make_prediction(image_path):
    img = load_and_preprocess(image_path)
    
    if img is None:
        print("No img detected")
        return

    img_batch = np.expand_dims(img, axis=0) 

    prediction = model.predict(img_batch)
    score = prediction[0][0]

    if score > 0.5:
        label = "Parasitized"
        confidence = score
    else:
        label = "Uninfected"
        confidence = 1 - score

    print(f"File: {image_path}")
    print(f"Result: {label}")
    print(f"Accuracy: {confidence * 100:.2f}%")

make_prediction('images/test.png') 