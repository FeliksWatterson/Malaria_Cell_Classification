import cv2
import numpy as np

def load_and_preprocess(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, target_size)
    
    img = img.astype('float32') / 255.0
    
    return img