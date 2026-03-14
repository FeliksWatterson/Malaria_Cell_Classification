import numpy as np
import cv2
from PIL import Image 

def manual_histogram_equalization(image_array):
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]
    
    histogram = np.bincount(v_channel.flatten(), minlength=256)
    cdf = histogram.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
    cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')
    
    hsv_image[:, :, 2] = cdf_final[v_channel]
    equalized_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    return equalized_rgb

def manual_resize(image_array, target_size=(128, 128)):
    target_h, target_w = target_size
    orig_h, orig_w, channels = image_array.shape
    
    x_ratio = float(orig_w - 1) / (target_w - 1) if target_w > 1 else 0
    y_ratio = float(orig_h - 1) / (target_h - 1) if target_h > 1 else 0
    
    y_indices = np.arange(target_h) * y_ratio
    x_indices = np.arange(target_w) * x_ratio
    
    y_floor = np.floor(y_indices).astype(int)
    y_ceil = np.minimum(y_floor + 1, orig_h - 1)
    dy = y_indices - y_floor
    
    x_floor = np.floor(x_indices).astype(int)
    x_ceil = np.minimum(x_floor + 1, orig_w - 1)
    dx = x_indices - x_floor
    
    X_floor, Y_floor = np.meshgrid(x_floor, y_floor)
    X_ceil, Y_ceil = np.meshgrid(x_ceil, y_ceil)
    DX, DY = np.meshgrid(dx, dy)
    
    w1 = (1 - DX) * (1 - DY)
    w2 = DX * (1 - DY)
    w3 = (1 - DX) * DY
    w4 = DX * DY
    
    p1 = image_array[Y_floor, X_floor]
    p2 = image_array[Y_floor, X_ceil]
    p3 = image_array[Y_ceil, X_floor]
    p4 = image_array[Y_ceil, X_ceil]
    
    resized_image = np.zeros((target_h, target_w, channels), dtype=np.float32)
    for c in range(channels):
        resized_image[:, :, c] = (p1[:, :, c] * w1 + 
                                  p2[:, :, c] * w2 + 
                                  p3[:, :, c] * w3 + 
                                  p4[:, :, c] * w4)
                                  
    return resized_image.astype(image_array.dtype)

def load_and_preprocess(image_path, target_size=(128, 128)):
    try:
        img_pil = Image.open(image_path).convert('RGB')
        img_array = np.array(img_pil)
        img_equalized = manual_histogram_equalization(img_array)
        img_resized = manual_resize(img_equalized, target_size)
        img_normalized = img_resized.astype('float32') / 255.0
        return img_normalized
        
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None