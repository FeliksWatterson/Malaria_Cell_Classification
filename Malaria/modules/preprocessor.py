import numpy as np
from PIL import Image 

def manual_histogram_equalization(image_array):
    equalized_image = np.zeros_like(image_array)
    for channel in range(3):
        channel_data = image_array[:, :, channel]
        histogram = np.bincount(channel_data.flatten(), minlength=256)
        cdf = histogram.cumsum()
        cdf_masked = np.ma.masked_equal(cdf, 0)
        cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
        cdf_final = np.ma.filled(cdf_normalized, 0).astype('uint8')
        equalized_image[:, :, channel] = cdf_final[channel_data]
        
    return equalized_image

def manual_resize(image_array, target_size=(128, 128)):
    target_h, target_w = target_size
    orig_h, orig_w, channels = image_array.shape
    
    y_ratio = orig_h / target_h
    x_ratio = orig_w / target_w
    
    y_indices = (np.arange(target_h) * y_ratio).astype(int)
    x_indices = (np.arange(target_w) * x_ratio).astype(int)
    
    resized_image = image_array[y_indices[:, None], x_indices]
    return resized_image

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