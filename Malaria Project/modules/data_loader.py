import os
import numpy as np
from tqdm import tqdm
from modules.preprocessor import load_and_preprocess

def load_dataset_by_path(parasitized_dir, uninfected_dir, limit=None):
    images = []
    labels = []
    
    directories = {
        0: uninfected_dir,
        1: parasitized_dir
    }
    
    for label, folder_path in directories.items():
        print(f"{os.path.basename(folder_path)} (Nhãn: {label})")
        
        file_names = os.listdir(folder_path)
        
        if limit:
            file_names = file_names[:limit]
            
        for file_name in tqdm(file_names):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder_path, file_name)
                
                processed_img = load_and_preprocess(full_path)
                
                if processed_img is not None:
                    images.append(processed_img)
                    labels.append(label)
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"Tổng cộng: {len(X)} ảnh.")
    return X, y