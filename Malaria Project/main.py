import os
import numpy as np
from sklearn.model_selection import train_test_split
from modules.data_loader import load_dataset_by_path

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, 'dataset')
parasitized_dir = os.path.join(base_dir, 'Parasitized')
uninfected_dir = os.path.join(base_dir, 'Uninfected')

X, y = load_dataset_by_path(parasitized_dir, uninfected_dir, limit=500)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}") 
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")

print(f"Dung lượng bộ nhớ ước tính: {X.nbytes / (1024 * 1024):.2f} MB")