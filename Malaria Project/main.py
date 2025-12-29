import os
import numpy as np
from sklearn.model_selection import train_test_split
from modules.data_loader import load_dataset_by_path
from modules.model import build_model

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, 'dataset')
parasitized_dir = os.path.join(base_dir, 'Parasitized')
uninfected_dir = os.path.join(base_dir, 'Uninfected')

X, y = load_dataset_by_path(parasitized_dir, uninfected_dir, limit=500)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}") 
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")

print(f"Memory Used: {X.nbytes / (1024 * 1024):.2f} MB")

model = build_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,           
    batch_size=32,         
    validation_data=(X_test, y_test), 
    verbose=1
)

model.save('malaria_model.h5')
