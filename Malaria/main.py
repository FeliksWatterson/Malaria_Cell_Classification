import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modules.data_loader import load_dataset_by_path
from modules.model import build_model

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, 'dataset')
parasitized_dir = os.path.join(base_dir, 'Parasitized')
uninfected_dir = os.path.join(base_dir, 'Uninfected')

print("Uploading data...")
X, y = load_dataset_by_path(parasitized_dir, uninfected_dir, limit=10000) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}") 
print(f"X_test:  {X_test.shape}")
print(f"y_train: {y_train.shape}")

print(f"Memory Used: {X.nbytes / (1024 * 1024):.2f} MB")

aug = ImageDataGenerator(
    rotation_range=20,    
    zoom_range=0.15,       
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    shear_range=0.15,       
    horizontal_flip=True,  
    fill_mode="nearest"
)

model = build_model()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

print("Training...")
save_path = os.path.join(current_dir, 'malaria_model.keras')

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=save_path, 
    monitor='val_accuracy', 
    save_best_only=True,    
    mode='max',
    verbose=1               
)

history = model.fit(
    aug.flow(X_train, y_train, batch_size=32),     
    validation_data=(X_test, y_test),
    epochs=30,                                
    callbacks=[early_stopping, model_checkpoint], 
    verbose=1
)

