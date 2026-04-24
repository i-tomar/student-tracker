import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pickle

print("Loading dataset...")
df = pd.read_csv("ml_dataset.csv")

# Map text labels to integers
label_mapping = {"Focused": 0, "Distracted": 1, "Sleeping": 2}
df['label_encoded'] = df['label'].map(label_mapping)

# Features: EAR, Pitch, Yaw, Roll
X = df[['ear', 'pitch', 'yaw', 'roll']].values
y = df['label_encoded'].values

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples.")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save the scaler so we can use it during real-time inference
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Saved feature scaler to scaler.pkl")

# Build the Neural Network Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 classes: Focused, Distracted, Sleeping
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

print("\nTraining Model...")
# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=25,
    batch_size=32,
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nModel Validation Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save("study_companion.keras")
print("\nModel saved as study_companion.keras! You are ready to plug it into the app.")
