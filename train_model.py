import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data')

# Dynamically get actions from folders in MP_Data
# Sort them to ensure consistent ordering across all scripts
actions = np.array(sorted([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]))
print(f"Detected {len(actions)} actions: {actions}")

# Save actions to a file for use in other scripts
np.save('actions.npy', actions)
print("Actions list saved as actions.npy")

no_sequences = 30
sequence_length = 30

# Map labels to numbers
label_map = {label:num for num, label in enumerate(actions)}

# --- LOAD DATA ---
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            img_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            if os.path.exists(img_path):
                res = np.load(img_path)
                window.append(res)
            else:
                # Handle missing frames (e.g., if data collection was incomplete)
                print(f"Warning: {img_path} not found. Skipping frame.")
                window.append(np.zeros(126)) # Append zeros to maintain shape
        
        if len(window) == sequence_length:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- BUILD MODEL ---
model = Sequential()
# Input shape is now (30, 126) for two hands
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- TRAIN MODEL ---
print("Training model with multi-hand support...")
model.fit(X_train, y_train, epochs=200)

model.summary()

# --- SAVE MODEL ---
model.save('psl_model.h5')
print("Model saved as psl_model.h5")
