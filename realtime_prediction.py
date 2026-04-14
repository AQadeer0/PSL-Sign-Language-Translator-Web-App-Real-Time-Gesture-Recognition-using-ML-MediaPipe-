import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf

# --- CONFIGURATION ---
# --- CONFIGURATION ---
# Load actions from file if exists, otherwise detect from folders
if os.path.exists('actions.npy'):
    actions = np.load('actions.npy')
    print(f"Loaded {len(actions)} actions from actions.npy")
else:
    DATA_PATH = os.path.join('MP_Data')
    actions = np.array(sorted([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]))
    print(f"Detected {len(actions)} actions from folders")
model = load_model('psl_model.h5')

# --- SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_keypoints(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label
            landmarks = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if hand_label == 'Left':
                lh = landmarks
            else:
                rh = landmarks
    return np.concatenate([lh, rh])

# --- REAL-TIME LOOP ---
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image, results = mediapipe_detection(frame, hands)
        draw_landmarks(image, results)
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            # Optimized prediction: using model() instead of model.predict() is faster
            input_data = np.expand_dims(sequence, axis=0)
            input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
            res = model(input_tensor, training=False).numpy()[0]
            
            predicted_action = actions[np.argmax(res)]
            predictions.append(np.argmax(res))
            
            if np.unique(predictions[-10:])[0] == np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if predicted_action != sentence[-1]:
                            sentence.append(predicted_action)
                    else:
                        sentence.append(predicted_action)

            if len(sentence) > 5: 
                sentence = sentence[-5:]

        # UI Overlay
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
