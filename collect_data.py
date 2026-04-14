import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import mediapipe as mp
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

# --- CONFIGURATION ---
DATA_PATH = os.path.join('MP_Data')

# Get action name from user input
action = input("Enter the Sign Name (e.g., Assalam-o-Alaikum): ").strip()
if not action:
    print("Error: Sign name cannot be empty.")
    exit()

# Number of sequences (videos) per action
no_sequences = 30
# Each sequence is 30 frames long
sequence_length = 30

# Create directories for the specific action
for sequence in range(no_sequences):
    try:
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
    except FileExistsError:
        pass

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

# --- DATA COLLECTION LOOP ---
cap = cv2.VideoCapture(0)
# Set mediapipe model with max_num_hands=2
with mp_hands.Hands(
    model_complexity=0, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    max_num_hands=2
) as hands:
    
    print(f"Starting collection for: {action}")
    
    for sequence in range(no_sequences):
        for frame_num in range(sequence_length):

            ret, frame = cap.read()
            if not ret: break

            image, results = mediapipe_detection(frame, hands)
            draw_landmarks(image, results)
            
            if frame_num == 0:
                cv2.putText(image, 'STARTING COLLECTION', (120,200),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Action: {} | Video: {}'.format(action, sequence), (15,20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(2000) 
            else:
                cv2.putText(image, 'Action: {} | Video: {}'.format(action, sequence), (15,20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
            
            # Export keypoints (126 values)
            keypoints = extract_keypoints(results)
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished collecting 30 sequences for '{action}'.")
