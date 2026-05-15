import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Setup CSV
DATA_FILE = 'landmarks.csv'
labels = ['A', 'B', 'C'] # Classes to collect
print(f"Ready to collect data for: {labels}")
print("Press 'A', 'B', or 'C' on your keyboard to capture a frame for that class.")
print("Press 'Q' to quit.")

cap = cv2.VideoCapture(0)
data = []

# If file exists, load it to append
if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE)
    data = df.values.tolist()
    print(f"Loaded existing data with {len(data)} rows.")

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        landmarks_row = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                for landmark in hand_landmarks.landmark:
                    landmarks_row.extend([landmark.x, landmark.y, landmark.z])

        cv2.putText(image, f"Samples: {len(data)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('MediaPipe Hands - Data Collection', image)

        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('q'):
            break
        elif key in [ord('a'), ord('b'), ord('c')]:
            if len(landmarks_row) == 63: # 21 landmarks * 3 coordinates
                char = chr(key).upper()
                row = [char] + landmarks_row
                data.append(row)
                print(f"Captured {char}. Total {char} samples: {sum(1 for d in data if d[0] == char)}")
                # Visual feedback
                cv2.rectangle(image, (0,0), (640, 480), (0, 255, 0), 20)
                cv2.imshow('MediaPipe Hands - Data Collection', image)
                cv2.waitKey(100) # pause briefly
            else:
                print("Hand not detected clearly. Try again.")

finally:
    # Save to CSV
    if len(data) > 0:
        # Create column names
        cols = ['label']
        for i in range(21):
            cols.extend([f'x{i}', f'y{i}', f'z{i}'])
        
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(DATA_FILE, index=False)
        print(f"Saved {len(df)} rows to {DATA_FILE}")
        
    cap.release()
    cv2.destroyAllWindows()
