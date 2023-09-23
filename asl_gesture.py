import cv2 
import mediapipe as mp
import pandas as pd  
import numpy as np 
import pickle 
import time
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True,
                                       max_num_hands=1, min_detection_confidence=0.7)

prev_time = 0
fps = 0

def image_processed(hand_img):

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
    
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])


with open('asl_model.pkl', 'rb') as f:
    svm = pickle.load(f)

import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
i = 0    
while True:
    
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # frame = cv.flip(frame,1)
    data = image_processed(frame)
    
    # print(data.shape)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,63))
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (2,50)
    
    # fontScale
    fontScale = 2
    
    # Blue color in BGR
    color = (255, 255, 255)
    thickness = 3
    
   # Using cv2.putText() method
    cv2.rectangle(frame, (0,0), (230,70), (245,117,16), -1)
    frame = cv2.putText(frame, str(y_pred[0]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
# Display the FPS on the frame
    cv2.putText(frame, f"FPS: {int(fps)}", (230,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (245,117,16), 2)

    # Display the frame with OpenCV
    cv2.imshow('ASL Sign Language Detection', frame)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Initialize min and max coordinates for bounding box
            x_min, y_min = frame.shape[1], frame.shape[0]
            x_max, y_max = 0, 0

            for landmark in landmarks.landmark:
                x, y, _ = frame.shape[1] * landmark.x, frame.shape[0] * landmark.y, landmark.z
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)

            # Draw bounding box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(255,0, 0), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=0))
    
    cv2.imshow('Hand Tracking', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()