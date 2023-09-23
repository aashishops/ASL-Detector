import cv2
import mediapipe as mp
import numpy as np
import pickle
import streamlit as st
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.title("ASL Sign Language Detection")

# Initialize Hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
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

# Load your SVM model
with open('asl_model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Create a placeholder for displaying the webcam video
video_placeholder = st.empty()

# Create a flag to stop the webcam feed
stop_webcam = False

# Create a flag to skip some frames
skip_frames = 0
skip_frequency = 5  # Adjust this based on your desired frame rate

# Variables for calculating FPS
start_time = time.time()
frame_count = 0

while not stop_webcam:
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Cannot open camera")
        break

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error("Can't receive frame (stream end?). Exiting ...")
            break

        skip_frames += 1
        if skip_frames % skip_frequency != 0:
            continue  # Skip this frame

        frame_count += 1

        # Image processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_flip = cv2.flip(img_rgb, 1)
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
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=0))

        # Display the prediction on the Streamlit app
        data = image_processed(frame)
        data = np.array(data)
        y_pred = svm.predict(data.reshape(-1, 63))
        color = (255, 255, 255)
        thickness = 3
        org = (2,50)
        fontScale = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(frame, (0,0), (230,70), (245,117,16), -1)
        frame = cv2.putText(frame, str(y_pred[0]), org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

        # Calculate and display FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        cv2.putText(frame, f"FPS: {int(fps)}", (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (245,117,16), 2)

        # Display the webcam video
        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        # Check if the stop button is pressed
       
# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
