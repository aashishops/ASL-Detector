import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoTransformerBase, webrtc_streamer
#
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandTrackingTransformer(VideoTransformerBase):
    def __init__(self):
        self.prev_time = 0
        with open('asl_model.pkl', 'rb') as f:
            self.svm = pickle.load(f)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Flip the frame horizontally to correct the mirroring issue
        image = cv2.flip(image, 1)

        # Image processing
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize Hands
        hands = mp_hands.Hands(static_image_mode=True,
                               max_num_hands=1, min_detection_confidence=0.7)

        # Results
        output = hands.process(img_rgb)
        hands.close()

        try:
            data = output.multi_hand_landmarks[0]

            # Calculate bounding box directly from hand landmarks
            x_min, y_min, x_max, y_max = self.get_bounding_box(data, img_rgb.shape[1], img_rgb.shape[0])

            # Draw bounding box
            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, data, mp_hands.HAND_CONNECTIONS)

            # ASL sign prediction
            data = self.extract_landmark_data(data)
            y_pred = self.svm.predict(data.reshape(-1, 63))

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time

            # Display FPS and prediction
            cv2.rectangle(image, (0, 0), (230, 70), (245, 117, 16), -1)
            image = cv2.putText(image, str(y_pred[0]), (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, f"FPS: {int(fps)}", (500, 76), cv2.FONT_HERSHEY_SIMPLEX, 1, (245, 117, 16), 2)

        except:
            pass

        return image

    def get_bounding_box(self, landmarks, width, height):
        x_min, x_max, y_min, y_max = width, 0, height, 0

        for landmark in landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        return x_min, y_min, x_max, y_max

    def extract_landmark_data(self, landmarks):
        data = []
        for landmark in landmarks.landmark:
            data.extend([landmark.x, landmark.y, landmark.z])
        return np.array(data)

def main():
    st.title("ASL Sign Language Detection with Hand Tracking")

    webrtc_ctx = webrtc_streamer(
        key="hand-tracking",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=HandTrackingTransformer,
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.write("ASL Sign Language Detection is running.")
        st.write("Press 'Stop' to stop the camera.")
        

if __name__ == "__main__":
    main()
