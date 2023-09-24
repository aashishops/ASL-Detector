# Sign Language Detector 🤙☝️👆👌🤞
[![DOCS](https://img.shields.io/badge/Documentation-see%20docs-green?style=flat-square&logo=appveyor)](https://github.com/aashishops/ASL-Detector/blob/main/README.Md) 
[![UI ](https://img.shields.io/badge/User%20Interface-Link%20to%20UI-orange?style=flat-square&logo=appveyor)](https://asl-detector.streamlit.app/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://asl-detector.streamlit.app/)

## Introduction
#### Welcome to the American Sign Language Detector!
The ASL Sign Language Detector is a powerful tool designed to recognize and interpret American Sign Language gestures.In this project, we generated hand landmarks using Mediapipe. Subsequently, we trained the model using the Support Vector Machine (SVM) algorithm provided by scikit-learn. We Used this [Kaggle Dataset](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) to train the model

## Key Features
- **Real-time Recognition**: The detector is capable of recognizing ASL signs in real-time, making it suitable for live interactions.
- **User-Friendly Interface**: The application comes with a user-friendly interface, making it accessible to users of all backgrounds.





## Demo

 [Click here <img src="https://raw.githubusercontent.com/aashishops/ASL-Detector/main/images/streamlit-logo-1A3B208AE4-seeklogo.com.png" alt="Streamlit Logo" width="25">](https://asl-detector.streamlit.app/) to Detect the Sign Language



- [Open the Link](https://asl-detector.streamlit.app/)
- Click the Start Button to turn on the camera and ensure there is enough light
![To Start](https://raw.githubusercontent.com/aashishops/ASL-Detector/main/images/ezgif-1-d9627fd4d4.gif)
 
 - Show some ASL Signs For it to Detect

 ![To Detect](https://raw.githubusercontent.com/aashishops/ASL-Detector/main/images/ezgif-1-8c2f960ba0.gif)

 - Click Stop to Turn off the Camera.

## Run Locally

Clone the project

```bash
  git clone https://github.com/aashishops/ASL-Detector
```

Go to the project directory

```bash
  cd ASL-Detector
```

Install Prerequisits

```bash
  pip install -r requirements.txt 
```

Run The app on Streamlit

```bash
  streamlit run app.py
```
OR
Run The app on OpenCV

```bash
  python asl_gesture.py
```


## How it Detects
 - **Hand Tracking**: It employs hand tracking to locate and track the user's hand gestures.

- **Gesture Recognition**: A machine learning model trained on a dataset of ASL signs recognizes and interprets the gestures


## How it was done 
### - Landmark Generation :
The landmarks were generated using the Mediapipe library.
![landmarks](https://raw.githubusercontent.com/aashishops/ASL-Detector/main/images/image.png)

These landmark coordinates are collected as dataset in [Dataset.csv](https://github.com/aashishops/ASL-Detector/blob/main/dataset.csv) for training and classification.


The Dataset was trained on Support Vector Machine Model using Scikit learn.
## License


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://raw.githubusercontent.com/aashishops/ASL-Detector/main/LICENSE)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://badges.mit-license.org/)
