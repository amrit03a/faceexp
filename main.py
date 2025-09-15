import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load model
model = load_model("best_model.h5")
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Real-Time Face Expression Detection")

run = st.checkbox("Run Camera")
FRAME_WINDOW = st.image([])

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("No frame captured from camera")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi_color, (48, 48)) / 255.0
        roi_reshaped = np.expand_dims(roi_resized, axis=0)
        pred = model.predict(roi_reshaped)
        label = class_labels[np.argmax(pred)]
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame_rgb)

cap.release()
