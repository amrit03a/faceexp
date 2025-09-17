import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("best_model.h5")
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def predict_expression(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi_color, (48, 48)) / 255.0
        roi_reshaped = np.expand_dims(roi_resized, axis=0)
        pred = model.predict(roi_reshaped)
        label = class_labels[np.argmax(pred)]
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return frame

demo = gr.Interface(
    fn=predict_expression,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="image"
)

demo.launch()
