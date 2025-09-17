import os

os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_CONFIG_DIR'] = '/tmp'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av


# Load model
@st.cache_resource
def load_emotion_model():
    try:
        model = tf.keras.models.load_model("best_model.h5", compile=False)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


model = load_emotion_model()
if model is None:
    st.stop()

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


class FaceExpressionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                roi_color = img[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi_color, (48, 48)) / 255.0
                roi_reshaped = np.expand_dims(roi_resized, axis=0)
                pred = model.predict(roi_reshaped)
                label = class_labels[np.argmax(pred)]
                cv2.putText(img, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            except Exception as e:
                # Continue processing even if one face fails
                pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("Real-Time Face Expression Detection")
    st.write("Click 'START' to begin detection. Please allow camera access when prompted.")

    webrtc_ctx = webrtc_streamer(
        key="face-expression",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceExpressionProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if not webrtc_ctx.state.playing:
        st.info("Click 'START' to begin detection.")


if __name__ == "__main__":
    main()