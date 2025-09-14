import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import os

# -------------------- Suppress TensorFlow info/warnings --------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors

# -------------------- Load pre-trained emotion detection model --------------------
MODEL_PATH = 'emotion_model.h5'
emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Emotion labels (adjust according to your model)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -------------------- Preprocess and detect emotion --------------------
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))  # <-- match model input
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=-1)  # channel
        roi = np.expand_dims(roi, axis=0)   # batch
        preds = emotion_model.predict(roi, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]
        results.append((x, y, w, h, emotion))
    return results

# -------------------- Streamlit App --------------------
def main():
    st.title("😊 Real-Time Face Emotion Detection")
    st.markdown("Detect emotions (Angry, Happy, Sad, etc.) in real-time from your webcam.")

    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read frame from webcam.")
            break

        results = detect_emotion(frame)
        for (x, y, w, h, emotion) in results:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

if __name__ == '__main__':
    main()
