import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import os
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # only errors

#Load pre-trained emotion detection model
MODEL_PATH = 'emotion_model.h5'
emotion_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


#preprocess and detect emotion 
def detect_emotion(frame, emotion_model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_model.predict(roi, verbose=0)
        probs = preds[0]
        top_idx = int(np.argmax(probs))
        emotion = emotion_labels[top_idx]
        confidence = float(probs[top_idx])
        prob_dict = {label: float(prob) for label, prob in zip(emotion_labels, probs)}
        results.append((x, y, w, h, emotion, confidence, prob_dict))
    return results


def style_page():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

            .stApp {
                background:
                    radial-gradient(1400px 900px at 10% -10%, rgba(0, 170, 255, 0.12), transparent),
                    radial-gradient(1200px 800px at 100% 0%, rgba(255, 140, 66, 0.10), transparent),
                    linear-gradient(180deg, #f8fbff 0%, #eef4f9 100%);
                color: #123;
                font-family: 'Manrope', sans-serif;
            }

            h1, h2, h3 {
                font-family: 'Space Grotesk', sans-serif;
                letter-spacing: -0.02em;
            }

            .hero {
                border-radius: 18px;
                padding: 28px 28px 18px 28px;
                background: linear-gradient(135deg, rgba(10, 38, 71, 0.94), rgba(16, 98, 141, 0.88));
                color: #f4fbff;
                box-shadow: 0 16px 32px rgba(11, 42, 75, 0.2);
                margin-bottom: 1rem;
            }

            .hero-title {
                font-size: 2rem;
                font-weight: 800;
                margin-bottom: 0.35rem;
                color: #f4fbff;
            }

            .hero-sub {
                font-size: 1rem;
                opacity: 0.95;
                line-height: 1.6;
                margin: 0;
            }

            .panel {
                background: #ffffff;
                border: 1px solid #d9e4f1;
                border-radius: 14px;
                padding: 14px 16px;
                box-shadow: 0 8px 24px rgba(10, 43, 75, 0.07);
            }

            .note {
                background: #f5fbff;
                border: 1px solid #cde7ff;
                border-left: 4px solid #0090d0;
                border-radius: 12px;
                padding: 12px 14px;
                color: #1a3d58;
                font-size: 0.97rem;
                line-height: 1.5;
            }

            .caption {
                font-size: 0.9rem;
                color: #375777;
                margin-top: 0.2rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


#Streamlit App 
def main():
    st.set_page_config(page_title="Real-Time Emotion Detector", page_icon="😊", layout="wide")
    style_page()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">EmotionLens: Real-Time Face Emotion Detection</div>
            <p class="hero-sub">
                Clean live emotion detection with explainable confidence scores.<br/>
                See what the model predicts, how sure it is, and recent trend behavior in one dashboard.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.8, 1.2], gap="large")

    with left_col:
        st.markdown("### Live Camera")
        run = st.toggle("Start webcam", value=False)
        frame_window = st.image([])

        guide_col_1, guide_col_2 = st.columns(2)
        with guide_col_1:
            st.markdown(
                """
                <div class="panel">
                    <strong>How to use</strong>
                    <div class="caption">1) Start webcam 2) Face camera 3) Check confidence + trends</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with guide_col_2:
            st.markdown(
                """
                <div class="panel">
                    <strong>Best quality tips</strong>
                    <div class="caption">Good lighting, centered face, minimal motion improves stability.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("### Explainability")
        status_box = st.empty()
        metric_col_1, metric_col_2 = st.columns(2)
        faces_metric = metric_col_1.empty()
        confidence_metric = metric_col_2.empty()
        st.markdown("#### Emotion confidence distribution")
        bars_placeholder = st.empty()
        st.markdown(
            """
            <div class="note">
                The model outputs probabilities for all emotions. Higher confidence means stronger model certainty,
                not guaranteed correctness.
            </div>
            """,
            unsafe_allow_html=True,
        )

    cap = None
    emotion_model = None

    if run:
        with st.spinner("Loading emotion model..."):
            emotion_model = load_emotion_model()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam. Check permissions and try again.")
            return

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read frame from webcam.")
            break

        results = detect_emotion(frame)

        primary_probs = {label: 0.0 for label in emotion_labels}
        primary_emotion = "No face"
        primary_confidence = 0.0

        if results:
            # Choose the highest-confidence face as the main explanation target.
            strongest = max(results, key=lambda item: item[5])
            primary_emotion = strongest[4]
            primary_confidence = strongest[5]
            primary_probs = strongest[6]

        for (x, y, w, h, emotion, confidence, _) in results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (5, 112, 197), 2)
            cv2.putText(
                frame,
                f"{emotion} ({confidence * 100:.0f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (12, 194, 130),
                2,
            )

        frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        status_box.info(f"Current detected emotion: {primary_emotion}")
        faces_metric.metric("Faces detected", len(results))
        confidence_metric.metric("Top confidence", f"{primary_confidence * 100:.1f}%")

        with bars_placeholder.container():
            for label in emotion_labels:
                prob = primary_probs.get(label, 0.0)
                st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

        # (Trend chart removed for simplified demo)

    if cap is not None:
        cap.release()

    if not run:
        status_box.info("Webcam is stopped. Enable Start webcam to begin live analysis.")
        faces_metric.metric("Faces detected", 0)
        confidence_metric.metric("Top confidence", "0.0%")
        with bars_placeholder.container():
            for label in emotion_labels:
                st.progress(0.0, text=f"{label}: 0.0%")


if __name__ == '__main__':
    main()
