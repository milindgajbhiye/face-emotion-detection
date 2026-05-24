# Tutorial: Face Emotion Detection (Updated)

This tutorial explains the **current version** of the project and includes the newly added UI, explainability, and performance improvements.

---

## 1) What this project does

This app detects facial emotions in real time from your webcam using:
- **OpenCV** for face detection
- **TensorFlow/Keras** for emotion prediction
- **Streamlit** for the web interface

Detected emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## 2) What’s newly added

The latest app version (`app.py`) includes:

- A refreshed dashboard-style UI with hero section and styled panels
- **Start webcam** / **Stop webcam** buttons (session-state based control)
- A dedicated **Explainability** panel
- Real-time **emotion confidence distribution** bars
- Live metrics:
  - Faces detected
  - Top confidence
- Better runtime behavior:
  - Cached model loading with `@st.cache_resource`
  - Background model warm-up thread for faster initial UI response
  - Frame skipping to reduce CPU usage
  - Webcam backend fallback (`cv2.CAP_DSHOW` then default)
  - Lower capture resolution for better performance

---

## 3) Project files you should know

- `app.py` → **Main and latest app**
- `MyFaceEmotion/FaceDetection.py` → Older/simple Streamlit version kept in repo
- `emotion_model.h5` → Model file used by `app.py`
- `requirements.txt` → Python dependencies
- `README.md` → Repository overview

> For current behavior and UI, use `app.py` as the reference.

---

## 4) Setup

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Webcam access enabled on your system

### Install dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

If needed, ensure Streamlit/OpenCV/TensorFlow are available in your environment.

### Model file

Make sure `emotion_model.h5` exists in repository root (same level as `app.py`).

---

## 5) Run the latest app

From repository root:

```bash
streamlit run app.py
```

Expected success output includes a line similar to:

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Then open the shown local URL in your browser.

---

## 6) How to use the app

1. Open the app in browser.
2. Click **Start webcam**.
3. Face the camera with good lighting.
4. Observe:
   - Face bounding boxes and top emotion on video
   - Current primary emotion card
   - Faces detected metric
   - Top confidence metric
   - Confidence bars for all emotion classes
5. Click **Stop webcam** to end capture.

---

## 7) How prediction works (current flow)

1. Frame is captured from webcam.
2. Frame is converted to grayscale.
3. Faces are detected using Haar cascade.
4. For each face, ROI is:
   - Cropped
   - Resized to `64x64`
   - Normalized to `[0,1]`
   - Expanded to model input dimensions
5. Model predicts probabilities for all emotion labels.
6. Highest-confidence emotion is shown as the primary prediction.

---

## 8) Notes on performance and behavior

- Inference runs every few frames (frame skip) to reduce CPU load.
- Last detection results are reused between inference frames.
- Model loading is cached and warmed up in background.
- Capture resolution is reduced to improve speed on regular laptops.
- If webcam fails to open with DirectShow backend, app retries default backend.

---

## 9) Troubleshooting

### Webcam not opening
- Check OS/browser/app camera permissions.
- Close other apps using the camera.
- Restart Streamlit app.

### Model loading error
- Ensure `emotion_model.h5` is present and valid.
- Confirm TensorFlow is installed correctly.

### Slow performance
- Close heavy background applications.
- Use better lighting and stable camera position.
- Consider running in a clean virtual environment.

---

## 10) Quick command summary

```bash
# install deps
pip install -r requirements.txt

# run current app
streamlit run app.py
```

---

This tutorial now reflects the latest implementation and newly added features in the repository.
