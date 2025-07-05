import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Load the model and dependencies
behavior_model = load_model("mobilenetv2_fer2013.h5")
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    """Preprocess face image for emotion prediction"""
    face_img = cv2.resize(face_img, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(face_img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# -------- Filepath based emotion detection (for uploaded images) --------
def detect_emotion(image_path: str):
    """
    Detect emotion from a saved image (uploaded via Gradio).
    """
    try:
        if not os.path.isfile(image_path):
            return "Error: Invalid image path."

        frame = cv2.imread(image_path)
        if frame is None:
            return "Error: Image could not be read."

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            input_data = preprocess_face(face_img)
            predictions = behavior_model.predict(input_data)
            return class_names[np.argmax(predictions)]

        return "No face detected"

    except Exception as e:
        return f"Error: {str(e)}"

# -------- Frame-based emotion detection (for webcam) --------
def detect_emotion_from_frame(frame):
    """
    Detect emotion directly from webcam frame (numpy array).
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            input_data = preprocess_face(face_img)
            predictions = behavior_model.predict(input_data)
            return class_names[np.argmax(predictions)], frame

        return "No face detected", frame

    except Exception as e:
        return f"Error: {str(e)}", frame

# -------- Local Webcam Feed: Run this ONLY locally --------
def start_webcam_emotion_detection():
    """
    Start live webcam feed and print detected emotion.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    print("[INFO] Starting continuous emotion detection via webcam...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame.")
            break

        emotion, frame_with_box = detect_emotion_from_frame(frame)
        print(f"[Emotion Detection] Current emotion: {emotion}")

        cv2.imshow("Live Emotion Detection", frame_with_box)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------- Optional Intent Router --------
import mimetypes

def detect_intent(input_path: str, text_input: str = None):
    """
    Determines the type of input based on file type and text presence.
    """
    if text_input and not input_path:
        return {"intent": "text", "text_input": text_input.strip()}

    if input_path:
        file_type, _ = mimetypes.guess_type(input_path)
        if file_type:
            if file_type.startswith("audio"):
                return {"intent": "audio", "input_path": input_path}
            elif file_type.startswith("image") and text_input:
                return {"intent": "image", "input_path": input_path, "text_input": text_input.strip()}

    return {"intent": "unknown"}

# -------- Run emotion analysis locally only (not in app.py) --------
if __name__ == "__main__":
    start_webcam_emotion_detection()
