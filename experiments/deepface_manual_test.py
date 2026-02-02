import cv2
import numpy as np
from deepface import DeepFace

img = cv2.imread("experiments/Photos/kunal_face.jpg")

# Haar face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

print("Faces detected:", len(faces))

x, y, w, h = faces[0]
face = img[y:y+h, x:x+w]

# Proper preprocessing
face = cv2.resize(face, (160, 160))
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
face = face.astype("uint8")

print("Face shape:", face.shape, face.dtype)

result = DeepFace.represent(
    img_path=face,
    model_name="Facenet",
    detector_backend="skip",
    enforce_detection=False
)

print("SUCCESS")
print("Embedding length:", len(result[0]["embedding"]))
