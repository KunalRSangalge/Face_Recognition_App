from deepface import DeepFace
import cv2

img_path = "experiments/Photos/kunal_face.jpg"

print("Running DeepFace on image path...")

result = DeepFace.represent(
    img_path=img_path,
    model_name="Facenet",
    detector_backend="opencv",
    enforce_detection=True
)

print("SUCCESS")
print("Embedding length:", len(result[0]["embedding"]))
