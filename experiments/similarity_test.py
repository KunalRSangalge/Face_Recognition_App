import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


##uninstalled mediapipe because of conflict with deepface

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        raise ValueError(f"No face detected in {image_path}")

    landmarks = results.multi_face_landmarks[0]
    embedding = []

    for lm in landmarks.landmark:
        embedding.extend([lm.x, lm.y, lm.z])

    return np.array(embedding)

# Load embeddings
emb_1 = get_embedding("Photos/test_face_1.jpg")
emb_2 = get_embedding("Photos/test_face2.jpg")
emb_other = get_embedding("Photos/test_face_other.jpg")

# Compute cosine similarity
sim_same = cosine_similarity([emb_1], [emb_2])[0][0]
sim_diff = cosine_similarity([emb_1], [emb_other])[0][0]

print("Cosine similarity (same person):", sim_same)
print("Cosine similarity (different person):", sim_diff)
