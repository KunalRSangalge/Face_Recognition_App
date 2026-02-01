from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_embedding(img_path):
    result = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet",
        enforce_detection=True
    )
    return np.array(result[0]["embedding"])

emb1 = get_embedding("Photos/test_face_1.jpg")
emb2 = get_embedding("Photos/test_face2.jpg")
emb_other = get_embedding("Photos/test_face_other.jpg")

sim_same = cosine_similarity([emb1], [emb2])[0][0]
sim_diff = cosine_similarity([emb1], [emb_other])[0][0]

print("Similarity (same person):", sim_same)
print("Similarity (different person):", sim_diff)
