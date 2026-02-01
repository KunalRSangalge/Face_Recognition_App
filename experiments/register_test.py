import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(BASE_DIR, "Photos")

from deepface import DeepFace
import numpy as np
from utils.storage import load_database, save_database

def get_embedding(img_path):
    result = DeepFace.represent(
        img_path=img_path,
        model_name="Facenet",
        enforce_detection=True
    )
    return np.array(result[0]["embedding"])

# Load existing database
embeddings_db, names_db = load_database()

# Register a new user
person_name = "Kunal"  
image_paths = [
    os.path.join(PHOTOS_DIR, "test_face_1.jpg"),
    os.path.join(PHOTOS_DIR, "test_face2.jpg")
]

new_embeddings = []

for img_path in image_paths:
    emb = get_embedding(img_path)
    new_embeddings.append(emb)

new_embeddings = np.array(new_embeddings)

# Append to database
embeddings_db = np.vstack([embeddings_db, new_embeddings])
names_db = np.concatenate([names_db, [person_name] * len(new_embeddings)])

# Save
save_database(embeddings_db, names_db)

print(f"Registered {person_name} with {len(new_embeddings)} embeddings")
print("Total embeddings in DB:", len(names_db))
