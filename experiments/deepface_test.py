from deepface import DeepFace

result = DeepFace.represent(
    img_path="Photos/test_face_1.jpg",
    model_name="Facenet",
    enforce_detection=True
)

embedding = result[0]["embedding"]

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])
