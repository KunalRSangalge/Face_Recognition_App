import cv2
import mediapipe as mp
import numpy as np
##uninstalled mediapipe because of conflict with deepface

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# Load image
img = cv2.imread("Photos/test_face.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = face_mesh.process(img_rgb)

if not results.multi_face_landmarks:
    print("No face detected")
    exit()

landmarks = results.multi_face_landmarks[0]

# Convert landmarks to embedding vector
embedding = []
for lm in landmarks.landmark:
    embedding.extend([lm.x, lm.y, lm.z])

embedding = np.array(embedding)

print("Face embedding generated!")
print("Embedding shape:", embedding.shape)
print("First 10 values:", embedding[:10])
