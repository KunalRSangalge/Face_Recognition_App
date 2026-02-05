import streamlit as st
import cv2
import numpy as np
import os
import time
from utils.storage import save_database,load_database,reset_database,delete_person
import uuid

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def normalize_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# def extract_embedding_from_file(img_path):
#     from deepface import DeepFace
#     import numpy as np

#     result = DeepFace.represent(
#         img_path=img_path,
#         model_name="Facenet",
#         detector_backend="opencv",
#         enforce_detection=True
#     )
#     return np.array(result[0]["embedding"])

def extract_embedding_from_file(img_path):
    from deepface import DeepFace
    import cv2
    import numpy as np

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        raise ValueError("No face detected")

    # take largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("uint8")

    result = DeepFace.represent(
        img_path=face,
        model_name="Facenet",
        detector_backend="skip",
        enforce_detection=False
    )

    return np.array(result[0]["embedding"])



st.set_page_config(page_title="Face Recognition App", layout="centered")
st.title("🧠 Real-Time Face Recognition App")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Home",  "Register New User", "Delete User or Database", "Live Recognition"]
)

if mode == "Home":
    st.success("Select a mode from the sidebar to get started.")

elif mode == "Delete User or Database":
    embeddings_db, names_db = load_database()
    unique_names = sorted(set(names_db))

    person_to_delete = st.selectbox("Select person to delete", unique_names)

    if st.button("Delete Person"):
        delete_person(person_to_delete)
        st.success(f"Deleted {person_to_delete} from database.")


    if st.button("⚠️ Delete Entire Database"):
        reset_database()
        st.success("Database cleared successfully.")

elif mode == "Register New User":
    st.subheader("📝 Register New User")

    person_name = st.text_input("Enter person's name")

    reg_mode = st.radio(
        "Choose registration mode",
        ["Live Registration (Webcam)", "Upload Images (Recommended)"]
    )

    if "reg_embeddings" not in st.session_state:
        st.session_state.reg_embeddings = []

    MAX_EMBEDDINGS = 10
    progress = st.progress(0)
    status = st.empty()

    # ---------- MODE 1: LIVE REGISTRATION ----------
    if reg_mode == "Live Registration (Webcam)":
        photo = st.camera_input("Capture face image")

        if photo is not None and person_name.strip() != "":
            # Save image to temp file
            os.makedirs("data/temp", exist_ok=True)
            temp_path = f"data/temp/{uuid.uuid4().hex}.jpg"

            with open(temp_path, "wb") as f:
                f.write(photo.getvalue())

            try:
                emb = extract_embedding_from_file(temp_path)
                st.session_state.reg_embeddings.append(emb)

                count = len(st.session_state.reg_embeddings)
                progress.progress(count / MAX_EMBEDDINGS)
                status.success(f"Captured {count} / {MAX_EMBEDDINGS}")

            except Exception:
                status.error("Face not detected properly. Try again.")

            finally:
                os.remove(temp_path)

    # ---------- MODE 2: UPLOAD REGISTRATION ----------
    elif reg_mode == "Upload Images (Recommended)":
        uploaded_files = st.file_uploader(
            "Upload 10–15 face images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True
        )

        if uploaded_files and person_name.strip() != "":
            for uploaded_file in uploaded_files:
                if len(st.session_state.reg_embeddings) >= MAX_EMBEDDINGS:
                    break

                os.makedirs("data/temp", exist_ok=True)
                temp_path = f"data/temp/{uuid.uuid4().hex}.jpg"

                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                try:
                    emb = extract_embedding_from_file(temp_path)
                    st.session_state.reg_embeddings.append(emb)

                except Exception:
                    pass

                finally:
                    os.remove(temp_path)

            count = len(st.session_state.reg_embeddings)
            progress.progress(count / MAX_EMBEDDINGS)
            status.success(f"Captured {count} / {MAX_EMBEDDINGS}")

    # ---------- SAVE TO DATABASE ----------
    if len(st.session_state.reg_embeddings) >= MAX_EMBEDDINGS:
        embeddings_db, names_db = load_database()

        new_embeddings = np.array(st.session_state.reg_embeddings)
        embeddings_db = np.vstack([embeddings_db, new_embeddings])
        names_db = np.concatenate(
            [names_db, [person_name] * len(new_embeddings)]
        )

        save_database(embeddings_db, names_db)
        st.session_state.reg_embeddings = []

        st.success(f"✅ Registered {person_name} successfully!")


elif mode == "Live Recognition":
    
    import cv2
    import time
    import numpy as np
    from deepface import DeepFace
    from utils.storage import load_database
    from utils.recognition import recognize,recognize_centroid,build_centroids
    from sklearn.metrics.pairwise import cosine_similarity
    st.subheader("📷 Live Face Detection + Recognition (DEBUG MODE)")

    start = st.button("Start Camera")
    stop = st.button("Stop Camera")

    frame_placeholder = st.empty()

    # Load database once
    embeddings_db, names_db = load_database()

    centroids = build_centroids(embeddings_db, names_db)
    # FPS tracking
    prev_time = 0
    fps = 0

    if start:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            # ---------------- FPS CALCULATION ----------------
            curr_time = time.time()
            if prev_time != 0:
                fps = 0.9 * fps + 0.1 * (1 / (curr_time - prev_time))
            prev_time = curr_time
            # -------------------------------------------------

            # ---------------- FACE DETECTION (EVERY FRAME) ----------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
                # 🔲 Draw bounding box
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # ✂️ Crop face
                face = frame[y:y+h, x:x+w]

                # Preprocess for FaceNet
                face = cv2.resize(face, (160, 160))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = face.astype("uint8")

                
                # embs = np.array(list(centroids.values()))
                # sims = cosine_similarity([emb], np.array(list(centroids.values())))[0]
                # for n, s in sorted(zip(centroids.keys(), sims), key=lambda x: -x[1]):
                #     print(n, round(s, 3))

                try:
                    reps = DeepFace.represent(
                        img_path=face,
                        model_name="Facenet",
                        detector_backend="skip",
                        enforce_detection=False
                    )

                    if reps:
                        emb = np.array(reps[0]["embedding"])
                        # name, score = recognize(
                        #     emb, embeddings_db, names_db
                        # )
                        name, score = recognize_centroid(emb, centroids)

                    else:
                        name, score = "Unknown", 0.0

                except Exception as e:
                    print("Embedding error:", e)
                    name, score = "Error", 0.0

                # 🏷️ Draw label
                cv2.putText(
                    frame,
                    f"{name} ({score:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # -------------------------------------------------

            fps_text = f"FPS: {int(fps)}"
            cv2.putText(
                frame,
                fps_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            if stop:
                break

        cap.release()


