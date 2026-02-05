import numpy as np
import os

EMBEDDINGS_PATH = "data/embeddings.npy"
NAMES_PATH = "data/names.npy"

def load_database():
    if (
        os.path.exists(EMBEDDINGS_PATH)
        and os.path.exists(NAMES_PATH)
        and os.path.getsize(EMBEDDINGS_PATH) > 0
        and os.path.getsize(NAMES_PATH) > 0
    ):
        embeddings = np.load(EMBEDDINGS_PATH)
        names = np.load(NAMES_PATH)
    else:
        embeddings = np.empty((0, 128))
        names = np.array([])

    return embeddings, names

def save_database(embeddings, names):
    os.makedirs("data", exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(NAMES_PATH, names)

def reset_database():
    embeddings = np.empty((0,128))
    names = np.array([])
    save_database(embeddings,names)

def delete_person(person_name):
    embeddings, names = load_database()
    mask = names != person_name
    save_database(embeddings[mask], names[mask])
