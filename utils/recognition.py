import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.62
MARGIN_THRESHOLD = 0.08

##recognition by comparing all the embeddings in database
def recognize_old(embedding, embeddings_db, names_db):
    if len(embeddings_db) == 0:
        return "Unknown",0.0
    
    sims = cosine_similarity([embedding], embeddings_db)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score >= THRESHOLD:
        return names_db[best_idx], float(best_score)
    else:
        return "Unknown",float(best_score)
    

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
SIM_THRESHOLD = 0.62      # absolute similarity threshold
MARGIN_THRESHOLD = 0.08   # confidence margin
# ----------------------------------------


def recognize(embedding, embeddings_db, names_db):
    """
    Recognize a face embedding against a database using
    open-set recognition logic (threshold + margin).

    Returns:
        predicted_name (str)
        best_similarity (float)
    """

    if embeddings_db is None or len(embeddings_db) == 0:
        return "Unknown", 0.0

    sims = cosine_similarity([embedding], embeddings_db)[0]

    sorted_idx = np.argsort(sims)[::-1]

    best_idx = sorted_idx[0]
    best_sim = sims[best_idx]

    if len(sorted_idx) > 1:
        second_best_sim = sims[sorted_idx[1]]
    else:
        second_best_sim = 0.0

    if (
        best_sim < SIM_THRESHOLD
        or (best_sim - second_best_sim) < MARGIN_THRESHOLD
    ):
        return "Unknown", float(best_sim)

    return names_db[best_idx], float(best_sim)


##new type -> will compare the centroid of the database
def build_centroids(embeddings, names):
    centroids = {}
    for name in set(names):
        person_embs = embeddings[names == name]
        centroids[name] = np.mean(person_embs, axis=0)
    return centroids

def recognize_centroid(live_emb, centroids,
                       threshold=0.75, margin=0.15):
    if not centroids:
        return "Unknown", 0.0

    names = list(centroids.keys())
    embs = np.array(list(centroids.values()))

    sims = cosine_similarity([live_emb], embs)[0]

    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]

    # second best
    sims_sorted = np.sort(sims)
    second_best = sims_sorted[-2] if len(sims_sorted) > 1 else 0.0

    # HARD rejection
    if best_sim < threshold:
        return "Unknown", float(best_sim)

    if (best_sim - second_best) < margin:
        return "Unknown", float(best_sim)

    return names[best_idx], float(best_sim)

