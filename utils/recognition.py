import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.55

def recognize(embedding, embeddings_db, names_db):
    if len(embeddings_db) == 0:
        return "Unknown",0.0
    
    sims = cosine_similarity([embedding], embeddings_db)[0]
    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score >= THRESHOLD:
        return names_db[best_idx], float(best_score)
    else:
        return "Unknown",float(best_score)
    
