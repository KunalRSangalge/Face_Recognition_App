import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.storage import load_database

def main():
    embeddings, names = load_database()

    if len(embeddings) == 0:
        print("❌ Database is empty")
        return

    print(f"✅ Loaded {len(embeddings)} embeddings")
    print("People in DB:", sorted(set(names)))
    print("-" * 60)

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # ---- Print pairwise similarities ----
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            same_person = names[i] == names[j]
            tag = "SAME" if same_person else "DIFF"

            print(
                f"{names[i]:10s} ↔ {names[j]:10s} | "
                f"Similarity = {sim_matrix[i][j]:.4f} | {tag}"
            )

    # ---- Aggregate statistics ----
    same_sims = []
    diff_sims = []

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if names[i] == names[j]:
                same_sims.append(sim_matrix[i][j])
            else:
                diff_sims.append(sim_matrix[i][j])

    print("\n📊 STATISTICS")
    print("-" * 60)
    print(f"Same person  → min: {min(same_sims):.3f}, "
          f"max: {max(same_sims):.3f}, "
          f"avg: {np.mean(same_sims):.3f}")

    print(f"Different     → min: {min(diff_sims):.3f}, "
          f"max: {max(diff_sims):.3f}, "
          f"avg: {np.mean(diff_sims):.3f}")

if __name__ == "__main__":
    main()
