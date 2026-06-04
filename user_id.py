import pandas as pd
import numpy as np

# Carica il dataset originale
tracks = pd.read_csv("music.csv")

# ==========================
# 1. CREA songs.csv
# ==========================

songs = tracks[
    [
        "track_id",
        "artists",
        "track_name",
        "popularity",
        "duration_ms",
        "explicit",
        "danceability",
        "energy"
    ]
].copy()

# Rimuove eventuali duplicati
songs = songs.drop_duplicates(subset=["track_id"])

songs.to_csv("songs.csv", index=False)

print("songs.csv creato")
print("Numero canzoni:", len(songs))

# ==========================
# 2. CREA interactions.csv
# ==========================

n_users = 1000
listens_per_user = 50

rows = []

np.random.seed(42)

for user_id in range(1, n_users + 1):

    listened_tracks = songs.sample(listens_per_user)

    for _, song in listened_tracks.iterrows():

        rows.append({
            "user_id": user_id,
            "track_id": song["track_id"],
            "plays": np.random.randint(1, 20)
        })

interactions = pd.DataFrame(rows)

interactions.to_csv("interactions.csv", index=False)

print("interactions.csv creato")
print("Numero interazioni:", len(interactions))

# ==========================
# 3. CONTROLLI
# ==========================

print("\n--- SONGS ---")
print(songs.head())

print("\n--- INTERACTIONS ---")
print(interactions.head())

print("\nUtenti:", interactions["user_id"].nunique())
print("Brani ascoltati:", interactions["track_id"].nunique())