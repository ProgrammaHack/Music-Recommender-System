from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

SONGS_PATH = DATA_DIR / "songs.csv"
INTERACTIONS_PATH = DATA_DIR / "interactions.csv"
ARTIFACT_PATH = MODELS_DIR / "mf_recommender.joblib"

print("Caricamento dati...")

songs = pd.read_csv(SONGS_PATH)
interactions = pd.read_csv(INTERACTIONS_PATH)

# Pulizia minima
songs = songs.drop_duplicates(subset=["track_id"]).copy()
interactions = interactions.dropna(subset=["user_id", "track_id", "plays"]).copy()
interactions["plays"] = pd.to_numeric(interactions["plays"], errors="coerce").fillna(0)
interactions = interactions[interactions["plays"] > 0].copy()

# Teniamo solo i track_id presenti nel file songs.csv
interactions = interactions[interactions["track_id"].isin(songs["track_id"])].copy()

# Peso più stabile per gli ascolti
interactions["log_plays"] = np.log1p(interactions["plays"])

print("Creazione indici...")

user_codes, user_ids = pd.factorize(interactions["user_id"], sort=True)
item_codes, track_ids = pd.factorize(interactions["track_id"], sort=True)

user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
track_to_idx = {track_id: idx for idx, track_id in enumerate(track_ids)}

print("Costruzione matrice sparsa...")
X = csr_matrix(
    (interactions["log_plays"].values, (user_codes, item_codes)),
    shape=(len(user_ids), len(track_ids))
)

# Numero fattori latenti
n_components = min(50, max(2, min(X.shape) - 1))

print(f"Training SVD con {n_components} fattori latenti...")
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(X)
item_factors = svd.components_.T

# Item ascoltati da ogni utente
seen_by_user = [set() for _ in range(len(user_ids))]
for u_code, i_code in zip(user_codes, item_codes):
    seen_by_user[int(u_code)].add(int(i_code))

# Popolarità globale dagli ascolti
popular_track_ids = (
    interactions.groupby("track_id")["plays"]
    .sum()
    .sort_values(ascending=False)
    .index.tolist()
)

popular_item_codes = [
    track_to_idx[track_id]
    for track_id in popular_track_ids
    if track_id in track_to_idx
]

# Metadati canzoni
song_meta = songs.set_index("track_id")[
    ["track_name", "artists", "popularity", "duration_ms", "explicit", "danceability", "energy"]
].to_dict(orient="index")

artifacts = {
    "user_ids": user_ids.tolist(),
    "track_ids": track_ids.tolist(),
    "user_to_idx": user_to_idx,
    "track_to_idx": track_to_idx,
    "user_factors": user_factors,
    "item_factors": item_factors,
    "seen_by_user": seen_by_user,
    "popular_item_codes": popular_item_codes,
    "song_meta": song_meta,
    "svd_explained_variance": svd.explained_variance_ratio_.tolist(),
}

joblib.dump(artifacts, ARTIFACT_PATH)

print("\nTraining completato.")
print(f"Utenti: {len(user_ids)}")
print(f"Brani: {len(track_ids)}")
print(f"Fattori latenti: {n_components}")
print(f"Modello salvato in: {ARTIFACT_PATH}")