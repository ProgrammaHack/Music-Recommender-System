from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "models" / "mf_recommender.joblib"

artifacts = joblib.load(ARTIFACT_PATH)

user_ids = artifacts["user_ids"]
track_ids = artifacts["track_ids"]
user_to_idx = artifacts["user_to_idx"]
track_to_idx = artifacts["track_to_idx"]
user_factors = artifacts["user_factors"]
item_factors = artifacts["item_factors"]
seen_by_user = artifacts["seen_by_user"]
popular_item_codes = artifacts["popular_item_codes"]
song_meta = artifacts["song_meta"]

def _build_row(track_code, score, rank):
    track_id = track_ids[track_code]
    meta = song_meta.get(track_id, {})
    return {
        "rank": rank,
        "track_id": track_id,
        "track_name": meta.get("track_name", ""),
        "artists": meta.get("artists", ""),
        "score": float(score),
        "popularity": meta.get("popularity", None),
        "danceability": meta.get("danceability", None),
        "energy": meta.get("energy", None),
        "duration_ms": meta.get("duration_ms", None),
        "explicit": meta.get("explicit", None),
    }

def popular_recommendations(top_n=10):
    recs = []
    for rank, track_code in enumerate(popular_item_codes[:top_n], start=1):
        recs.append(_build_row(track_code, score=0.0, rank=rank))
    return recs

def recommend_for_user(user_id, top_n=10):
    """
    Raccomanda brani a un utente già presente nel dataset.
    Se l'utente non esiste, restituisce i brani più popolari.
    """
    if user_id not in user_to_idx:
        return popular_recommendations(top_n=top_n)

    u_idx = user_to_idx[user_id]

    # Score di tutti i brani per quell'utente
    scores = user_factors[u_idx] @ item_factors.T

    # Escludiamo i brani già ascoltati
    seen = seen_by_user[u_idx]

    ranked_item_idx = np.argsort(scores)[::-1]

    recs = []
    rank = 1
    for item_idx in ranked_item_idx:
        if int(item_idx) in seen:
            continue
        recs.append(_build_row(int(item_idx), scores[item_idx], rank))
        rank += 1
        if len(recs) >= top_n:
            break

    if not recs:
        return popular_recommendations(top_n=top_n)

    return recs

def get_stats():
    return {
        "n_users": len(user_ids),
        "n_items": len(track_ids),
        "n_factors": user_factors.shape[1],
    }