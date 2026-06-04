import streamlit as st
import pandas as pd

from recommender import recommend_for_user, get_stats, user_ids

st.set_page_config(
    page_title="Music Recommender System",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Music Recommender System")
st.caption("Collaborative filtering con Matrix Factorization")

stats = get_stats()

col1, col2, col3 = st.columns(3)
col1.metric("Utenti", stats["n_users"])
col2.metric("Brani", stats["n_items"])
col3.metric("Fattori latenti", stats["n_factors"])

st.markdown("---")

st.subheader("Scegli un utente")

user_list = sorted(list(user_ids))
selected_user = st.selectbox("User ID", user_list)

top_n = st.slider("Numero di raccomandazioni", min_value=5, max_value=20, value=10, step=1)

if st.button("Genera raccomandazioni"):
    recs = recommend_for_user(selected_user, top_n=top_n)
    df = pd.DataFrame(recs)

    st.success(f"Raccomandazioni per user_id = {selected_user}")

    st.dataframe(
        df[[
            "rank",
            "track_name",
            "artists",
            "score",
            "popularity",
            "danceability",
            "energy",
            "explicit"
        ]],
        use_container_width=True
    )

    st.subheader("Top score")
    st.bar_chart(df.set_index("track_name")["score"])

st.markdown("---")
st.write("Progetto di raccomandazione musicale basato su interazioni utente-brano.")