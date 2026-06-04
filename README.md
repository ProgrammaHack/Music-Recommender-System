# 🎵 Music Recommender System

Progetto di **sistema di raccomandazione musicale** basato su **collaborative filtering** e **matrix factorization**, presentato tramite una **web app Streamlit**.

L’obiettivo è semplice: **consigliare canzoni a un utente in base ai suoi ascolti precedenti**.

---

## ✨ Cosa fa il progetto

- legge un dataset di brani (`songs.csv`)
- legge un dataset di interazioni utente-brano (`interactions.csv`)
- costruisce una matrice utenti × brani
- allena un modello di factorization
- salva il modello in un file `.joblib`
- mostra le raccomandazioni in una web app Streamlit

---

## 📁 Struttura del progetto

```text
music_recommender/
├── data/
│   ├── songs.csv
│   └── interactions.csv
├── models/
│   └── mf_recommender.joblib
├── app.py
├── recommender.py
├── train_model.py
├── main.py
└── requirements.txt
````

### Cosa fa ogni file

**`data/songs.csv`**
Contiene le informazioni sui brani:

* `track_id`
* `artists`
* `track_name`
* `popularity`
* `duration_ms`
* `explicit`
* `danceability`
* `energy`

**`data/interactions.csv`**
Contiene le interazioni degli utenti:

* `user_id`
* `track_id`
* `plays`

**`train_model.py`**
Allena il modello e salva tutto in `models/mf_recommender.joblib`.

**`recommender.py`**
Carica il modello salvato e genera le raccomandazioni.

**`app.py`**
È la web app Streamlit con l’interfaccia utente.

**`requirements.txt`**
Lista delle librerie necessarie.

---

## 🧠 Idea del sistema

Il sistema usa il comportamento degli utenti per imparare preferenze latenti.

### Matrice utenti-item

Ogni riga rappresenta un utente, ogni colonna un brano e ogni valore rappresenta un certo numero di ascolti (`plays`).

$$
R \in \mathbb{R}^{n_{users} \times n_{items}}
$$

Dove:

* `n_users` = numero di utenti
* `n_items` = numero di canzoni

---

## 📐 Formula della matrix factorization

L’idea è approssimare la matrice delle interazioni con il prodotto di due matrici più piccole:

$$
R \approx UV
$$

Dove:

* `U` = matrice utenti × fattori latenti
* `V` = matrice fattori latenti × brani

La previsione per l’utente `u` e il brano `i` è:

$$
\hat{r}_{ui} = U_u \cdot V_i
$$

cioè il **prodotto scalare** tra il vettore dell’utente e il vettore del brano.

L’errore su una coppia osservata è:

$$
e_{ui} = r_{ui} - \hat{r}_{ui}
$$

---

## Librerie usate e perché

### `pandas`

Usata per leggere, pulire e organizzare i CSV.

### `numpy`

Usata per i calcoli numerici e gli score di raccomandazione.

### `scikit-learn`

Usata per `TruncatedSVD`, che implementa una forma di matrix factorization adatta a matrici sparse.

### `scipy`

Usata per costruire la matrice sparsa `csr_matrix`.

### `joblib`

Usata per salvare e ricaricare il modello addestrato.

### `streamlit`

Usata per costruire la web app.

---

## ⚙️ Training del modello

Il training avviene nel file `train_model.py`.

### 1. Caricamento dati

```python
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

songs = pd.read_csv("data/songs.csv")
interactions = pd.read_csv("data/interactions.csv")
```

### Perché questo passaggio

* `songs.csv` serve per recuperare titolo e artista
* `interactions.csv` serve per capire cosa ha ascoltato ogni utente
* `TruncatedSVD` lavora bene con dati sparsi

---

### 2. Pulizia minima dei dati

```python
songs = songs.drop_duplicates(subset=["track_id"]).copy()
interactions = interactions.dropna(subset=["user_id", "track_id", "plays"]).copy()
interactions["plays"] = pd.to_numeric(interactions["plays"], errors="coerce").fillna(0)
interactions = interactions[interactions["plays"] > 0].copy()
interactions = interactions[interactions["track_id"].isin(songs["track_id"])].copy()
interactions["log_plays"] = np.log1p(interactions["plays"])
```

### Perché questo passaggio

* rimuove duplicati e valori mancanti
* elimina ascolti nulli o non validi
* usa `log1p(plays)` per ridurre il peso di ascolti troppo alti

La trasformazione è:

$$
\log(1 + plays)
$$

Serve perché i conteggi degli ascolti spesso sono molto sbilanciati.

---

### 3. Creazione degli indici utenti e brani

```python
user_codes, user_ids = pd.factorize(interactions["user_id"], sort=True)
item_codes, track_ids = pd.factorize(interactions["track_id"], sort=True)

user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
track_to_idx = {track_id: idx for idx, track_id in enumerate(track_ids)}
```

### Perché questo passaggio

Le matrici numeriche lavorano meglio con indici interi. Qui trasformiamo gli ID reali in posizioni numeriche.

---

### 4. Costruzione della matrice sparsa

```python
X = csr_matrix(
    (interactions["log_plays"].values, (user_codes, item_codes)),
    shape=(len(user_ids), len(track_ids))
)
```

### Perché questo passaggio

La matrice utenti-item è molto grande e molto vuota. La versione sparsa occupa meno memoria.

---

### 5. Training con TruncatedSVD

```python
n_components = min(50, max(2, min(X.shape) - 1))
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_factors = svd.fit_transform(X)
item_factors = svd.components_.T
```

### Cosa significa

`TruncatedSVD` trova una rappresentazione compatta della matrice, imparando fattori latenti per utenti e canzoni.

* `user_factors` = vettori degli utenti
* `item_factors` = vettori dei brani

Questo è il cuore della matrix factorization.

---

### 6. Salvataggio del modello

```python
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

joblib.dump(artifacts, "models/mf_recommender.joblib")
```

### Perché questo passaggio

Salviamo tutto quello che serve per fare raccomandazioni senza riaddestrare il modello ogni volta.

---

## 🎧 Logica delle raccomandazioni

Il file `recommender.py` contiene la funzione che calcola i consigli.

### Parte principale

```python
scores = user_factors[u_idx] @ item_factors.T
```

### Significato

Calcola uno score per ogni canzone facendo il prodotto tra il vettore dell’utente e tutti i vettori dei brani.

Poi il sistema:

* esclude i brani già ascoltati
* ordina i restanti per score
* restituisce i primi `N`

---

### Funzione di raccomandazione

```python
def recommend_for_user(user_id, top_n=10):
    if user_id not in user_to_idx:
        return popular_recommendations(top_n=top_n)

    u_idx = user_to_idx[user_id]
    scores = user_factors[u_idx] @ item_factors.T
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
```

### Perché questo passaggio

* se l’utente esiste, usa il suo profilo
* se non esiste, propone i brani più popolari
* non ripete canzoni già ascoltate

---

## 🌐 Web app Streamlit

Il file `app.py` gestisce l’interfaccia.

### Esempio

```python
import streamlit as st
import pandas as pd
from recommender import recommend_for_user, get_stats, user_ids

st.title("🎵 Music Recommender System")
selected_user = st.selectbox("User ID", sorted(list(user_ids)))

if st.button("Genera raccomandazioni"):
    recs = recommend_for_user(selected_user, top_n=10)
    df = pd.DataFrame(recs)
    st.dataframe(df)
```

### Perché questo passaggio

Streamlit permette di creare una pagina web semplice in cui l’utente sceglie un ID e riceve una lista di brani consigliati.

---

## Come eseguire il progetto

### 1. Installare le librerie

```bash
pip install -r requirements.txt
```

### 2. Allenare il modello

```bash
python train_model.py
```

Questo crea il file:

```text
models/mf_recommender.joblib
```

### 3. Avviare la web app

```bash
streamlit run app.py
```

---

## Come valutare il modello

Per valutare un recommender si possono usare:

* **Hit Rate**
* **Precision@K**
* **Recall@K**
* **RMSE** se si tratta il problema come previsione numerica

### Esempio di formula di Precision@K

$$
\text{Precision@K} = \frac{\text{raccomandazioni corrette tra le top K}}{K}
$$

### Esempio di formula di Recall@K

$$
\text{Recall@K} = \frac{\text{raccomandazioni corrette trovate}}{\text{elementi rilevanti totali}}
$$

---

## ⚠️ Limiti del progetto

Il sistema usa solo:

* `user_id`
* `track_id`
* `plays`

Quindi:

* personalizza bene i consigli
* ma non conosce direttamente il genere musicale
* non usa esplicitamente il contenuto audio per decidere

Per questo, una possibile evoluzione è un **sistema ibrido**:

* collaborative filtering
* content-based filtering con `danceability`, `energy`, `popularity`

---

## Possibili miglioramenti

* filtro per mood o genere
* raccomandazioni basate su canzoni simili
* grafico delle caratteristiche musicali dell’utente
* sistema ibrido con punteggio finale combinato
* pagina “perché te lo consiglio”

---

## Cosa dimostra questo progetto

Questo progetto mostra come:

* trasformare ascolti utente in dati numerici
* costruire una matrice utenti-item
* allenare una matrix factorization
* generare raccomandazioni personalizzate
* creare una web app interattiva con Streamlit

---

## 📌 Nota finale

Il progetto è stato pensato in modo modulare per essere facile da leggere, facile da presentare e facile da migliorare con tecniche più avanzate in futuro.

```
`
