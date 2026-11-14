"""
content_based.py  -  Content-based artist recommendation using musical features

  1) Use artist feature vectors (artist_features.csv): Spotify track-level audio
     features averaged per artist.
  2) Build a user preference vector as the play-count-weighted average of the
     vectors of the artists the user listened to.
  3) Recommend artists whose vectors are close to the preference vector
     (artists already heard in train are excluded).
  4) Align feature scales before computing similarity, since the features have
     different numeric ranges (StandardScaler).
  5) Do not use other users' listening patterns (a different source of evidence
     from collaborative filtering).

Final configuration (defaults)
  Chosen from the team's evaluation of the experiment files (Precision@10 on the shared
  test split): user weights = log1p(plays), StandardScaler after log1p on the four skewed
  features (speechiness, acousticness, instrumentalness, liveness), method = cosine_learned.
  Differences between settings were small and were observed on test, so they are not an
  independently validated optimum. To reproduce all experiments, pass
  --methods cosine,cosine_learned,knn together with the individual options.

Techniques used
  - Normalization     : StandardScaler (+ optional log1p), L2 normalization of vectors
  - Vector space model: cosine similarity between user preference vector and artist
                        vectors (main method, "cosine")
  - k-NN              : score by similarity to the k most similar artists the user
                        has heard ("knn")
  - Loss function + gradient descent : learn per-feature weights with a BPR loss
                        ("cosine_learned")
  - Clustering        : group artists into music-style clusters with KMeans and
                        compute each user's taste diversity

Inputs  (outputs/): artist_features.csv, train.csv   (user_id, artist_name, plays, preference)
Outputs (content_based_result/):
                    content_based_recs.csv - Top-K recommendations per user and method
                                             (used by the teammate doing the evaluation)
                    With --save-extras, also: content_based_feature_weights.csv,
                    artist_clusters.csv, artist_cluster_profiles.csv, user_taste_diversity.csv
                    (analysis files; clustering is skipped unless --save-extras is given)

Note: Evaluation (Precision/Recall, etc.) is implemented by another teammate, so it is
      not included here. test.csv is never read; only train.csv is used to build user
      vectors and to learn the feature weights.

Usage (from the repository root)
  python content_based.py
  python content_based.py --out-dir outputs --results-dir content_based_result --k 10 --weight log
  python content_based.py --methods cosine,cosine_learned,knn --no-log-skewed --tag base   # experiment baseline

Requirements: numpy, pandas, scikit-learn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo",
]
# Features concentrated near 0 (right-skewed). log1p is applied with --log-skewed
SKEWED_FEATURES = ["speechiness", "acousticness", "instrumentalness", "liveness"]


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def find_file(out_dir: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        found = sorted(out_dir.glob(pattern))
        if found:
            return found[0]
    raise FileNotFoundError(f"Could not find any of {patterns} in {out_dir}.")


def load_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"user_id": str, "artist_name": str})
    missing = {"user_id", "artist_name", "plays"} - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns {sorted(missing)}. Actual columns: {list(df.columns)}")
    return df


def build_interaction_matrix(df: pd.DataFrame, user_index: dict, item_index: dict,
                             scheme: str) -> np.ndarray:
    """User x artist weight matrix. A value of 0 means the user did not listen to the artist in train."""
    d = df[df["user_id"].isin(list(user_index)) & df["artist_name"].isin(list(item_index))]
    agg = d.groupby(["user_id", "artist_name"], as_index=False)["plays"].sum()
    u = agg["user_id"].map(user_index).to_numpy()
    i = agg["artist_name"].map(item_index).to_numpy()
    if scheme == "log":
        v = np.log1p(agg["plays"].to_numpy())
    elif scheme == "plays":
        v = agg["plays"].to_numpy()
    else:                                        # binary
        v = np.ones(len(agg))
    R = np.zeros((len(user_index), len(item_index)), dtype=np.float32)
    R[u, i] = v
    return R


# ---------------------------------------------------------------------------
# 2. Normalization + artist/user vectors (vector space model)
# ---------------------------------------------------------------------------
def build_item_matrix(features: pd.DataFrame, log_skewed: bool, scaling: str = "standard") -> np.ndarray:
    """Audio features -> artist matrix.

    scaling="standard": standardized (mean 0, std 1) so that wide-range features such as
                        loudness and tempo do not dominate the similarity.
    scaling="none":     raw values (used only to show why scaling is needed).
    """
    X = features[AUDIO_FEATURES].astype(float).copy()
    X = X.fillna(X.median())
    if log_skewed:
        for col in SKEWED_FEATURES:
            X[col] = np.log1p(X[col])
    if scaling == "none":
        return X.to_numpy(dtype=np.float32)
    return StandardScaler().fit_transform(X).astype(np.float32)


def user_profiles(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """User preference vector = play-count-weighted average of the vectors of listened artists (then L2-normalized)."""
    denom = np.maximum(R.sum(axis=1, keepdims=True), 1e-9)
    return normalize((R @ Z) / denom).astype(np.float32)


def cosine_scores(P_hat: np.ndarray, X_hat: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """P_hat and X_hat are L2-normalized vectors. If w is given, use a per-feature weighted dot product."""
    if w is not None:
        P_hat = P_hat * w
    return P_hat @ X_hat.T


# ---------------------------------------------------------------------------
# 3. k-NN (artist level, based on content similarity)
# ---------------------------------------------------------------------------
def knn_scores(X_hat: np.ndarray, R: np.ndarray, n_neighbors: int) -> np.ndarray:
    """Score of candidate artist c = weighted average of (similarity x preference weight)
    over the k artists the user has heard that are most similar to c.
    Other users' listening records are not used."""
    n_users, n_items = R.shape
    S = X_hat @ X_hat.T
    scores = np.full((n_users, n_items), -np.inf, dtype=np.float32)
    for u in range(n_users):
        liked = np.flatnonzero(R[u])
        if len(liked) == 0:
            continue
        w = R[u, liked]
        sims = S[:, liked]
        kk = min(n_neighbors, len(liked))
        idx = np.argpartition(sims, -kk, axis=1)[:, -kk:]
        sim_top = np.take_along_axis(sims, idx, axis=1)
        w_top = w[idx]
        scores[u] = (sim_top * w_top).sum(axis=1) / np.maximum(w_top.sum(axis=1), 1e-9)
    return scores


# ---------------------------------------------------------------------------
# 4. Loss function + gradient descent: learn per-feature weights (BPR loss)
# ---------------------------------------------------------------------------
def learn_feature_weights(R: np.ndarray, Z: np.ndarray, X_hat: np.ndarray, *, epochs: int,
                          lr: float, reg: float, val_frac: float, min_liked: int, seed: int):
    """
    Score:  s(u, i) = sum_f w_f * p_u[f] * x_i[f]   (cosine similarity when w = 1)
    Loss:   L = mean( log(1 + exp(-(s(u,pos) - s(u,neg)))) ) + reg * ||w - 1||^2   (BPR + L2)
    Update: gradient descent on w, constrained to w >= 0.

    Leakage prevention: inside train, a fraction (val_frac) of each user's artists is held
    out as "validation positives". The user vector built from the remaining artists is
    trained to score the validation positives higher than negatives (artists the user did
    not hear). test.csv is never used.
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = R.shape
    fit_R = R.copy()
    pos_u, pos_i = [], []
    for u in range(n_users):
        liked = np.flatnonzero(R[u])
        if len(liked) < min_liked:
            continue
        n_val = max(1, int(len(liked) * val_frac))
        val = rng.choice(liked, size=n_val, replace=False)
        fit_R[u, val] = 0
        pos_u.extend([u] * n_val)
        pos_i.extend(val.tolist())
    pos_u, pos_i = np.array(pos_u), np.array(pos_i)
    if len(pos_u) == 0:
        print("[Warning] No users available for weight learning; keeping w = 1.")
        return np.ones(Z.shape[1], dtype=np.float32), []

    P_fit = user_profiles(fit_R, Z)                  # user vectors built without the held-out validation positives
    seen = R > 0
    w = np.ones(Z.shape[1], dtype=np.float64)
    history = []

    def sample_negatives():
        neg = rng.integers(0, n_items, size=len(pos_u))
        bad = seen[pos_u, neg]
        while bad.any():                              # resample if an already-heard artist was drawn
            neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
            bad = seen[pos_u, neg]
        return neg

    P_u = P_fit[pos_u].astype(np.float64)
    X_pos = X_hat[pos_i].astype(np.float64)
    for epoch in range(epochs + 1):
        neg = sample_negatives()
        A = P_u * (X_pos - X_hat[neg].astype(np.float64))   # d = A @ w  (positive score - negative score)
        d = A @ w
        loss = np.mean(np.logaddexp(0.0, -d)) + reg * np.sum((w - 1.0) ** 2)
        acc = float(np.mean(d > 0))
        if epoch % 50 == 0 or epoch == epochs:
            history.append((epoch, loss, acc))
        if epoch == epochs:
            break
        sig = 1.0 / (1.0 + np.exp(d))                        # sigma(-d)
        grad = -(A * sig[:, None]).mean(axis=0) + 2.0 * reg * (w - 1.0)
        w = np.clip(w - lr * grad, 0.0, None)

    w = w / max(w.mean(), 1e-9)                              # rescale to mean 1 for readability (ranking is unchanged)
    return w.astype(np.float32), history


# ---------------------------------------------------------------------------
# 5. Clustering (KMeans): music-style clusters + user taste diversity
# ---------------------------------------------------------------------------
def cluster_artists(Z: np.ndarray, k_min: int, k_max: int, seed: int):
    best_k, best_score = None, -1.0
    sample = min(2000, len(Z))
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(Z)
        score = silhouette_score(Z, labels, sample_size=sample, random_state=seed)
        print(f"  KMeans k={k:2d}  silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    km = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(Z)
    print(f"  -> selected number of clusters k={best_k} (silhouette={best_score:.3f})")
    return km.labels_, best_k


def taste_diversity(R: np.ndarray, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Closer to 1 when the artists a user heard are spread over many clusters (normalized entropy)."""
    onehot = np.eye(n_clusters, dtype=np.float32)[labels]
    share = R @ onehot
    share = share / np.maximum(share.sum(axis=1, keepdims=True), 1e-9)
    ent = -(share * np.log(np.maximum(share, 1e-12))).sum(axis=1)
    return ent / np.log(n_clusters)


# ---------------------------------------------------------------------------
# 6. Recommendation
# ---------------------------------------------------------------------------
def top_k(scores: np.ndarray, seen: np.ndarray, k: int) -> np.ndarray:
    s = np.where(seen, -np.inf, scores)                      # exclude artists already heard in train
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.arange(s.shape[0])[:, None]
    order = np.argsort(-s[rows, idx], axis=1)
    return idx[rows, order]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Content-based artist recommendation")
    p.add_argument("--out-dir", default="outputs", help="Folder containing the preprocessed input CSVs (artist_features.csv, train.csv)")
    p.add_argument("--results-dir", default="content_based_result", help="Folder where all result files are saved (created if missing)")
    p.add_argument("--k", type=int, default=10, help="Number of recommendations per user (K)")
    p.add_argument("--weight", choices=["log", "plays", "binary"], default="log",
                   help="User vector weights: log=log1p(play count), plays=raw play count, binary=listened or not")
    p.add_argument("--scaling", choices=["standard", "none"], default="standard",
                   help="Feature scaling: standard=StandardScaler, none=raw feature values")
    p.add_argument("--log-skewed", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply log1p to skewed features before standardization (default: on; use --no-log-skewed to turn off)")
    p.add_argument("--methods", default="cosine_learned",
                   help="Comma-separated methods to run: cosine, cosine_learned, knn "
                        "(default: cosine_learned = final method; use cosine,cosine_learned,knn to reproduce the experiments)")
    p.add_argument("--knn-neighbors", type=int, default=10)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--reg", type=float, default=0.01, help="L2 regularization strength for the weights (pulls w toward 1)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-extras", action="store_true",
                   help="Also run clustering and save the analysis files (feature weights, artist clusters, "
                        "cluster profiles, user taste diversity). By default only the recommendations file is saved")
    p.add_argument("--tag", default="",
                   help="Experiment name appended to every output file name (e.g. --tag w_log -> content_based_recs_w_log.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)            # input folder
    res_dir = Path(args.results_dir)        # output folder
    res_dir.mkdir(parents=True, exist_ok=True)
    K = args.k
    sfx = f"_{args.tag}" if args.tag else ""
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - {"cosine", "cosine_learned", "knn"}
    if unknown or not methods:
        raise ValueError(f"Unknown or empty --methods: {sorted(unknown)}. Choose from cosine, cosine_learned, knn.")
    print(f"Config: methods={methods} weight={args.weight} scaling={args.scaling} log_skewed={args.log_skewed} "
          f"knn_neighbors={args.knn_neighbors} reg={args.reg} seed={args.seed} k={K} tag='{args.tag}'")

    # ---- Data (train.csv only) ----
    features = pd.read_csv(find_file(out_dir, ["artist_features.csv", "artist_feature*.csv"]))
    features["artist_name"] = features["artist_name"].astype(str)
    features = features.drop_duplicates("artist_name").reset_index(drop=True)
    train = load_train(find_file(out_dir, ["train.csv"]))

    artists = features["artist_name"].tolist()
    item_index = {a: i for i, a in enumerate(artists)}
    users = sorted(train["user_id"].unique())
    user_index = {u: i for i, u in enumerate(users)}
    n_users, n_items = len(users), len(artists)
    print(f"Users: {n_users:,} | Candidate artists: {n_items:,} | Train rows: {len(train):,}")
    print("(Candidates are limited to artists that passed the preprocessing threshold of at least 5 listeners)")

    R = build_interaction_matrix(train, user_index, item_index, args.weight)
    seen = R > 0

    # ---- Normalization + vectors ----
    Z = build_item_matrix(features, args.log_skewed, args.scaling)
    X_hat = normalize(Z).astype(np.float32)
    P_hat = user_profiles(R, Z)

    # ---- Scores per method ----
    scores: dict[str, np.ndarray] = {}
    weights_df = None
    if "cosine" in methods:
        scores["cosine"] = cosine_scores(P_hat, X_hat)

    if "cosine_learned" in methods:
        print("\n[Gradient descent] Learning per-feature weights with BPR loss")
        w, history = learn_feature_weights(
            R, Z, X_hat, epochs=args.epochs, lr=args.lr, reg=args.reg,
            val_frac=args.val_frac, min_liked=5, seed=args.seed)
        for epoch, loss, acc in history:
            print(f"  epoch {epoch:4d}  loss={loss:.4f}  validation pair accuracy={acc:.3f}")
        weights_df = pd.DataFrame({"feature": AUDIO_FEATURES, "weight": np.round(w, 4)})
        print(weights_df.to_string(index=False))
        scores["cosine_learned"] = cosine_scores(P_hat, X_hat, w)

    if "knn" in methods:
        print("\n[k-NN] Scoring by the k most similar artists the user has heard")
        scores["knn"] = knn_scores(X_hat, R, args.knn_neighbors)

    # ---- Clustering ----
    if args.save_extras:
        print("\n[Clustering] Finding music-style clusters with KMeans")
        labels, n_clusters = cluster_artists(Z, args.k_min, args.k_max, args.seed)
        listeners = seen.sum(axis=0)
        profile = features[AUDIO_FEATURES].assign(cluster=labels).groupby("cluster").mean().round(3)
        profile.insert(0, "n_artists", np.bincount(labels, minlength=n_clusters))
        top_artists = []
        for c in range(n_clusters):
            members = np.flatnonzero(labels == c)
            best = members[np.argsort(-listeners[members])[:5]]
            top_artists.append(", ".join(artists[i] for i in best))
        profile["top_artists"] = top_artists
        print(profile.to_string())

        diversity = pd.DataFrame({
            "user_id": users,
            "n_train_artists": seen.sum(axis=1),
            "taste_diversity": np.round(taste_diversity(R, labels, n_clusters), 4),
        })

    # ---- Recommendation (Top-K) ----
    rec_rows = []
    for method, sc in scores.items():
        rec = top_k(sc, seen, K)
        for u in range(n_users):
            for rank, item in enumerate(rec[u], start=1):
                rec_rows.append((method, users[u], rank, artists[item], float(sc[u, item])))
        print(f"  {method:15s} distinct artists recommended: {len(np.unique(rec)):,} / {n_items:,} candidates")

    # ---- Save ----
    pd.DataFrame(rec_rows, columns=["method", "user_id", "rank", "artist_name", "score"]) \
        .to_csv(res_dir / f"content_based_recs{sfx}.csv", index=False)
    if args.save_extras:
        if weights_df is not None:
            weights_df.to_csv(res_dir / f"content_based_feature_weights{sfx}.csv", index=False)
        pd.DataFrame({"artist_name": artists, "cluster": labels}) \
            .to_csv(res_dir / f"artist_clusters{sfx}.csv", index=False)
        profile.to_csv(res_dir / f"artist_cluster_profiles{sfx}.csv")
        diversity.to_csv(res_dir / f"user_taste_diversity{sfx}.csv", index=False)
        print(f"\nSaved to {res_dir}/ (content_based_recs{sfx}.csv and 4 extra analysis files)")
    else:
        print(f"\nSaved to {res_dir}/content_based_recs{sfx}.csv")


if __name__ == "__main__":
    main()