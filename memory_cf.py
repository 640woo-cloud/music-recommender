"""아티스트 기반 협업 필터링"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    """log 청취횟수 벡터의 코사인 유사도 + 선호도 가중합 추천"""

    def fit(self, train):
        required = {"user_id", "artist_name", "preference"}
        if not required.issubset(train.columns) or train.empty:
            raise ValueError("비어 있지 않은 user_id, artist_name, preference가 필요합니다.")
        if train[list(required)].isna().any().any():
            raise ValueError("사용자, 아티스트, 선호도에 결측값이 있습니다.")
        if train.duplicated(["user_id", "artist_name"]).any():
            raise ValueError("사용자–아티스트 쌍은 중복되면 안 됩니다.")
        values = train["preference"].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("관측된 선호도는 유한한 양수여야 합니다.")
        self.users = pd.Index(sorted(train["user_id"].unique()))
        self.artists = pd.Index(sorted(train["artist_name"].unique()))
        self.matrix = csr_matrix(
            (values, (self.users.get_indexer(train["user_id"]),
                      self.artists.get_indexer(train["artist_name"]))),
            shape=(len(self.users), len(self.artists)),
        )
        # 행: 사용자, 열: 아티스트
        self.similarity = cosine_similarity(self.matrix.T)
        np.fill_diagonal(self.similarity, 0.0)
        return self

    def _rank(self, scores, k):
        if k < 1:
            raise ValueError("k는 1 이상이어야 합니다.")
        # 동점은 이름순. 근거가 없는 0점 후보는 추천하지 않는다.
        order = np.argsort(-scores, kind="stable")
        return order[scores[order] > 0][:k]

    def recommend(self, user_id, k=10):
        if user_id not in self.users:
            raise ValueError(f"train에 없는 사용자입니다: {user_id}")
        history = self.matrix.getrow(self.users.get_loc(user_id))
        # score(u, i) = sum_j preference(u, j) * similarity(j, i)
        scores = np.asarray(history @ self.similarity).ravel()
        scores[history.indices] = -np.inf
        selected = self._rank(scores, k)
        return pd.DataFrame({"user_id": user_id,
                             "rank": np.arange(1, len(selected) + 1),
                             "artist_name": self.artists[selected],
                             "score": scores[selected]})

    def recommend_all(self, k=10):
        return pd.concat([self.recommend(user, k) for user in self.users], ignore_index=True)

    def similar_artists(self, artist_name, k=10):
        if artist_name not in self.artists:
            raise ValueError(f"train에 없는 아티스트입니다: {artist_name}")
        scores = self.similarity[self.artists.get_loc(artist_name)]
        selected = self._rank(scores, k)
        return pd.DataFrame({"artist_name": self.artists[selected], "similarity": scores[selected]})

    def explain(self, user_id, artist_name, k=5):
        # 추천 점수에 많이 기여한 기존 청취 아티스트를 보여준다.
        if user_id not in self.users or artist_name not in self.artists:
            raise ValueError("train에 있는 사용자와 아티스트를 입력하세요.")
        history = self.matrix.getrow(self.users.get_loc(user_id))
        sim = self.similarity[history.indices, self.artists.get_loc(artist_name)]
        result = pd.DataFrame({"listened_artist": self.artists[history.indices],
                               "preference": history.data,
                               "similarity": sim,
                               "contribution": history.data * sim})
        return result[result["contribution"] > 0].sort_values(
            ["contribution", "listened_artist"], ascending=[False, True]).head(k).reset_index(drop=True)
