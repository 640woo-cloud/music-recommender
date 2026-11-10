# 추천 결과 평가 함수
# 모든 모델은 이 함수로 평가해서 같은 기준으로 비교
# 추천 결과는 user_id, rank, artist_name 컬럼으로 저장
import pandas as pd


def precision_at_k(recommended, actual, k=10):
    # 추천한 k개 중에 실제로 들은 아티스트 비율
    hits = len(set(recommended[:k]) & set(actual))
    return hits / k


def recall_at_k(recommended, actual, k=10):
    # 실제로 들은 아티스트 중에 추천으로 맞힌 비율
    hits = len(set(recommended[:k]) & set(actual))
    return hits / len(actual)


def evaluate(recs, test, k=10):
    # recs: 추천 결과 DataFrame (user_id, rank, artist_name)
    # test: 평가용 DataFrame (user_id, artist_name)
    recs = recs.sort_values(["user_id", "rank"])
    rec_dict = recs.groupby("user_id")["artist_name"].apply(list).to_dict()
    test_dict = test.groupby("user_id")["artist_name"].apply(list).to_dict()

    precisions = []
    recalls = []
    for user, actual in test_dict.items():
        recommended = rec_dict.get(user, [])  # 추천이 없는 사용자는 빈 리스트 -> 0점
        precisions.append(precision_at_k(recommended, actual, k))
        recalls.append(recall_at_k(recommended, actual, k))

    print(f"Precision@{k}: {sum(precisions) / len(precisions):.4f}")
    print(f"Recall@{k}: {sum(recalls) / len(recalls):.4f}")
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)