# 음악적 특성과 청취 이력을 활용한 아티스트 추천 시스템

머신러닝 텀프로젝트 7조

Last.fm 사용자 청취 이력과 Spotify 곡 오디오 특성을 연결해 사용자에게 아직 듣지 않은 아티스트 10명을 추천한다.
인기순 추천(baseline), 메모리 기반 협업 필터링(CF), 콘텐츠 기반 추천을 같은 train/test 분할과 같은 지표(Precision@10, Recall@10)로 비교한다.

## 데이터

| 데이터 | 내용 | 출처 |
|---|---|---|
| Last.fm 1K | 사용자 992명의 청취 기록 (약 1,915만 행) | http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html |
| Spotify Tracks | 곡 114,000개의 오디오 특성 (danceability, energy 등) | https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset |

원본 데이터는 용량 문제로 저장소에 올리지 않았다. 위 링크에서 받아 프로젝트 폴더에 넣으면 된다.

- `userid-timestamp-artid-artname-traid-traname.tsv` (Last.fm 1K)
- `dataset.csv` (Spotify Tracks)

전처리 결과(`outputs/*.csv`)는 저장소에 포함되어 있으므로 원본 없이도 03번 이후 단계는 바로 실행할 수 있다.

## 폴더 구조

```
music-recommender/
├── 01_data_preprocessing.ipynb   # 데이터 전처리 (아티스트 이름 매칭, 필터링)
├── 03_split.ipynb                # train/test 분할, 인기순 추천 baseline
├── 04_memory_based_cf.ipynb      # 메모리 기반 협업 필터링 + 이웃 수 실험
├── memory_cf.py                  # 아이템 기반 CF 클래스 (ItemBasedCF)
├── content_based.py              # 콘텐츠 기반 추천
├── evaluate.py                   # 공통 평가 함수 (Precision@K, Recall@K)
├── outputs/                      # 전처리·분할 결과, CF 추천 결과
├── content_based_result/         # 콘텐츠 기반 추천 결과
└── evaluation/                   # 최종 비교 평가 (결과 CSV, 평가 보고서)
```

## 실행 환경

Python 3.10 이상

```bash
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash: source .venv/Scripts/activate
pip install pandas numpy scipy scikit-learn jupyter
```

## 실행 순서

1. `01_data_preprocessing.ipynb` 실행 → `outputs/user_artist_plays.csv`, `outputs/artist_features.csv` 생성 (원본 데이터 필요)
2. `03_split.ipynb` 실행 → `outputs/train.csv`, `outputs/test.csv`, 인기순 추천 `outputs/recs_popular.csv` 생성
3. `04_memory_based_cf.ipynb` 실행 → `outputs/recs_memory_cf.csv`, `outputs/recs_memory_cf_selected.csv` 생성
4. 콘텐츠 기반 추천 실행

   ```bash
   python content_based.py
   ```
   → `content_based_result/content_based_recs.csv` 생성

5. 평가: 추천 결과 파일(`user_id, rank, artist_name`)을 `evaluate.py`로 평가한다.

   ```python
   import pandas as pd
   from evaluate import evaluate

   test = pd.read_csv("outputs/test.csv")
   recs = pd.read_csv("outputs/recs_popular.csv")
   evaluate(recs, test, k=10)
   ```

   네 가지 방법을 한 번에 평가하려면 다음을 실행한다. 결과는 `evaluation/results/final_metrics.csv`에 저장된다.

   ```bash
   python evaluation/run_evaluation.py
   ```

모든 추천 결과는 `user_id, rank, artist_name` 형식으로 저장해서 같은 평가 함수로 비교할 수 있게 했다.

## 방법 요약

### 1. 데이터 전처리
- 청취 기록을 사용자–아티스트별 재생 횟수로 집계하고, 선호도는 `log(1 + plays)`로 계산했다.
- 아티스트 이름을 정규화(NFKC, 소문자 변환, 공백 정리)해서 두 데이터셋을 연결했다. Spotify의 `artists` 열은 `;`로 나누어 아티스트별로 펼쳤다.
- 아티스트별로 오디오 특성 평균을 구해 아티스트 특성 벡터를 만들었다.
- 청취자가 5명 이상인 아티스트, 10명 이상의 아티스트를 들은 사용자만 남겼다 (변화가 없을 때까지 반복).

### 2. train/test 분할
- 사용자별로 들은 아티스트의 80%는 train, 20%는 test로 나누었다 (`random_state=42`).
- 추천할 때는 train에서 이미 들은 아티스트를 제외하고, test에 있는 아티스트를 맞히면 정답으로 본다.

### 3. 인기순 추천 (baseline)
- train에서 청취자 수가 많은 순서대로, 사용자가 아직 듣지 않은 아티스트 10명을 추천한다.

### 4. 메모리 기반 협업 필터링
- 사용자–아티스트 행렬에서 아티스트 간 코사인 유사도를 구하고, 사용자가 들은 아티스트와의 유사도에 선호도를 곱해 더한 점수로 추천한다.
- 유사 아티스트 이웃 수(전체, 10, 20, 50)는 train 안에서 다시 나눈 검증 데이터로 비교해 상위 50명을 선택했다. test는 설정 선택에 사용하지 않았다.

### 5. 콘텐츠 기반 추천
- 사용자가 들은 아티스트의 특성 벡터를 선호도로 가중 평균해서 사용자 취향 벡터를 만들고, 코사인 유사도가 높은 아티스트를 추천한다.
- 치우친 특성 4개(speechiness, acousticness, instrumentalness, liveness)는 log1p 후 StandardScaler로 정규화했다.
- 특성별 가중치는 BPR 손실과 경사하강법으로 학습했다 (`cosine_learned`). kNN 방식과 KMeans 군집 분석도 함께 실험했다.

## 주요 수치

| 항목 | 값 |
|---|---|
| 공통 아티스트 수 (이름 매칭) | 7,662명 |
| 매칭된 청취 비율 | 48.97% |
| 최종 사용자 / 아티스트 | 973명 / 4,824명 |
| 최종 사용자–아티스트 쌍 | 281,652개 |
| 행렬 희소성 | 94.00% |
| train / test | 224,925 / 56,727 |

## 결과

| 방법 | Precision@10 | Recall@10 |
|---|---|---|
| 무작위 추천 (기댓값) | 1.32% | - |
| 인기순 추천 (baseline) | 29.53% | 5.46% |
| 메모리 기반 CF (전체 이웃) | 46.43% | 10.02% |
| **메모리 기반 CF (상위 50 이웃, 최종)** | **52.15%** | **11.35%** |
| 콘텐츠 기반 (최종 설정) | 6.72% | 1.31% |

- 협업 필터링이 가장 성능이 좋았고, 인기순 추천보다 Precision@10이 약 22%p 높았다.
- 콘텐츠 기반 추천은 무작위보다는 높지만 인기순보다 낮았다. 아티스트 단위로 평균 낸 오디오 특성만으로는 취향을 구분하기 어려웠다.
- 사용자당 test 아티스트가 평균 58.3명이라 10명만 추천하면 Recall@10은 최대 31.91%까지만 나올 수 있다.

자세한 비교와 사례 분석은 `evaluation/reports/평가_비교_보고서.pdf`에 정리했다.

## 한계

- 아티스트 이름으로만 매칭해서 Spotify에 없는 아티스트(전체 아티스트의 약 95%)는 분석에서 빠졌다. 다만 청취량 기준으로는 약 49%가 남는다.
- 시간 순서가 아닌 무작위 분할이라 "다음에 들을 아티스트"를 예측하는 상황과는 차이가 있다.
- 콘텐츠 기반의 최종 설정은 test 결과를 보고 고른 것이라 독립적으로 검증된 최적값은 아니다.

## 팀원 및 역할

| 이름 | 담당 |
|---|---|
| 서관우 | 데이터 전처리, train/test 분할, 공통 평가 함수, 인기순 baseline |
| 서연 | 메모리 기반 협업 필터링 |
| 오서영 | 콘텐츠 기반 추천 |
| 이건희 | 최종 평가 및 비교 분석 |
