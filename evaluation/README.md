# 아티스트 추천 시스템 평가

아이템 기반 협업 필터링과 콘텐츠 기반 필터링의 최종 추천 파일을 같은 test로 평가합니다. 인기도순 추천은 두 방식이 단순 인기순보다 나은지 확인하는 비교 기준입니다.

| 방식 | Precision@10 | Recall@10 | 카탈로그 커버리지 |
|---|---:|---:|---:|
| 인기도순 추천 비교 기준 | 29.53% | 5.46% | 1.41% |
| 아이템 기반 협업 이웃 50명 | 52.15% | 11.35% | 22.64% |
| 콘텐츠 s_log + cosine_learned | 6.72% | 1.31% | 41.33% |

Precision과 Recall은 사용자 973명 각각에서 계산한 값의 평균입니다. 커버리지는 전체 사용자 Top 10에 한 번 이상 등장한 고유 아티스트 수를 후보 4,824명으로 나눈 값입니다. 개인별 다양성이나 만족도를 의미하지 않습니다.

## 보고서와 결과

- [수정 한글 PDF](reports/평가_비교_보고서.pdf)
- [편집용 Word](reports/평가_비교_보고서.docx)
- [최종 지표와 커버리지](results/final_metrics.csv)
- [최종 입력 해시 및 평가 메타데이터](results/final_evaluation_metadata.json)
- [기존 콘텐츠 전체 설정표](results/content_all_settings.csv)
- [기존 협업 설정별 평가표](results/memory_cf_metrics.csv)
- [기존 실험 재계산 기록](results/recomputed_metrics.csv)

## 재실행

Python 3.9 이상과 pandas가 필요합니다. 저장소 루트에서 실행합니다.

```bash
python evaluation/run_evaluation.py
```

별도 ZIP 없이 저장소의 train/test, 인기도순 추천, 협업 기본 및 50명 추천, 최종 콘텐츠 추천 파일을 읽습니다. 모델 재학습 없이 공통 evaluate.py로 평가합니다. 기본 실행은 results/final_metrics.csv와 results/final_evaluation_metadata.json을 갱신합니다.

과거 콘텐츠 실험도 재평가하려면 담당자로부터 받은 로컬 ZIP을 지정합니다.

```bash
python evaluation/run_evaluation.py --content-zip /path/to/content_based_result.zip
```

추가 실험 결과는 results/content_experiment_metrics.csv에 저장하며, 기존 원본 평가표를 덮어쓰지 않습니다. 사용자별 원본 추천 파일과 ZIP을 새로 업로드하지 않습니다.

## 조건과 한계

- 사용자별 무작위 약 80 대 20 분할이며, train 224,925행, test 56,727행입니다.
- 각 사용자에게 train에서 이미 들은 아티스트를 제외하고 10명을 추천합니다.
- 사용자–아티스트 중복, train/test 중복, 추천 사용자 및 후보 범위, 순위와 추천 수를 확인합니다.
- 인기도순 추천은 기존 03_split.ipynb의 청취 사용자 수 기반 방식입니다.
- 협업 50명은 train 내부 검증에서 선택했습니다. 콘텐츠는 test 실험 결과를 참고해 선택했으므로 독립적으로 검증한 최적 설정이라고 주장하지 않습니다.
- 콘텐츠 최종 파일은 8e01429에서 추가됐으며 기존 s_log 실험의 cosine_learned와 추천 순서 및 점수가 일치합니다.
- 기본 재실행은 전체 이웃 협업도 보조 비교로 포함합니다. 주요 비교는 최종 두 필터링과 인기도순 비교 기준입니다.
- 지표는 숨긴 청취 이력 복원 정도입니다. 미적중을 비선호로 해석하지 않고, 미래 청취나 실제 만족도를 나타낸다고 주장하지 않습니다.
