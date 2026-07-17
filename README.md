# 프로젝트 제목

프로젝트 소개

목차

1. 프로젝트 개요
2. 팀 소개
3. 프로젝트 구조
4. 개발 환경
5. 설치 및 실행 방법
6. 데이터 구성
7. 데이터 전처리 (EDA.md)
8. 모델 목록 및 학습
9. 평가 지표
10. 결과 (RESULT.md)
11. Git 협업 규칙
12. 파일명 규칙
13. 향후 개선 사항
14. 회고


---------------------------------------

## 1. 프로젝트 개요

* **🎡프로젝트 명**
    : 헬스장 회원 이탈(Churn) 예측 프로젝트

* **📃프로젝트 소개**
    : 100만 건 규모의 헬스장 회원 데이터를 기반으로, 회원의 인구통계·이용 패턴·행동 이력 데이터를 분석하여 이탈(Churn) 여부를 예측하는 프로젝트입니다. EDA를 통해 이탈에 실질적으로 영향을 주는 변수(연체 건수, 방문 빈도, 참여도 지표 등)와 영향이 없는 변수(성별, 요금제, 선호 시간대 등)를 구분하고, 데이터 전처리(결측치 처리, 인코딩, 스케일링) 및 Feature Selection을 거쳐 여러 머신러닝/딥러닝 모델(Logistic Regression, XGBoost, Deep Learning 등)의 성능을 비교합니다.

* **🎯프로젝트 목표**
    - 데이터 정제 및 분석
        - 결측치 처리, 불필요 컬럼 제거, 파생변수 생성(가입일 기반), 인코딩, 스케일링
        - EDA를 통한 변수별 Churn 영향력 분석 (범주형 교차분석, 수치형 상관관계 분석)
        - XGBoost Feature Importance 기반 변수 선택
    - 머신러닝 모델 선택
        - Logistic Regression, XGBoost, Deep Learning 등 다중 모델 비교
        - 클래스 불균형(61.43% : 38.57%)을 고려한 평가 지표 설정
    - 결과 확인 및 분석
        - 모델별 성능 비교 (F1-score, Recall, ROC-AUC 등)
        - 이탈에 영향을 미치는 핵심 요인 도출 및 비즈니스 인사이트 정리

----

## 2. 팀 소개 - 팀
* **🧑‍🔬 팀명:** **버 -** 이유
* **👥 팀원:** 
* **🔗 멤버 개인 깃허브 계정 연동**<br>
	- 오호민: necknam (gom532454@gmail.com) <br>
	- 오호민: necknam (gom532454@gmail.com)<br>
	- 오호민: necknam (gom532454@gmail.com)<br>
	- 오호민: necknam (gom532454@gmail.com)<br>
	- 오호민: necknam (gom532454@gmail.com)

---
## 3. 프로젝트 구조

```text
Project-2nd/
├── .env.example
├── .gitignore
├── .python-version
├── AGENTS.md
├── configs/
│   └── model_params.yaml
├── data/
│   ├── eda/
│   │   ├── boxplot.png
│   │   ├── ...
│   │   └── numeric_distribution.png
│   ├── evaluation/                       # 모델 평가 산출물
│   │   ├── .gitkeep
│   │   ├── plots/                        # ROC Curve, Confusion Matrix 이미지
│   │   │   ├── knn_50_confusion_matrix.png
│   │   │   ├── ...
│   │   │   └── xgboost_full_without_Late_Payment_Count_roc.png
│   │   ├── saved_models/                 # 평가 시점 학습 모델(joblib)
│   │   │   ├── knn_50_eval.joblib
│   │   │   ├── ...
│   │   │   └── xgboost_full_without_Late_Payment_Count_eval.joblib
│   │   └── saved_params/                 # 평가에 사용된 하이퍼파라미터(JSON)
│   │       ├── knn_50_params.json
│   │       ├── ...
│   │       └── xgboost_full_without_Late_Payment_Count_params.json
│   ├── model_list.json                   # Viewer용 모델/Feature 목록
│   ├── processed/
│   │   ├── .gitkeep
│   │   ├── churn_preprocessed_full.csv
│   │   └── churn_preprocessed_pct50.csv
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── gym_churn_1M_dataset.csv
│   └── results/                          # 모델 평가 결과(JSON), ROC 데이터
│       ├── knn_50_results.json			 # 전체 실험 결과 통합본
│       ├── ...
│       ├── xgboost_50_results.json
│       └── roc/                          # ROC 생성용 Parquet(y_true, y_score)
│           ├── knn_100.parquet
│           ├── ...
│           └── xgboost_full_without_Late_Payment_Count_without_Late_Payment_Count.parquet
├── EDA.md
├── GUIDE.md
├── models/                               # 최종 배포용 모델(파이프라인)
│   ├── .gitkeep
│   ├── knn_100_pipeline.joblib
│   ├── ...
│   └── xgboost_50_pipeline.joblib
├── notebooks/
│   ├── .gitkeep
│   └── 01_EDA.ipynb
├── project_structure.py
├── README.md
├── requirements.txt
├── RESULT.md
├── skills/
│   └── churn-ml-project/
│       ├── agents/
│       │   └── openai.yaml
│       └── SKILL.md
├── SKILLS.md
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   └── results.py                    # 결과 저장/로드 공통 함수
│   ├── config.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── runtime_environment.py
│   │   └── setup_python312_env.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluation_plot.py            # ROC/Confusion Matrix 저장
│   │   └── model_eval.py                 # 모델 학습·평가 수행
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model_knn.py
│   │   ├── model_lightgbm.py
│   │   ├── model_logistic.py
│   │   ├── model_randomforest.py
│   │   └── model_xgboost.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   └── components/                       # Viewer(Streamlit) UI 구성요소
│       ├── sidebar.py
│       ├── single_view.py                # 단일 모델 결과 조회
│       └── compare_view.py               # 다중 모델 비교
├── utils.py
├── run.py                                # Streamlit 실행 엔트리포인트
└── Viewer.py				  # 결과 확인
```


---

## 4. 개발 환경

| 항목 | 버전 |
|------|------|
| Python | 3.12 |
| pandas | 2.2 이상 |
| numpy | Intel Mac: 1.26 이상 2 미만, 그 외: 2.1 이상 |
| scikit-learn | 1.6 이상 |
| matplotlib | 3.9 이상 |
| seaborn | 0.13 이상 |
| pyarrow | 17 이상 |
----------

## 5. 설치 및 실행 방법

### 가상환경과 필요 라이브러리 설치

프로젝트 루트에서 Python 3.12로 아래 명령을 실행합니다.

```bash
python src/environment/setup_python312_env.py
```

기존 `.venv`가 Python 3.12가 아니라면 다음 명령으로 다시 만듭니다.

```bash
python src/environment/setup_python312_env.py --recreate
```

macOS에서 XGBoost용 OpenMP 런타임이 아직 없다면 아래 명령으로 Homebrew의 `libomp`까지 설치합니다.

```bash
python src/environment/setup_python312_env.py --install-system-dependencies
```

스크립트는 가상환경 생성, `requirements.txt` 설치, `pip check`, 패키지 import 검증을 순서대로 수행합니다.

| 환경 | 주요 의존성 분기 | 실행 장치 |
|------|----------------|----------|
| Intel Mac | NumPy 1.x, PyTorch 2.2.x, Numba 0.62.x와 llvmlite 0.45.x, Homebrew libomp | CPU |
| Apple Silicon Mac | NumPy 2.x, PyTorch 2.6 이상, Homebrew libomp | MPS 또는 CPU |
| Windows | NumPy 2.x, PyTorch 2.6 이상 | CUDA 가능 시 CUDA, 그 외 CPU |

`requirements.txt`의 플랫폼 마커가 각 환경에 맞는 버전을 자동으로 선택하므로, 운영체제별로 별도 requirements 파일을 사용할 필요는 없습니다.

### 결과 확인
- 평가 결과(JSON): data/eval/
- ROC Curve 및 Confusion Matrix: data/plots/
- 모델 성능 비교: RESULT.md

### 필수
Viewer와 Streamlit 실행 전, 아래 명령어를 실행하여 결과 데이터를 생성합니다.

python src/evaluation/model_eval.py

---

## 6. 데이터 구성

대용량 CSV 파일은 저장소 용량을 고려해 Git 추적에서 제외되어 있습니다. 아래 링크에서
다운로드한 뒤, 표의 경로와 파일명으로 저장하세요.

| 데이터 | 다운로드 | 저장 경로 |
|--------|----------|-----------|
| 전체 원본 데이터 | [Google Drive](https://drive.google.com/file/d/1NLmsnvaG223c5zYAuNXj0Bwf9lFvJOOp/view?usp=drive_link) | `data/processed/churn_preprocessed_full.csv` |
| 50% 전처리 데이터 | [Google Drive](https://drive.google.com/file/d/1aOtOPiCOc3eQxbQjzk5Po4067bMoYON9/view?usp=drive_link) | `data/processed/churn_preprocessed_pct50.csv` |

`data/raw/` 및 `data/processed/`의 CSV는 `.gitignore` 규칙으로 인해 커밋되지 않습니다.

---

7. ## 데이터 전처리

- 결측치 처리
- 중복 데이터 제거
- 이상치 처리
- 컬럼 삭제
- 인코딩
- 스케일링
- Feature Selection
- 데이터 분할

> 자세한 내용은 `EDA.md`를 참고하세요.

```text
정제(Cleaning) → 인코딩(Encoding) → 스케일링(Scaling) 

그림 추가 예정

'https://app.diagrams.net/'

추천 설정
파일 형식: .drawio (원본 저장)
README 삽입용: .svg (가장 추천) 또는 .png
저장 위치: ./images/

README에서는 이렇게 넣으면 됩니다.

## 데이터 전처리

<p align="center">
  <img src="./images/preprocessing_flow.svg" width="800">
</p>

팁
SVG로 저장하면 확대해도 깨지지 않습니다.
images/ 폴더를 따로 만들어 관리하면 README와 EDA.md에서 같은 이미지를 재사용할 수 있습니다.
도형 색상은 너무 화려하게 하기보다 흰색 배경 + 검은 테두리 + 파란색 강조 정도가 GitHub에서 가장 보기 좋습니다.

프로젝트 규모를 보면 전처리 → 모델 학습 → 평가까지 여러 다이어그램이 생길 가능성이 있으니, 
처음부터 images/ 폴더를 만들어 관리하는 것을 추천합니다.
```

---

## 8. 모델 목록 및 학습

### 모델 목록

| 모델 | 담당자 |
|------|--------|
| Logistic Regression | 최지흠 |
| XGBoost | 신진호 |
| ML | 주상현 |

### 모델 학습

| 항목 | 값   |
|------|-----|
| Train | 80% |
| Test | 20% |
| Random State | 42  |
| Cross Validation | .   |

---

## 9. 평가 지표

| Metric | 설명 |
|--------|------|
| Accuracy | 정확도 |
| Precision | 정밀도 |
| Recall | 재현율 |
| F1 Score | F1 점수 |
| ROC AUC | ROC-AUC |
| Confusion Matrix | 혼동 행렬 |
| Latency | 평균 추론 시간 |
| Total Time | 전체 실행 시간 |

---

## 10. 결과

> 자세한 실험 결과는 `RESULT.md`를 참고하세요.

```text
비교 plot 추가 예정

```

----

## 11. Git 협업 규칙

### 브랜치 규칙

* `main` 브랜치에는 직접 Push하지 않습니다.
* 기능별 브랜치를 생성하여 작업합니다.

```git
branch

main
└── 모델명-이니셜

예시
├── RandomForest-OHM
├── XGBoost-JSH
├── LightGBM-KSY
└── CatBoost-LHJ
```
### 커밋 내용 규칙: 모델명 - 내용 (커밋 내용은 무조건 작성해주시기 바랍니다.)
```text
* 'RandomForest - 모델 생성'
* 'XGBoost - 모델 하이퍼파라미터 수정'
* 'XGBoost - 100% 데이터 저장'
```

### 작업 순서

1. `git switch -c 브랜치명` (브랜치 생성 및 이동)
2. `git status`
3. `git add .` 또는 `git add '작업 파일'`
4. `git commit -m "작업 내용"`
5. `git push origin 브랜치명`
6. Pull Request(PR) 생성
7. 리뷰 후 Merge


### Pull Request 규칙

* 본인 브랜치에서 `main`으로만 PR을 생성합니다.
* 다른 팀원의 브랜치로 PR을 생성하지 않습니다.
* 리뷰 없이 직접 Merge하지 않습니다.

---

## 12. 모델 및 데이터 파일명

| 구분               | 위치                                       | 예시                                                               |
|------------------|------------------------------------------|------------------------------------------------------------------|
| 정제 데이터           | `data/processed/`            | `churn_preprocessed_full.csv`                                    |
| 모델 파일            | `src/models/`                | `model_knn.py`                                                   |
| 학습 eval 저장       | `data/evaluation/saved_models/` | `knn_50_eval.joblib` (원래는 `models/`에 하려 했으나 이미 저장함 옮기고 코드 수정할 예정) |
| 학습 파라미터 저장       | `data/evaluation/saved_params/` | `knn_50_params.json`                                             |
| 통합 평가 결과 JSON    | `data/results/`              | `knn_50_results.json`                                            |
| 통합 평가 결과 PARQUET | `data/results/roc/`          | `randomforest_50_100.parquet`                                    |
| 통합 평가 결과 plot    | `data/evaluation/plots/`     | `knn_50_roc.png`                                                 |
| 모델 학습 및 평가 모델    | `src/evaluation/`  | `model_eval.py`                                                  |

#### 파일명 형식

* `result_data.json`
* `모델명_실험라벨_auc_roc.parquet`

  * KNN·Logistic Regression의 `100`/`50`: 전체 피처 / 중요도 상위 50% 피처
  * XGBoost의 `100`/`50`: 학습 데이터 100% / 50%

#### 경로 규칙

* 모델 파일과 메타데이터는 `models/`에 생성합니다.
* 모든 모델의 최신 평가 결과는 `data/results/result_data.json`에 통합 저장합니다.
* ROC 그래프용 원천 데이터만 `data/results/roc/`에 모델·실험 라벨별 Parquet로 저장합니다.

## 결과 저장 (Result Logging)

### 성능 지표 (JSON 등)

| 항목 | 키 | 설명 |
|---|---|---|
| 실행 시간 | `total_time_sec` | 전체 추론/실행에 소요된 시간 (초) |
| 평균 지연 시간 | `avg_latency_ms` | 샘플 1건당 평균 지연 시간 (ms) |
| 샘플 수 | `total_samples` | 평가에 사용된 전체 샘플 개수 |
| Confusion Matrix | `confusion_matrix` | 실제값 vs 예측값 분류 행렬 |
| AUC Score | `auc_score` | ROC 곡선 아래 면적 |
| Accuracy | `accuracy` | 정확도 (`metrics["accuracy"]`) |
| Precision | `precision` | 정밀도 (`metrics["precision"]`) |
| Recall | `recall` | 재현율 (`metrics["recall"]`) |
| F1 Score | `f1_score` | F1 스코어 (`metrics["f1_score"]`) |

### Parquet 저장 내용

원시 예측 결과는 별도 Parquet 파일로 저장하며, 이후 재분석/재현을 위해 다음 컬럼을 포함합니다.

| 컬럼 | 설명 |
|---|---|
| `y_true` | 실제 라벨 (ground truth) |
| `y_score` | 모델 예측 점수 (확률 또는 로짓) |

> 참고: `confusion_matrix`, `accuracy`, `precision`, `recall`, `f1_score`, `auc_score` 등은 위 Parquet의 `y_true`, `y_score`로부터 재계산이 가능합니다.

---


## 13. 향후 개선 사항

- 하이퍼파라미터 최적화
- 추가 모델 비교
- 결측치 처리 방법에 따른 성능 변화 확인

---

## 14. 회고

### 오호민 (전체 구조 구상, ERD 구성, Streamlit 구조 설계)
- 프로젝트 전체 아키텍처를 설계하며 데이터 수집, 저장, 조회, 시각화까지의 흐름을 체계적으로 구성하는 경험을 할 수 있었습니다.
- Streamlit 구조를 모듈화하고 컴포넌트를 재사용할 수 있도록 설계하면서 유지보수성과 확장성을 고려한 개발의 중요성을 배웠습니다.
- 모바일 환경에서도 편리하게 사용할 수 있도록 UI/UX를 개선하며 사용자 중심 설계 경험을 쌓을 수 있었습니다.

## Model 비교 및 Streamlit 실행

```bash
# Viewer 및 Streamlit에서 사용할 결과 데이터 생성
python src/evaluation/model_eval.py

# Streamlit 실행
streamlit run run.py

# 모델 결과 확인
python Viewer.py
```

## fork 테스트