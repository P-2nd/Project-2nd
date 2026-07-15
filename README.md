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
Project_2/
│
├── configs/
│   └── model_params.yaml        # 실험 공통 설정 및 모델별 하이퍼파라미터
├── data/
│   ├── raw/                     # 원본 데이터
│   ├── processed/               # 전처리 데이터셋
│   ├── eda/               		 # eda 데이터
│   └── evaluation/              # 평가 지표, 예측값, 그래프
├── models/                       # 학습된 모델과 전처리기
├── notebooks/                    # 탐색용 노트북
├── skills/
│   └── churn-ml-project/
│       ├── SKILL.md             # ML 실험 작업 규칙
│       └── agents/openai.yaml   # Codex 스킬 표시 설정
├── src/
│   ├── common/                  # 공통 데이터·설정·저장 유틸리티
│   ├── preprocessing/           # 정제, 인코딩, 피처 엔지니어링
│   ├── models/                  # 모델별 학습 코드
│   └── evaluation/              # 평가지표와 시각화 코드
├── tests/
│
├── README.md
├── EDA.md
├── RESULT.md
├── SKILLS.md                    # 프로젝트 로컬 스킬 안내
├── .gitignore
└── requirements.txt             # 의존성 목록(추가 예정)
```

---

## 4. 개발 환경

| 항목 | 버전 |
|------|------|
| Python | |
| pandas | |
| numpy | |
| scikit-learn | |
| matplotlib | |
| seaborn | |
| pyarrow | |
----------

## 5. 설치 및 실행 방법

### 필요 라이브러리 설치
- pip install -r requirements.txt

### 결과 확인
- 평가 결과(JSON): data/eval/
- ROC Curve 및 Confusion Matrix: data/plots/
- 모델 성능 비교: RESULT.md


---

6. ## 데이터 구성

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

| 항목 | 값 |
|------|----|
| Train | 80% |
| Test | 20% |
| Random State | |
| Cross Validation | |

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


## 12. 모델 및 데이터 파일명 규칙

| 구분         | 위치              | 파일명 규칙                                                                              | 예시                                                 |
|------------|-----------------|-------------------------------------------------------------------------------------|----------------------------------------------------|
| 분류 데이터     | `../data/`      | `churn_100'.csv`<br/>`churn_50'.csv`                                                |                                                    |
| 모델 파일      | `./models/`     | `모델명.py`                                                                            | `DecisionTransformer.py`                           |
| 평가 결과      | `../data/eval/` | `모델명_100_eval.json`<br/>`모델명_50_eval.json`                                    | `DecisionTransformer_100_eval.json`                |
| AUC-ROC 결과 | `../data/eval/` | `모델명_100_auc_roc.parquet`<br/>`모델명_50_auc_roc.parquet`                         | `DecisionTransformer_100_auc_roc.parquet` |

#### 파일명 형식

* `모델명`
* `모델명_100_eval.json`
* `모델명_100_auc_roc.parquet`

  * `100` : 데이터 100%
  * `50` : 데이터 50%

#### 경로 규칙

* 모델 파일은 `./models/`에 생성합니다.
* 모델에서 데이터를 사용할 경우 `../data` 경로를 사용합니다.
* 평가 결과(`eval.json`)와 성능 결과(`auc_roc.parquet`)는 모두 `../data/eval`에 저장합니다.

### 데이터 저장
* 실행 시간 (`total_time_sec`)
* 평균 지연 시간 (`avg_latency_ms`)
* 샘플 수 (`total_samples`)
* Confusion Matrix
* AUC Score

#### Parquet 저장 내용

* `y_true`
* `y_score`

#### 저장 코드 예시

```python
# 속도 계산
total_time
avg_latency_ms = (total_time / len(y_true)) * 1000

# 컨퓨전 매트릭스 계산 (임계값 0.5 기준)
y_pred = (y_scores >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
# [주의] sklearn confusion_matrix는 [[TN, FP], [FN, TP]] 순서입니다.
cm_list = cm.tolist() # JSON 저장을 위해 넘파이 배열을 파이썬 리스트로 변환

# AUC 수치 계산
auc_value = roc_auc_score(y_true, y_scores)

# 파케이는 AUC 그래프 그리기용으로 원천 데이터만 저장
df_roc = pd.DataFrame({"y_true": y_true, "y_score": y_scores})

summary_data = {
    "performance": {
        "total_time_sec": total_time,
        "avg_latency_ms": avg_latency_ms,
        "total_samples": len(y_true)
    },
    "confusion_matrix": cm.tolist(),
    "auc_score": auc_value
}

model_name = "DecisionTransformer" 
percent = "100"
save_path = "../data" 

with open(f"{save_path}/{model_name}_{percent}_eval.json", "w", encoding="utf-8") as f: 
  json.dump(summary_data, f, indent=4, ensure_ascii=False) 
  
df_roc.to_parquet( 
  f"{save_path}/{model_name}_auc_roc.parquet", index=False )
```
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






