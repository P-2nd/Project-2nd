# 고객 이탈 예측 프로젝트 실행 가이드

> 발표일: 2026년 7월 22일  
> 목표: 고객 이탈 가능성을 예측하고, 고위험 고객에게 적용할 유지 활동을 제안한다.

이 문서는 수업 발표 가이드와 이 저장소의 실험 규칙을 함께 적용하기 위한 팀 작업 기준이다. 정확도만 높은 모델이 아니라 **문제 정의 → 데이터 분석 → 재현 가능한 학습·평가 → 저장 모델 추론 → Streamlit 시연 → 유지 전략**이 연결된 결과물을 만든다.

## 1. 프로젝트 범위와 완료 기준

### 1.1 문제 정의

프로젝트를 시작하기 전에 다음 문장을 확정한다.

> `[사용자]`가 `[의사결정]`을 할 수 있도록 `[데이터]`로 `[Target]`을 예측하고 `[유지 활동]`을 제안한다.

`docs/requirements.md`에 아래 내용을 작성한다.

| 항목 | 결정할 내용 |
|---|---|
| 사용자 | 예측 화면과 결과를 사용할 담당자 |
| 문제 | 이탈로 발생하는 손실 또는 불편 |
| Target | 0/1의 의미와 이탈 정의 |
| 예측 시점 | 언제 이탈 위험을 예측하는지 |
| 관찰 기간 | Feature 생성에 쓰는 과거 기간 |
| 결과 기간 | Target을 판단하는 미래 기간 |
| 오류 비용 | FN과 FP 중 더 큰 비용과 이유 |
| 주요 지표 | Recall, F1, PR-AUC 중 우선 지표 |
| 화면 | 입력값, 예측 결과, 유지 활동 |
| 완료 기준 | 저장 모델 재로딩, 신규 고객 예측, 앱 시연 여부 |

이탈 고객을 놓치는 비용이 보통 더 크므로, 최종 모델 선정 시 **이탈 클래스(1)의 Recall**을 우선 보되 Precision, F1, PR-AUC를 함께 검토한다.

### 1.2 필수 산출물

1. 전처리 결과서: 출처, Target, 데이터 품질 점검, EDA, 전처리 근거
2. 모델 학습 결과서: 분할·검증 방법, 기준 모델, 비교표, 최종 Test 결과, 한계
3. 학습된 최종 Pipeline 모델과 메타데이터
4. 저장 모델을 실제 호출하는 Streamlit 시연 화면
5. 설치·실행 방법과 결과를 담은 README 및 발표 자료

## 2. 데이터 관리와 문서화

### 2.1 데이터 카드

데이터를 받은 즉시 아래 항목을 기록한다. 실제 데이터가 아니라 합성 데이터라면 생성 규칙과 실제 적용 한계를 명시한다.

| 기록 항목 | 내용 |
|---|---|
| 출처 | 정확한 URL, 다운로드 날짜, 라이선스 |
| 분석 단위 | 고객/거래/행동/일/월 중 1행의 의미 |
| 키 | 고객 ID, 날짜 등 조인 기준 |
| 규모 | 행·열 수, 메모리 사용량 |
| Target | 기존 컬럼인지, 생성 규칙은 무엇인지 |
| 개인정보 | 제거 또는 비식별화할 컬럼 |

여러 테이블을 쓸 경우에는 조인 전에 분석 단위를 먼저 정한다. 예를 들어 고객 1명을 1행으로 두고, 관찰 기간의 거래·로그를 고객별로 집계한다. 조인 전후에는 행 수, 고객 수, 중복 수, 결측률 변화를 확인한다.

### 2.2 저장소 위치

| 대상 | 위치 |
|---|---|
| 수정하지 않는 원본 | `data/raw/` |
| 변환·학습용 데이터셋 | `data/processed/` |
| 지표, 예측값, 차트, 실행 설정 | `data/evaluation/` |
| 재사용 코드 | `src/` |
| 공통 설정·하이퍼파라미터 | `configs/model_params.yaml` |
| 학습된 Pipeline·전처리기 | `models/` |
| 탐색 노트북 | `notebooks/` |
| 테스트 | `tests/` |

원본 데이터, 가상환경, API 키, 비밀번호, 개인정보는 Git에 커밋하지 않는다. 모든 경로는 프로젝트 루트 기준 상대경로를 사용한다.

## 3. EDA와 전처리 기준

`EDA.md`에는 빈 템플릿이 아닌 실제 측정 결과를 기록한다. 다음 순서로 점검한다.

```text
shape·dtype
→ 중복·결측·불가능한 값
→ Target 분포
→ 수치형·범주형별 이탈률
→ 시간·고객 그룹별 분포
→ 누수 의심 Feature
→ 핵심 인사이트와 모델링 결정
```

필수 결과는 데이터 품질 요약, 핵심 시각화 5~7개, 비즈니스 인사이트 3개, 인사이트에 따른 모델링 결정 3개, 데이터 한계 1개 이상이다. 각 그래프는 다음 형식으로 해석한다.

> `[관찰 결과]`가 보인다. 따라서 `[다음 처리 또는 실험]`이 필요하다. 다만 `[인과관계 또는 일반화]`를 단정할 수 없다.

이탈 후에 생성된 정보(해지일, 해지 사유, 사후 점수) 또는 결과 기간의 행동 정보는 Feature에서 제외한다. 이들은 Target 누수를 유발한다.

## 4. 데이터 분리와 Pipeline

### 4.1 검증 원칙

- 고객당 1행 데이터는 `stratify=y`를 적용한 Train/Validation/Test 분할 또는 Train/Test + Train 내부 교차검증을 사용한다.
- 동일 고객의 반복 기록은 고객 ID 기준 Group split/GroupKFold를 사용한다.
- 시간 데이터는 과거를 Train, 이후를 Validation, 최신 구간을 Test로 둔다.
- Test 데이터는 최종 성능 확인에만 한 번 사용한다. 모델, 하이퍼파라미터, early stopping, threshold 선택에 사용하지 않는다.
- 난수 시드는 `configs/model_params.yaml`의 `random_state`로 고정·기록한다.

### 4.2 전처리 원칙

1. 먼저 Train/Validation/Test를 분리한다.
2. 결측 처리, 인코딩, 스케일링, Feature 선택은 Train 또는 CV 학습 fold에서만 `fit`한다.
3. Validation/Test에는 `transform`만 적용한다.
4. SMOTE를 사용한다면 `imblearn.Pipeline` 내부에서 학습 fold에만 적용한다.
5. 전처리기와 최종 모델을 하나의 Pipeline으로 저장한다.

전체 데이터를 미리 인코딩·스케일링한 `churn_preprocessed.csv`를 만든 뒤 분할하는 방식은 사용하지 않는다.

## 5. 100%·50% 실험 설계

이 저장소의 기본 설정은 `training_fractions: [1.0, 0.5]`다.

1. 전체 데이터에서 Test 세트를 먼저 고정한다.
2. 남은 Train 영역에서 Validation을 분리하거나 교차검증을 수행한다.
3. 100% 실험은 고정 Test 외의 전체 학습 데이터를 사용한다.
4. 50% 실험은 같은 학습 데이터에서만 층화 표본추출로 50%를 사용한다.
5. 두 실험은 **동일한 Test 세트, 동일한 평가 코드, 동일한 threshold**로 최종 비교한다.

즉, 50% 실험을 위해 Test 세트를 다시 추출하면 안 된다.

## 6. 모델 학습·선정·평가

### 6.1 권장 비교 순서

1. `DummyClassifier` — 최소 기준선
2. `LogisticRegression` — 해석 가능한 기준 모델
3. `DecisionTreeClassifier` 또는 `RandomForestClassifier`
4. XGBoost, LightGBM, CatBoost 중 하나 이상

딥러닝(MLP)은 정형 데이터에서 선택 사항이다. 머신러닝 기준 모델을 완성한 뒤, 비선형 관계 학습 등 명확한 비교 이유가 있을 때만 추가한다.

### 6.2 공통 평가 항목

모든 모델과 데이터 규모에 대해 다음을 같은 조건으로 저장한다.

- Accuracy, Precision, Recall, F1-score
- ROC-AUC와 PR-AUC
- Confusion Matrix
- 추론 지연 시간과 전체 실행 시간
- 사용한 split, random seed, 하이퍼파라미터, threshold, 아티팩트 경로

마케팅 대상 인원이 제한되면 Lift, Top-K Capture, 위험도 구간(Decile)별 실제 이탈률도 추가로 확인한다.

임계값은 기본값 0.5를 그대로 사용하거나 Validation 성능·업무 비용을 기준으로 결정한다. 결정된 임계값을 메타데이터에 기록하고 최종 Test에서 변경하지 않는다.

`RESULT.md`는 `data/evaluation/`에 저장된 평가 결과를 바탕으로 작성한다. README, RESULT, 설정 파일, 소스 코드의 모델 목록과 수치는 항상 일치해야 한다.

## 7. 모델 저장과 추론 검증

최종 모델은 예를 들어 `models/churn_pipeline.joblib`로 저장한다. 모델과 함께 다음 메타데이터를 저장한다.

- 학습 날짜, 데이터 버전, Python·주요 패키지 버전
- Feature 이름·순서·자료형
- Target 0/1 의미
- 선택한 threshold
- Validation/Test 지표

새 프로세스에서 모델을 다시 로드해 신규 고객 1명의 입력을 예측하는 테스트를 `tests/`에 작성한다. 출력에는 이탈 확률과 threshold 기준 이탈 여부가 포함되어야 한다.

## 8. Streamlit 시연 기준

Streamlit은 저장된 모델을 로드하여 추론만 수행한다. 앱 실행 시 재학습하거나 미리 계산한 JSON 확률을 표시해서는 안 된다.

권장 화면은 다음과 같다.

1. **고객 현황**: 전체 고객 수, 이탈률, 핵심 EDA 차트, 위험도 분포
2. **모델 성능**: 모델 비교표, 최종 혼동 행렬, Precision/Recall/F1/PR-AUC, 주요 Feature 해석
3. **개별 고객 이탈 예측**: 고객 정보 입력, 이탈 확률, 위험 등급, 권장 유지 활동

입력값을 바꾸면 예측 확률이 실제로 달라져야 하며, 학습과 화면에서 Feature 이름·자료형·전처리가 같아야 한다. 모델 파일 누락이나 잘못된 입력에는 이해 가능한 오류 메시지를 제공한다.

## 9. Git 협업 규칙

### 9.1 브랜치 규칙

- `main` 브랜치에는 직접 Push하지 않는다.
- 모델·기능·문서 작업별 브랜치를 생성하고 Pull Request로 `main`에 병합한다.
- 한 브랜치에는 가능한 한 하나의 작업 목적만 포함한다.

```text
main
├── KNN-CJH
├── Logistic-CJH
├── XGBoost-SJH
├── MLP-JSH
└── Docs-CJH
```

### 9.2 커밋 규칙

커밋 메시지는 `작업 영역 - 변경 내용` 형식으로 작성한다. 변경 내용을 알 수 없는 메시지나 빈 메시지는 사용하지 않는다.

```text
KNN - 거리 가중치 하이퍼파라미터 수정
XGBoost - 상위 50% 특성 평가 결과 저장
EDA - 한글 분포 이미지 추가
README - 모델 평가 결과 설명 보완
```

### 9.3 작업 순서

1. `git switch -c 브랜치명`으로 작업 브랜치를 생성하고 이동한다.
2. `git status`로 현재 변경 파일을 확인한다.
3. 작업 완료 후 관련 테스트와 실행 결과를 확인한다.
4. `git add 작업파일`로 해당 작업에 필요한 파일만 Stage한다.
5. `git commit -m "작업 영역 - 변경 내용"`으로 커밋한다.
6. `git push origin 브랜치명`으로 원격 저장소에 Push한다.
7. 본인 브랜치에서 `main`을 대상으로 Pull Request를 생성한다.
8. 팀원 리뷰와 충돌 확인 후 Merge한다.

### 9.4 Pull Request 규칙

- 본인 작업 브랜치에서 `main`으로만 Pull Request를 생성한다.
- 다른 팀원의 작업 브랜치를 Merge 대상으로 사용하지 않는다.
- PR 설명에 작업 목적, 변경 파일, 실행·검증 결과를 작성한다.
- 데이터 경로, 모델 결과 수치, README·RESULT 문서가 서로 일치하는지 확인한다.
- 리뷰 없이 직접 Merge하지 않는다.

### 9.5 공통 협업 원칙

- 데이터, EDA·Feature, 모델링, Streamlit·통합, 문서 작업마다 담당자와 검토자를 지정한다.
- 원본 데이터, 가상환경, 대용량 재생성 모델, 비밀정보는 `.gitignore` 규칙을 따른다.
- 다른 팀원의 작업을 덮어쓰지 않도록 작업 전후 `git status`와 변경 파일을 확인한다.
- 하루 한 번 통합 브랜치에서 전체 실행 흐름을 확인한다.

## 10. 모델·데이터 파일 및 결과 저장 규칙

### 10.1 파일명 구성

파일명은 다음 자리표시자를 조합해 작성한다.

| 자리표시자 | 의미 | 값 예시 |
|---|---|---|
| `<model>` | 모델 식별자 | `knn`, `logistic`, `randomforest`, `lightgbm`, `xgboost` |
| `<experiment>` | 데이터·특성·제외 실험 구분 | `full`, `50`, `full_without_PT_Session_Count` |
| `<artifact>` | 산출물 종류 | `eval`, `params`, `results`, `roc`, `confusion_matrix` |

- 모델 식별자는 영문 소문자와 밑줄(`snake_case`)을 사용한다.
- 공백, 한글, 운영체제별 특수문자를 파일명에 사용하지 않는다.
- 같은 모델이라도 데이터 또는 특성 구성이 다르면 `<experiment>`로 구분한다.
- 특정 특성을 제외한 실험은 `<experiment>_without_<feature>` 형식을 사용한다.
- 프로젝트 내부 경로는 모두 프로젝트 루트 기준 상대경로로 저장한다.

### 10.2 파일별 위치와 형식

| 구분 | 저장 위치 | 파일명 형식 |
|---|---|---|
| 원본 데이터 | `data/raw/` | 원본 제공 파일명 유지 |
| 전처리 데이터 | `data/processed/` | `churn_preprocessed_<feature_set>.csv` |
| 모델 정의 | `src/models/` | `model_<model>.py` |
| 개별 학습 스크립트 | `src/models/` | `train_<model>.py` |
| 평가용 학습 모델 | `data/evaluation/saved_models/` | `<model>_<experiment>_eval.joblib` |
| 학습 파라미터 | `data/evaluation/saved_params/` | `<model>_<experiment>_params.json` |
| 모델별 평가 결과 | `data/results/` | `<model>_<experiment>_results.json` |
| 통합 평가 결과 | `data/results/` | `result_data.json` |
| ROC 원천 데이터 | `data/results/roc/` | `<model>_<experiment>.parquet` |
| 혼동행렬 이미지 | `data/evaluation/plots/` | `<model>_<experiment>_confusion_matrix.png` |
| ROC Curve 이미지 | `data/evaluation/plots/` | `<model>_<experiment>_roc.png` |
| 배포용 Pipeline | `models/` | `<model>_<experiment>_pipeline.joblib` |
| 모델 메타데이터 | `models/` | `<model>_<experiment>_metadata.json` |

`data/evaluation/saved_models/`의 모델은 평가 과정에서 다시 만들 수 있으므로 Git 추적에서 제외한다. Git에 포함할 파일과 재생성 가능한 대용량 산출물은 `.gitignore` 규칙을 따른다.

### 10.3 KNN 파일명 예시

아래 파일명은 KNN의 상위 50% 특성 실험을 가정한 **예시**이며, 고정 규칙이 아니다. 다른 모델도 `<model>`과 `<experiment>`만 바꿔 같은 형식을 사용한다.

```text
src/models/model_knn.py
data/evaluation/saved_models/knn_50_eval.joblib
data/evaluation/saved_params/knn_50_params.json
data/results/knn_50_results.json
data/results/roc/knn_50.parquet
data/evaluation/plots/knn_50_confusion_matrix.png
data/evaluation/plots/knn_50_roc.png
```

### 10.4 결과 JSON 저장 항목

모델별 `*_results.json`에는 다음 항목을 저장한다.

| 항목 | JSON 키 | 설명 |
|---|---|---|
| 모델 식별 정보 | `model_key`, `model_name` | 모델과 실험 조합을 식별하는 이름 |
| 실험 정보 | `experiment` | 특성 구성과 비교 기준 |
| 정확도 | `accuracy` | 전체 예측 중 정답 비율 |
| 정밀도 | `precision` | 이탈 예측 고객 중 실제 이탈 비율 |
| 재현율 | `recall` | 실제 이탈 고객 중 탐지한 비율 |
| F1 점수 | `f1_score` | Precision과 Recall의 조화평균 |
| ROC-AUC | `auc_score` | 임계값 전반의 이탈 구분 성능 |
| 혼동행렬 | `confusion_matrix` | TN, FP, FN, TP 결과 |
| 전체 추론 시간 | `total_time_sec` | Test 데이터 전체 추론 시간(초) |
| 평균 지연 시간 | `avg_latency_ms` | 회원 1명당 평균 추론 시간(ms) |
| 평가 표본 수 | `total_samples` | Test 데이터 표본 수 |
| ROC 데이터 정보 | `parquet` | Parquet 상대경로와 컬럼 정보 |

통합 결과인 `result_data.json`은 모델과 실험 라벨별 최신 결과, threshold, ROC-AUC, PR-AUC, Pipeline·메타데이터 경로를 함께 관리한다.

### 10.5 ROC Parquet 저장 항목

ROC Curve 재생성과 임계값 재분석을 위해 다음 두 컬럼을 저장한다.

| 컬럼 | 설명 |
|---|---|
| `y_true` | 실제 이탈 라벨(0 또는 1) |
| `y_score` | 모델이 예측한 이탈 확률 또는 점수 |

Accuracy, Precision, Recall, F1 Score, Confusion Matrix 등은 `y_true`, `y_score`와 적용한 threshold를 이용해 다시 계산할 수 있다.

## 11. 발표 전 최종 점검

```text
[ ] 데이터 출처·Target·예측 시점이 문서화되어 있다.
[ ] 누수 Feature를 제거했고, 전처리는 Train/CV에서만 fit했다.
[ ] Test는 모델·threshold 선택에 사용하지 않았다.
[ ] 100%·50% 실험이 같은 Test 세트를 사용한다.
[ ] 평가 산출물과 RESULT.md 수치가 일치한다.
[ ] 저장 Pipeline을 새 프로세스에서 로드해 신규 고객을 예측한다.
[ ] Streamlit이 저장 모델로 실제 확률과 유지 활동을 표시한다.
[ ] README의 설치부터 실행까지 2~3개 명령으로 재현 가능하다.
```
