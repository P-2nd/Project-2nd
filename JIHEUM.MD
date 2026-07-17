# 이탈 예측 모델 실행·분석 노트

이 문서는 프로젝트를 새로 클론한 뒤 KNN, Logistic Regression, XGBoost 실험을
다시 실행하고 결과를 확인하기 위한 안내입니다. 모든 명령은 **프로젝트 루트**에서
실행합니다.

## 1. 최초 실행 준비

### 1) 가상환경과 패키지 설치

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2) 전처리 데이터 준비

대용량 CSV는 Git에 포함되지 않습니다. 아래 두 파일을 준비해야 KNN·Logistic
Regression을 바로 실행할 수 있습니다.

```text
data/processed/churn_preprocessed_full.csv
data/processed/churn_preprocessed_pct50.csv
```

다운로드 위치와 링크는 [README.md](README.md)의 **데이터 구성** 항목을 따른다.

- `full`: 전체 피처를 포함한 전처리 데이터
- `pct50`: 중요도 상위 50% 피처만 남긴 전처리 데이터

원본 데이터까지 사용해 전처리 과정을 다시 만들 경우에는 다음 파일도 필요하다.

```text
data/raw/gym_churn_1M_dataset.csv
```

## 2. 모델 실행

### KNN

```bash
# 전체 피처(100%)
python src/models/train_knn_classifier.py --dataset full

# 중요도 상위 50% 피처
python src/models/train_knn_classifier.py --dataset pct50
```

### Logistic Regression

```bash
# 전체 피처(100%)
python src/models/train_logistic_regression.py --dataset full

# 중요도 상위 50% 피처
python src/models/train_logistic_regression.py --dataset pct50
```

### XGBoost

```bash
# 학습 데이터 100%, 50%를 같은 Test/Validation 데이터로 비교
python src/models/train_xgboost.py --fractions 1.0 0.5
```

`--max-rows 30000`처럼 옵션을 주면 KNN·Logistic Regression의 표본 크기를 바꿀 수
있다. 옵션을 생략하면 `configs/model_params.yaml`의 값을 사용한다.

## 3. 결과 파일 위치

```text
data/results/
├── result_data.json             # 모든 모델의 최신 성능을 통합한 파일
└── roc/
    ├── knn_100_auc_roc.parquet
    ├── logistic_regression_100_auc_roc.parquet
    └── xgboost_100_auc_roc.parquet

models/
├── knn_100_pipeline.joblib
├── logistic_regression_100_pipeline.joblib
└── xgboost_100_pipeline.joblib
```

- `data/results/result_data.json`에는 `KNN`, `LogisticRegression`, `XGBoost`의
  `100`/`50` 실험 결과가 함께 저장된다.
- 같은 모델·라벨을 다시 실행하면 해당 결과만 최신 실행값으로 교체된다.
- ROC Parquet에는 그래프 재생성에 필요한 `y_true`(실제값), `y_score`(이탈 확률)가
  저장된다.

## 4. 100과 50의 의미

모든 모델에서 숫자는 같지만 의미가 완전히 같지는 않다.

| 모델 | 100 | 50 |
|---|---|---|
| KNN | 전체 피처 | 중요도 상위 50% 피처 |
| Logistic Regression | 전체 피처 | 중요도 상위 50% 피처 |
| XGBoost | 학습 데이터 100% | 학습 데이터 50% |

따라서 현재 XGBoost의 50은 **피처를 절반으로 줄인 실험이 아니라 학습 행 수를
절반으로 줄인 실험**이다. 세 모델을 최종 비교할 때는 동일한 피처와 동일한
Train/Validation/Test 분할로 다시 맞추는 것이 가장 공정하다.

## 5. F1 임계값과 Accuracy

KNN과 Logistic Regression은 Validation 데이터에서 F1이 가장 높은 임계값을 자동으로
선택한다.

```python
threshold = choose_f1_threshold(y_validation, validation_probabilities)
```

낮은 임계값은 실제 이탈 고객을 더 많이 찾아 Recall을 높이지만, 유지 고객을 이탈로
잘못 경고하는 수가 늘어 Accuracy와 Precision은 낮아질 수 있다.

Accuracy를 우선하려면 위 코드를 다음처럼 바꿀 수 있다.

```python
threshold = 0.5
```

단, 이 경우 Recall과 F1이 낮아질 수 있으므로 쿠폰·전화·메시지 등 이탈 방지 행동의
비용을 고려해 임계값을 정한다.

## 6. 피처 제외 위치

현재 실험 코드의 피처 제외 설정은 다음과 같다.

```python
# src/models/train_knn_classifier.py
X = frame.drop(columns=[target_column, "PT_Session_Count"])

# src/models/train_logistic_regression.py
X = frame.drop(columns=[target_column, "Late_Payment_Count"])
```

피처를 바꾼 뒤에는 해당 모델의 `full`과 `pct50`을 다시 실행해야 모델 파일과
`result_data.json`이 현재 코드와 일치한다.

## 7. Logistic Regression 설정 위치

반복 횟수 상한은 `configs/model_params.yaml`에서 관리한다.

```yaml
logistic_regression:
  C: 1.0
  class_weight: balanced
  max_iter: 1000
  solver: lbfgs
```

- `max_iter`: 최적화 계산의 최대 반복 횟수
- `C`: 값이 클수록 규제가 약해짐
- `class_weight: balanced`: 이탈/유지 클래스 비율 차이를 보정

`src/config.py`의 `TRAIN_LOOP_COUNT`는 현재 Logistic Regression 학습에는 사용되지
않는다.

## 8. 모델 선택 가이드

현재 실험에서는 XGBoost가 F1·ROC-AUC·PR-AUC 측면에서 가장 좋은 최종 후보이다.

- **XGBoost**: 최종 후보. 비선형 관계를 잘 다루고 대용량 예측에도 유리하다.
- **Logistic Regression**: 피처별 계수를 설명하기 쉬운 기준 모델이다.
- **KNN**: Recall이 높을 수 있지만 오경고가 많고 대용량 추론 비용이 커질 수 있다.

추가 비교 모델로는 다음 순서를 추천한다.

1. **CatBoostClassifier**: 원본 범주형 피처를 직접 처리하는 트리 부스팅 모델
2. **LightGBM (LGBMClassifier)**: XGBoost와 비교하기 좋은 빠른 트리 부스팅 모델
3. **HistGradientBoostingClassifier**: scikit-learn만으로 만드는 부스팅 기준 모델

CatBoost는 범주형 값을 직접 처리하는 방식이 장점이므로, 원-핫 인코딩된 CSV보다
결측 처리·불필요 컬럼 제거만 한 데이터로 별도 실험하는 편이 좋다.

## 9. 실행 전 빠른 확인

```bash
# 명령 옵션만 확인하며 학습은 하지 않음
python src/models/train_knn_classifier.py --help
python src/models/train_logistic_regression.py --help
python src/models/train_xgboost.py --help

# 통합 결과 확인
cat data/results/result_data.json
```
