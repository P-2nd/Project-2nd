"""전처리 완료 데이터로 XGBoost 이탈 예측 모델을 학습·평가·저장한다.

GUIDE 5장의 실험 설계를 따른다. Test와 Validation을 먼저 한 번만 고정하고,
`training_fractions`(기본 100%, 50%)에 따라 **학습 데이터만** 층화 표본추출로 줄인다.
두 실험은 같은 Test, 같은 Validation, 같은 평가 코드를 쓰므로 학습 데이터 양의
차이만 성능 차이로 나타난다.

입력 기본값은 `src/preprocessing/preprocessing.py`가 만드는
`data/processed/churn_preprocessed_full.csv`다. 이 파일은 이미 인코딩·스케일링이
끝나 있고, 스케일러는 `random_state=42`, `test_size=0.2`, `stratify=y` 분할의
Train에만 fit되어 있다. 따라서 이 스크립트도 **같은 분할 조건을 그대로 써야**
원래 Test 구간이 스케일러 학습에 섞이지 않는다. 설정 파일의 random_state와
test_size를 바꾸면 이 전제가 깨진다.

Pipeline의 인코딩·결측 대체 단계는 정제만 끝난 데이터가 들어와도 동작하도록
남겨 둔 안전망이며, 위 기본 입력에서는 사실상 통과 단계로 동작한다.

실행 예시:
    python src/models/train_xgboost.py
    python src/models/train_xgboost.py --input data/processed/churn_preprocessed_pct50.csv
    python src/models/train_xgboost.py --fractions 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config  # noqa: E402  경로 설정 후 임포트해야 한다.
from src.common.results import roc_data_path, upsert_result  # noqa: E402


MODEL_NAME = "XGBoost"
DEFAULT_INPUT = config.PROCESSED_DATA_DIR / "churn_preprocessed_full.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="전처리 완료 CSV 경로 (기본값: data/processed/churn_100.csv)",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=None,
        help="학습 데이터 비율. 지정하지 않으면 설정 파일의 training_fractions를 쓴다.",
    )
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    with config.MODEL_PARAMS_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_processed_frame(path: Path, target_column: str) -> pd.DataFrame:
    """정제 데이터를 읽고 학습에 필요한 최소 조건을 확인한다."""
    if not path.is_file():
        raise FileNotFoundError(
            f"전처리 완료 CSV를 찾을 수 없습니다: {path}\n"
            "src/preprocessing/preprocessing.py를 먼저 실행해 data/processed/를 채우거나 "
            "--input으로 경로를 지정하세요."
        )

    frame = pd.read_csv(path)
    if target_column not in frame:
        raise ValueError(
            f"Target 컬럼 '{target_column}'이 {path.name}에 없습니다. "
            f"현재 컬럼: {list(frame.columns)}"
        )
    return frame


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    """가입일에서 달력 Feature를 만든다. 이미 파생되어 있으면 그대로 둔다.

    가입일은 예측 시점 이전에 확정되는 값이므로 Target 누수가 아니다.
    """
    if config.DATE_COLUMN not in frame:
        return frame

    result = frame.copy()
    dates = pd.to_datetime(result[config.DATE_COLUMN], errors="coerce")
    result["membership_start_year"] = dates.dt.year
    result["membership_start_month"] = dates.dt.month
    result["membership_start_dayofweek"] = dates.dt.dayofweek
    return result.drop(columns=[config.DATE_COLUMN])


def build_features(frame: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """식별자와 Target을 제외한 입력 Feature와 라벨을 나눈다."""
    prepared = add_date_features(frame)
    drop_columns = [column for column in (target_column, config.ID_COLUMN) if column in prepared]
    X = prepared.drop(columns=drop_columns)
    y = prepared[target_column].astype(int)
    return X, y


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """트리 모델에 맞춘 전처리기를 만든다.

    XGBoost는 분할 기준으로 학습하므로 스케일링이 필요 없다. 결측 대체와 인코딩은
    전처리 단계에서 이미 끝났다고 보지만, 새 데이터가 들어와도 깨지지 않도록 안전망으로 남긴다.

    전처리 산출물의 원핫 컬럼은 bool로 저장되는데 pandas는 bool을 number로 보지
    않는다. bool을 숫자로 함께 묶지 않으면 이미 인코딩된 컬럼이 다시 OneHotEncoder를
    거쳐 컬럼이 두 배로 늘어난다.
    """
    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def build_classifier(xgboost_config: dict[str, Any], random_state: int) -> XGBClassifier:
    params = dict(xgboost_config["params"])
    scale_pos_weight = xgboost_config.get("scale_pos_weight")
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)

    return XGBClassifier(
        **params,
        random_state=random_state,
        early_stopping_rounds=int(xgboost_config["early_stopping_rounds"]),
    )


def choose_threshold(
    y_validation: pd.Series, probabilities: np.ndarray, xgboost_config: dict[str, Any]
) -> float:
    """Validation에서만 임계값을 정한다. Test로는 절대 고르지 않는다."""
    strategy = xgboost_config.get("threshold_strategy", "f1")
    if strategy == "fixed":
        return float(xgboost_config.get("threshold_value", 0.5))
    if strategy != "f1":
        raise ValueError(f"threshold_strategy는 'f1' 또는 'fixed'여야 합니다: {strategy}")

    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    if not len(thresholds):
        return 0.5

    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def evaluate(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float, elapsed_seconds: float
) -> dict[str, Any]:
    """GUIDE 6.2가 요구하는 지표를 한 번에 계산한다."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "total_samples": int(len(y_true)),
        "total_inference_seconds": elapsed_seconds,
        "average_inference_ms": elapsed_seconds / len(y_true) * 1000,
    }


def top_feature_importances(
    pipeline: Pipeline, limit: int = 20
) -> list[dict[str, float | str]]:
    """EDA.md의 Feature Importance 표에 바로 옮길 수 있도록 상위 기여 Feature를 뽑는다."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    order = np.argsort(importances)[::-1][:limit]
    return [
        {"feature": str(feature_names[index]), "importance": float(importances[index])}
        for index in order
    ]


def subsample_training_data(
    X_train: pd.DataFrame, y_train: pd.Series, fraction: float, random_state: int
) -> tuple[pd.DataFrame, pd.Series]:
    """이탈 비율을 유지한 채 학습 데이터만 fraction만큼 남긴다."""
    if fraction >= 1.0:
        return X_train, y_train

    X_subset, _, y_subset, _ = train_test_split(
        X_train,
        y_train,
        train_size=fraction,
        random_state=random_state,
        stratify=y_train,
    )
    return X_subset, y_subset


def fraction_label(fraction: float) -> str:
    """README 12장의 파일명 규칙에 쓰는 비율 표기(1.0 -> 100, 0.5 -> 50)."""
    return str(int(round(fraction * 100)))


def run_experiment(
    fraction: float,
    splits: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """한 가지 학습 비율로 학습·평가·저장을 수행하고 지표를 돌려준다."""
    label = fraction_label(fraction)
    random_state = settings["random_state"]
    xgboost_config = settings["xgboost_config"]

    X_train, y_train = subsample_training_data(
        splits["X_train"], splits["y_train"], fraction, random_state
    )
    print(f"\n=== {MODEL_NAME} {label}% 실험 ===")
    print(f"- 학습 {len(X_train):,}명 / Validation {len(splits['X_validation']):,}명 "
          f"/ Test {len(splits['X_test']):,}명")

    started_at = time.perf_counter()

    # 전처리기는 학습 데이터에만 fit한다. Validation/Test에는 transform만 적용한다.
    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(splits["X_validation"])

    # early stopping은 변환된 Validation을 필요로 하므로 모델을 먼저 학습한 뒤
    # 학습이 끝난 전처리기와 함께 하나의 Pipeline으로 묶는다.
    model = build_classifier(xgboost_config, random_state)
    model.fit(
        X_train_transformed,
        y_train,
        eval_set=[(X_validation_transformed, splits["y_validation"])],
        verbose=False,
    )
    training_seconds = time.perf_counter() - started_at

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    validation_probabilities = pipeline.predict_proba(splits["X_validation"])[:, 1]
    threshold = choose_threshold(splits["y_validation"], validation_probabilities, xgboost_config)

    inference_started_at = time.perf_counter()
    test_probabilities = pipeline.predict_proba(splits["X_test"])[:, 1]
    inference_seconds = time.perf_counter() - inference_started_at

    metrics = evaluate(splits["y_test"], test_probabilities, threshold, inference_seconds)
    total_seconds = time.perf_counter() - started_at

    metadata = {
        "model": MODEL_NAME,
        "training_fraction": fraction,
        "input_path": settings["input_path"],
        "rows": {
            "train": int(len(X_train)),
            "validation": int(len(splits["X_validation"])),
            "test": int(len(splits["X_test"])),
        },
        "features": settings["feature_columns"],
        "target_column": settings["target_column"],
        "target_meaning": {str(key): value for key, value in config.TARGET_LABELS.items()},
        "random_state": random_state,
        "split": {
            "test_size": settings["test_size"],
            "validation_size": float(xgboost_config["validation_size"]),
            "stratified": True,
            "note": "Test와 Validation은 모든 fraction에서 동일하다.",
        },
        "hyperparameters": dict(xgboost_config["params"]),
        "scale_pos_weight": xgboost_config.get("scale_pos_weight"),
        "early_stopping_rounds": int(xgboost_config["early_stopping_rounds"]),
        "best_iteration": int(getattr(model, "best_iteration", 0)),
        "threshold_strategy": xgboost_config.get("threshold_strategy", "f1"),
        "threshold_from_validation": threshold,
        "training_seconds": training_seconds,
        "total_seconds": total_seconds,
        "python_version": sys.version.split()[0],
    }

    artifacts = save_artifacts(
        label=label,
        pipeline=pipeline,
        metadata=metadata,
        metrics=metrics,
        y_test=splits["y_test"],
        test_probabilities=test_probabilities,
    )
    metadata["artifacts"] = artifacts

    print_summary(label, splits["y_test"], threshold, metrics)
    return {"fraction": fraction, "label": label, "metrics": metrics, "threshold": threshold}


def save_artifacts(
    label: str,
    pipeline: Pipeline,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    y_test: pd.Series,
    test_probabilities: np.ndarray,
) -> dict[str, str]:
    """모델·메타데이터·ROC 원천 데이터를 저장하고 통합 결과 파일을 갱신한다."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.MODELS_DIR / f"xgboost_{label}_pipeline.joblib"
    metadata_path = config.MODELS_DIR / f"xgboost_{label}_metadata.json"
    roc_path = roc_data_path("xgboost", label)

    joblib.dump(pipeline, model_path)

    artifacts = {
        "pipeline": str(model_path.relative_to(config.PROJECT_ROOT)),
        "metadata": str(metadata_path.relative_to(config.PROJECT_ROOT)),
        "result_data": str(config.RESULT_DATA_PATH.relative_to(config.PROJECT_ROOT)),
        "roc_source": str(roc_path.relative_to(config.PROJECT_ROOT)),
    }
    metadata_with_artifacts = {**metadata, "artifacts": artifacts}

    metadata_path.write_text(
        json.dumps(metadata_with_artifacts, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    roc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"y_true": y_test.to_numpy(), "y_score": test_probabilities}).to_parquet(
        roc_path, index=False
    )
    upsert_result(
        model_key="xgboost",
        model_name=MODEL_NAME,
        label=label,
        experiment={
            "comparison_axis": "training_fraction",
            "training_fraction": float(metadata["training_fraction"]),
            "feature_count": int(len(metadata["features"])),
        },
        metrics=metrics,
        threshold=float(metadata["threshold_from_validation"]),
        total_time_sec=float(metadata["total_seconds"]),
        artifacts=artifacts,
        extras={"feature_importances_top20": top_feature_importances(pipeline)},
    )
    return artifacts


def print_summary(
    label: str, y_test: pd.Series, threshold: float, metrics: dict[str, Any]
) -> None:
    """RESULT.md에 그대로 옮길 수 있도록 지표를 한글로 정리해 출력한다."""
    true_negative, false_positive = metrics["confusion_matrix"][0]
    false_negative, true_positive = metrics["confusion_matrix"][1]
    majority_baseline = max(float((y_test == 0).mean()), float((y_test == 1).mean()))

    print(f"\n[{MODEL_NAME} {label}% Test 성능]")
    print(f"- Accuracy : {metrics['accuracy']:.4f} (전체 예측 중 맞힌 비율)")
    print(f"- Precision: {metrics['precision']:.4f} (이탈 경고 고객 중 실제 이탈 비율)")
    print(f"- Recall   : {metrics['recall']:.4f} (실제 이탈 고객 중 찾아낸 비율)")
    print(f"- F1 Score : {metrics['f1']:.4f} (Precision과 Recall의 균형)")
    print(f"- ROC-AUC  : {metrics['roc_auc']:.4f} (이탈·유지 고객을 구분하는 순위 성능)")
    print(f"- PR-AUC   : {metrics['pr_auc']:.4f} (이탈 고객 탐지 성능)")
    print(f"- Latency  : {metrics['average_inference_ms']:.4f} ms/명")
    print(f"- 임계값   : {threshold:.4f} (Validation에서 선택)")

    print(f"\n[{label}% 혼동 행렬]")
    print(f"- 실제 유지 → 유지 예측  : {true_negative:,}명")
    print(f"- 실제 유지 → 이탈 오경고: {false_positive:,}명")
    print(f"- 실제 이탈 → 유지로 놓침: {false_negative:,}명")
    print(f"- 실제 이탈 → 이탈 예측  : {true_positive:,}명")
    print(
        f"- 다수 클래스만 예측하는 단순 기준선 Accuracy {majority_baseline * 100:.1f}% 대비 "
        f"현재 {metrics['accuracy'] * 100:.1f}%"
    )


def print_comparison(results: list[dict[str, Any]]) -> None:
    """RESULT.md의 '데이터셋별 성능 비교' 표에 바로 넣을 수 있는 형태로 출력한다."""
    print(f"\n=== {MODEL_NAME} 학습 데이터 비율별 비교 (동일 Test 세트) ===")
    print("| Dataset   | Accuracy | Precision | Recall | F1 Score | ROC AUC | Latency (ms) |")
    print("| --------- | -------: | --------: | -----: | -------: | ------: | -----------: |")
    for result in results:
        metrics = result["metrics"]
        print(
            f"| churn_{result['label']:<4}| {metrics['accuracy']:.4f}   | {metrics['precision']:.4f}    "
            f"| {metrics['recall']:.4f} | {metrics['f1']:.4f}   | {metrics['roc_auc']:.4f}  "
            f"| {metrics['average_inference_ms']:.4f}       |"
        )


def main() -> None:
    args = parse_args()
    settings_file = load_config()
    xgboost_config = settings_file["xgboost"]

    random_state = int(settings_file["random_state"])
    target_column = settings_file["target_column"]
    test_size = float(settings_file["test_size"])
    fractions = args.fractions or [float(value) for value in settings_file["training_fractions"]]

    frame = load_processed_frame(args.input, target_column)
    X, y = build_features(frame, target_column)
    print(f"[입력] {args.input} — {len(frame):,}행, Feature {X.shape[1]}개")
    print(f"[이탈 분포] 유지 {(y == 0).mean():.1%} / 이탈 {(y == 1).mean():.1%}")

    # Test를 먼저 고정한다(GUIDE 5장). 모든 fraction이 이 Test로만 비교된다.
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Validation도 한 번만 고정해 임계값 선택과 early stopping 조건을 같게 맞춘다.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=float(xgboost_config["validation_size"]),
        random_state=random_state,
        stratify=y_train_validation,
    )

    splits = {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
    }
    settings = {
        "random_state": random_state,
        "target_column": target_column,
        "test_size": test_size,
        "xgboost_config": xgboost_config,
        "feature_columns": X.columns.tolist(),
        "input_path": str(args.input),
    }

    results = [run_experiment(fraction, splits, settings) for fraction in sorted(fractions, reverse=True)]
    print_comparison(results)


if __name__ == "__main__":
    main()
