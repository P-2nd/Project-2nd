"""작은 표본으로 KNN 분류 기반 이탈 예측 흐름을 확인하는 모델.

KNN은 100만 행 전체 데이터에서 예측 비용이 매우 크므로, 이 스크립트는
기본적으로 3만 행의 균등 표본만 사용한다. 최종 모델을 고르기 위한 실험이
아니며, 전처리·분할·평가·저장 전체 흐름을 검증하는 용도다.
"""

from __future__ import annotations

import argparse
import json
import threading
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_params.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "gym_churn_1M_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "knn_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "knn_metadata.json"
EVALUATION_PATH = PROJECT_ROOT / "data" / "evaluation" / "knn_metrics.json"

ID_COLUMN = "Member_ID"
DATE_COLUMN = "Membership_Start_Date"
DROP_COLUMNS = [
    ID_COLUMN,
    "Treadmill_Avg_Speed_Kmh",
    "Treadmill_Avg_Incline_Pct",
    "Gender",
    "Membership_Type",
    "Peak_Hour_Preference",
    "Monthly_Fee",
]
CATEGORICAL_COLUMNS = ["Cardio_Preference", "Supplement_Usage", "Profile_Type"]
STANDARD_COLUMNS = [
    "Age",
    "Monthly_Visits",
    "Avg_Workout_Duration_Min",
    "Group_Class_Attendance",
    "Avg_Equipment_Wait_Time_Min",
    "Start_Year",
    "Start_Month",
    "Start_Weekday",
    "Membership_Days",
]
ROBUST_COLUMNS = ["PT_Session_Count", "Late_Payment_Count"]


class ProgressIndicator:
    """CPU 작업이 실행 중임을 터미널에 경과 시간으로 표시한다."""

    def __init__(self, label: str, interval_seconds: float = 0.5) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self.elapsed_seconds = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> "ProgressIndicator":
        self._started_at = time.perf_counter()
        print(f"\n[시작] {self.label}", flush=True)
        self._thread = threading.Thread(target=self._show_elapsed_time, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.elapsed_seconds = time.perf_counter() - self._started_at
        status = "완료" if exc_type is None else "실패"
        print(
            f"\r[{status}] {self.label}: {self.elapsed_seconds * 1000:,.0f} ms",
            flush=True,
        )

    def _show_elapsed_time(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            elapsed_ms = (time.perf_counter() - self._started_at) * 1000
            print(
                f"\r[진행 중] {self.label}: {elapsed_ms:,.0f} ms 경과...",
                end="",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="원본 CSV 경로 (기본값: data/raw/gym_churn_1M_dataset.csv)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="KNN에 사용할 최대 행 수. 지정하지 않으면 설정 파일 값을 사용.",
    )
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_even_sample(path: Path, max_rows: int, random_state: int) -> pd.DataFrame:
    """대용량 CSV를 청크 단위로 읽어 최대 ``max_rows``개를 균등 표본으로 모은다."""
    if not path.is_file():
        raise FileNotFoundError(
            f"원본 CSV를 찾을 수 없습니다: {path}\n"
            "파일을 data/raw/gym_churn_1M_dataset.csv에 두거나 --input 경로를 지정하세요."
        )

    chunk_size = 100_000
    total_rows = sum(
        len(chunk) for chunk in pd.read_csv(path, usecols=[ID_COLUMN], chunksize=chunk_size)
    )

    if total_rows <= max_rows:
        return pd.read_csv(path)

    samples: list[pd.DataFrame] = []
    seen_rows = 0
    selected_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
        seen_rows += len(chunk)
        target_selected_rows = round(max_rows * seen_rows / total_rows)
        sample_size = target_selected_rows - selected_rows
        if sample_size:
            samples.append(
                chunk.sample(n=sample_size, random_state=random_state + chunk_index)
            )
            selected_rows += sample_size

    return pd.concat(samples, ignore_index=True)

    """학습 시점에도 알 수 있는 가입일에서 단순 달력 Feature를 만든다."""


def add_eda_features(frame: pd.DataFrame) -> pd.DataFrame:
    """EDA에서 확정한 결측치 처리, 컬럼 선택, 가입일 파생변수를 적용한다."""
    result = frame.copy()
    missing_columns = [column for column in DROP_COLUMNS if column not in result]
    if missing_columns:
        raise ValueError(f"EDA 전처리에 필요한 컬럼이 CSV에 없습니다: {missing_columns}")

    result["Cardio_Preference"] = result["Cardio_Preference"].fillna("No Preference")
    result["Supplement_Usage"] = result["Supplement_Usage"].fillna("No Protein Supplements")
    dates = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    reference_date = dates.max()
    if pd.isna(reference_date):
        raise ValueError(f"'{DATE_COLUMN}'에 해석 가능한 날짜가 없습니다.")
    result["Start_Year"] = dates.dt.year
    result["Start_Month"] = dates.dt.month
    result["Start_Weekday"] = dates.dt.dayofweek
    result["Membership_Days"] = (reference_date - dates).dt.days
    return result.drop(columns=DROP_COLUMNS + [DATE_COLUMN])

    """프로젝트 밖의 임시 입력 경로가 메타데이터에 남지 않게 한다."""


def display_input_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return f"external source supplied at runtime: {path.name}"

    """Validation 데이터에서 F1이 최대인 임계값을 선택한다."""


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5

    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def evaluate(
        y_true: pd.Series, probabilities: np.ndarray, threshold: float, elapsed_seconds: float
) -> dict[str, Any]:
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


def print_result_summary(
        sample_rows: int, y_test: pd.Series, threshold: float, metrics: dict[str, Any]
) -> None:
    """평가 수치를 발표·학습에 바로 활용할 수 있도록 한글로 출력한다."""
    true_negative, false_positive = metrics["confusion_matrix"][0]
    false_negative, true_positive = metrics["confusion_matrix"][1]
    majority_class_accuracy = max(float((y_test == 0).mean()), float((y_test == 1).mean()))

    print("\n[KNN 모델 학습 완료]")
    print(f"- 사용 표본: {sample_rows:,}명")
    print(f"- Test 데이터: {metrics['total_samples']:,}명")
    print("- 목적: 전처리·학습·평가 흐름을 확인하는 소규모 KNN 모델")

    print("\n[모델 판단 기준]")
    print(
        f"- Validation F1 최대 임계값: {threshold:.4f} "
        f"(이탈 확률이 {threshold * 100:.2f}% 이상이면 이탈로 판단)"
    )

    print("\n[Test 성능]")
    print(f"- Accuracy : {metrics['accuracy']:.4f} (전체 예측 중 맞힌 비율)")
    print(f"- Precision: {metrics['precision']:.4f} (이탈 경고 고객 중 실제 이탈 비율)")
    print(f"- Recall   : {metrics['recall']:.4f} (실제 이탈 고객 중 찾아낸 비율)")
    print(f"- F1 Score : {metrics['f1']:.4f} (Precision과 Recall의 균형)")
    print(f"- ROC-AUC  : {metrics['roc_auc']:.4f} (이탈·유지 고객을 구분하는 순위 성능)")
    print(f"- PR-AUC   : {metrics['pr_auc']:.4f} (이탈 고객 탐지 성능)")

    print("\n[혼동 행렬 해석]")
    print(f"- 실제 유지 → 유지 예측: {true_negative:,}명")
    print(f"- 실제 유지 → 이탈 오경고: {false_positive:,}명")
    print(f"- 실제 이탈 → 유지로 놓침: {false_negative:,}명")
    print(f"- 실제 이탈 → 이탈 예측: {true_positive:,}명")

    print("\n[한글 해석]")
    print(
        f"- 실제 이탈 고객 {true_positive + false_negative:,}명 중 {true_positive:,}명을 찾아 "
        f"Recall이 {metrics['recall'] * 100:.1f}%입니다."
    )
    print(
        f"- 이탈 경고 {true_positive + false_positive:,}명 중 실제 이탈은 {true_positive:,}명으로, "
        f"오경고가 {false_positive:,}명 있습니다."
    )
    print(
        f"- 다수 클래스만 예측하는 비교 기준 Accuracy는 {majority_class_accuracy * 100:.1f}%인데, "
        f"현재 KNN Accuracy는 {metrics['accuracy'] * 100:.1f}%입니다."
    )
    print("- KNN의 이탈 탐지 성향을 다른 모델과 함께 비교해 최종 모델을 선정합니다.")


def main() -> None:
    args = parse_args()
    config = load_config()
    knn_config = config["knn"]
    random_state = int(config["random_state"])
    target_column = config["target_column"]
    max_rows = args.max_rows or int(knn_config["max_rows"])

    with ProgressIndicator("원본 CSV에서 3만 행 표본 추출"):
        raw_frame = load_even_sample(args.input, max_rows, random_state)
    if target_column not in raw_frame:
        matching_columns = [
            column
            for column in raw_frame.columns
            if column.casefold() == target_column.casefold()
        ]
        if len(matching_columns) == 1:
            target_column = matching_columns[0]
        elif len(matching_columns) > 1:
            raise ValueError(
                f"Target 컬럼 '{target_column}'과 대소문자만 다른 컬럼이 여러 개 있습니다: "
                f"{matching_columns}"
            )

    if target_column not in raw_frame:
        raise ValueError(f"Target 컬럼 '{target_column}'이 CSV에 없습니다.")

    with ProgressIndicator("EDA 전처리 및 Train/Validation/Test 분할"):
        frame = add_eda_features(raw_frame)
        X = frame.drop(columns=[target_column])
        y = frame[target_column].astype(int)

        X_train_validation, X_test, y_train_validation, y_test = train_test_split(
            X,
            y,
            test_size=float(config["test_size"]),
            random_state=random_state,
            stratify=y,
        )
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=float(knn_config["validation_size"]),
            random_state=random_state,
            stratify=y_train_validation,
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "standard_numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                STANDARD_COLUMNS,
            ),
            (
                "robust_numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", RobustScaler()),
                    ]
                ),
                ROBUST_COLUMNS,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=int(knn_config["n_neighbors"]),
                    weights=knn_config["weights"],
                    n_jobs=int(knn_config["n_jobs"]),
                ),
            ),
        ]
    )

    with ProgressIndicator("전처리 Pipeline 및 KNN 모델 학습"):
        pipeline.fit(X_train, y_train)
    with ProgressIndicator("Validation 예측 및 F1 임계값 선택"):
        validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
        threshold = choose_f1_threshold(y_validation, validation_probabilities)

    with ProgressIndicator("Test 데이터 이탈 확률 예측") as inference_timer:
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    inference_seconds = inference_timer.elapsed_seconds
    metrics = evaluate(y_test, test_probabilities, threshold, inference_seconds)

    with ProgressIndicator("모델과 평가 결과 파일 저장"):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVALUATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        metadata = {
            "model": "KNeighborsClassifier",
            "purpose": "small-sample KNN model; not a full 1M-row final-model experiment",
            "input_path": display_input_path(args.input),
            "sample_rows": int(len(frame)),
            "features": X.columns.tolist(),
            "target_column": target_column,
            "target_meaning": {"0": "retained", "1": "churned"},
            "random_state": random_state,
            "threshold_from_validation": threshold,
            "preprocessing": {
                "dropped_columns": DROP_COLUMNS,
                "date_features": ["Start_Year", "Start_Month", "Start_Weekday", "Membership_Days"],
                "standard_scaled_columns": STANDARD_COLUMNS,
                "robust_scaled_columns": ROBUST_COLUMNS,
                "one_hot_columns": CATEGORICAL_COLUMNS,
            },
            "artifacts": {
                "pipeline": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
                "metrics": str(EVALUATION_PATH.relative_to(PROJECT_ROOT)),
            },
        }
        METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        EVALUATION_PATH.write_text(
            json.dumps({"metadata": metadata, "test_metrics": metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print_result_summary(len(frame), y_test, threshold, metrics)
    print("\n[저장 파일]")
    print(f"- 모델 Pipeline: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- 평가 결과: {EVALUATION_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
