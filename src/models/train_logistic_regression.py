"""전처리된 피처 CSV로 Logistic Regression 이탈 예측 모델을 학습한다.

KNN과 동일하게 전체 피처(100%)와 중요도 상위 50% 피처를 각각 학습·평가한다.
검증 데이터의 F1 점수가 최대가 되는 임계값을 선택한 후 별도 Test 데이터로
성능을 평가해 두 모델을 같은 기준으로 비교할 수 있게 한다.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import RobustScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_FULL_DATA_PATH, PROCESSED_PCT50_DATA_PATH

CONFIG_PATH = PROJECT_ROOT / "configs" / "model_params.yaml"
MODEL_NAME = "LogisticRegression"
MODEL_DISPLAY_NAME = "Logistic Regression"
DEFAULT_INPUTS = {
    "full": PROCESSED_FULL_DATA_PATH,
    "pct50": PROCESSED_PCT50_DATA_PATH,
}
DATASET_LABELS = {"full": "100", "pct50": "50"}
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
    """CPU 작업의 경과 시간을 터미널에 표시한다."""

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
        print(f"\r[{status}] {self.label}: {self.elapsed_seconds * 1000:,.0f} ms", flush=True)

    def _show_elapsed_time(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            elapsed_ms = (time.perf_counter() - self._started_at) * 1000
            print(f"\r[진행 중] {self.label}: {elapsed_ms:,.0f} ms 경과...", end="", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("full", "pct50"),
        default="full",
        help="전처리 피처 세트: full(전체 피처) 또는 pct50(중요도 상위 50%% 피처).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="사용할 전처리 CSV 경로. 지정하면 --dataset보다 우선합니다.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="학습에 사용할 최대 행 수. 지정하지 않으면 설정 파일 값을 사용.",
    )
    return parser.parse_args()


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_even_sample(path: Path, max_rows: int, random_state: int) -> pd.DataFrame:
    """대용량 전처리 CSV를 청크 단위로 읽어 균등 표본을 모은다."""
    if not path.is_file():
        raise FileNotFoundError(
            f"전처리 CSV를 찾을 수 없습니다: {path}\n"
            "전처리 스크립트로 data/processed CSV를 생성하거나 --input 경로를 지정하세요."
        )

    chunk_size = 100_000
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if not columns:
        raise ValueError(f"CSV 헤더가 비어 있습니다: {path}")
    total_rows = sum(
        len(chunk) for chunk in pd.read_csv(path, usecols=[columns[0]], chunksize=chunk_size)
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
            samples.append(chunk.sample(n=sample_size, random_state=random_state + chunk_index))
            selected_rows += sample_size

    return pd.concat(samples, ignore_index=True)


def resolve_target_column(frame: pd.DataFrame, configured_target: str) -> str:
    """설정값과 CSV 헤더의 대소문자 표기가 달라도 타깃을 찾는다."""
    if configured_target in frame:
        return configured_target

    matching_columns = [
        column for column in frame.columns if column.casefold() == configured_target.casefold()
    ]
    if len(matching_columns) == 1:
        return matching_columns[0]
    if len(matching_columns) > 1:
        raise ValueError(
            f"Target 컬럼 '{configured_target}'과 대소문자만 다른 컬럼이 여러 개 있습니다: "
            f"{matching_columns}"
        )
    raise ValueError(f"Target 컬럼 '{configured_target}'이 CSV에 없습니다.")


def display_input_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return f"external source supplied at runtime: {path.name}"


def resolve_input_path(args: argparse.Namespace) -> Path:
    return args.input if args.input is not None else DEFAULT_INPUTS[args.dataset]


def artifact_paths(dataset: str) -> tuple[str, Path, Path, Path]:
    """입력 피처 세트별로 독립적인 모델·메타데이터·평가 경로를 만든다."""
    label = DATASET_LABELS[dataset]
    return (
        label,
        PROJECT_ROOT / "models" / f"logistic_regression_{label}_pipeline.joblib",
        PROJECT_ROOT / "models" / f"logistic_regression_{label}_metadata.json",
        PROJECT_ROOT / "data" / "evaluation" / f"{MODEL_NAME}_{label}_eval.json",
    )


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
    dataset_label: str,
    sample_rows: int,
    y_test: pd.Series,
    threshold: float,
    metrics: dict[str, Any],
) -> None:
    """평가 수치를 발표·학습에 바로 활용할 수 있도록 한글로 출력한다."""
    true_negative, false_positive = metrics["confusion_matrix"][0]
    false_negative, true_positive = metrics["confusion_matrix"][1]
    majority_class_accuracy = max(float((y_test == 0).mean()), float((y_test == 1).mean()))

    print(f"\n[{MODEL_DISPLAY_NAME} {dataset_label}% 모델 학습 완료]")
    print(f"- 사용 표본: {sample_rows:,}명")
    print(f"- Test 데이터: {metrics['total_samples']:,}명")
    print("- EDA 전처리·피처 선택 결과를 사용하고, 학습 구간 기준으로 다시 스케일링했습니다.")
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
        f"현재 {MODEL_DISPLAY_NAME} Accuracy는 {metrics['accuracy'] * 100:.1f}%입니다."
    )
    print(f"- {MODEL_DISPLAY_NAME}의 이탈 탐지 성향을 다른 모델과 함께 비교해 최종 모델을 선정합니다.")


def main() -> None:
    args = parse_args()
    config = load_config()
    logistic_config = config["logistic_regression"]
    random_state = int(config["random_state"])
    max_rows = args.max_rows or int(logistic_config["max_rows"])
    input_path = resolve_input_path(args)
    dataset_label, model_path, metadata_path, evaluation_path = artifact_paths(args.dataset)

    with ProgressIndicator(f"전처리 CSV에서 최대 {max_rows:,}행 표본 추출"):
        frame = load_even_sample(input_path, max_rows, random_state)
    target_column = resolve_target_column(frame, config["target_column"])

    with ProgressIndicator("전처리 피처 Train/Validation/Test 분할"):
        X = frame.drop(columns=[target_column])
        y = frame[target_column].astype(int)
        non_numeric_columns = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        if non_numeric_columns:
            raise ValueError(
                "전처리 CSV에는 숫자형 피처만 있어야 합니다. "
                f"다음 컬럼을 확인하세요: {non_numeric_columns}"
            )
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
            test_size=float(logistic_config["validation_size"]),
            random_state=random_state,
            stratify=y_train_validation,
        )

    standard_columns = [column for column in STANDARD_COLUMNS if column in X]
    robust_columns = [column for column in ROBUST_COLUMNS if column in X]
    remaining_columns = [
        column for column in X.columns if column not in standard_columns + robust_columns
    ]
    transformers = [
        (
            "standard_numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                ]
            ),
            standard_columns,
        ),
        (
            "robust_numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", RobustScaler()),
                ]
            ),
            robust_columns,
        ),
        (
            "remaining_numeric",
            SimpleImputer(strategy="median", add_indicator=True),
            remaining_columns,
        ),
    ]
    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=float(logistic_config["C"]),
                    class_weight=logistic_config["class_weight"],
                    max_iter=int(logistic_config["max_iter"]),
                    solver=logistic_config["solver"],
                ),
            ),
        ]
    )

    with ProgressIndicator("전처리 Pipeline 및 Logistic Regression 모델 학습"):
        pipeline.fit(X_train, y_train)
    with ProgressIndicator("Validation 예측 및 F1 임계값 선택"):
        validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
        threshold = choose_f1_threshold(y_validation, validation_probabilities)
    with ProgressIndicator("Test 데이터 이탈 확률 예측") as inference_timer:
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_probabilities, threshold, inference_timer.elapsed_seconds)

    with ProgressIndicator("모델과 평가 결과 파일 저장"):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        metadata = {
            "model": MODEL_NAME,
            "purpose": "logistic-regression churn model using EDA-preprocessed features",
            "input_path": display_input_path(input_path),
            "input_dataset": args.dataset if args.input is None else "custom",
            "dataset_label": dataset_label,
            "sample_rows": int(len(frame)),
            "features": X.columns.tolist(),
            "target_column": target_column,
            "target_meaning": {"0": "retained", "1": "churned"},
            "random_state": random_state,
            "threshold_from_validation": threshold,
            "preprocessing": "Applied upstream in the input CSV; this script imputes and "
            "scales numeric features within the training pipeline.",
            "artifacts": {
                "pipeline": str(model_path.relative_to(PROJECT_ROOT)),
                "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
                "metrics": str(evaluation_path.relative_to(PROJECT_ROOT)),
            },
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        evaluation_path.write_text(
            json.dumps({"metadata": metadata, "test_metrics": metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print_result_summary(dataset_label, len(frame), y_test, threshold, metrics)
    print("\n[저장 파일]")
    print(f"- 모델 Pipeline: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"- 평가 결과: {evaluation_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
