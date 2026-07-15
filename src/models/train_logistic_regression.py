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

# 이 파일(src/models/...)을 기준으로 프로젝트 최상위 폴더를 찾는다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.results import roc_data_path, upsert_result
from src.config import PROCESSED_FULL_DATA_PATH, PROCESSED_PCT50_DATA_PATH, RESULT_DATA_PATH

CONFIG_PATH = PROJECT_ROOT / "configs" / "model_params.yaml"
MODEL_NAME = "LogisticRegression"
MODEL_DISPLAY_NAME = "Logistic Regression"
# --dataset 옵션을 생략했을 때 사용할 전처리 데이터 파일이다.
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
    """오래 걸릴 수 있는 작업의 경과 시간을 터미널에 보여 주는 도우미 클래스."""

    def __init__(self, label: str, interval_seconds: float = 0.5) -> None:
        self.label = label
        self.interval_seconds = interval_seconds
        self.elapsed_seconds = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def __enter__(self) -> "ProgressIndicator":
        # with ProgressIndicator(...) 블록이 시작될 때 자동으로 실행된다.
        self._started_at = time.perf_counter()
        print(f"\n[시작] {self.label}", flush=True)
        self._thread = threading.Thread(target=self._show_elapsed_time, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        # with 블록이 끝나면 타이머 스레드를 멈추고 총 시간을 출력한다.
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
    """터미널 실행 옵션(--dataset, --input, --max-rows)을 읽는다."""
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
    """YAML 설정 파일에서 Logistic Regression 하이퍼파라미터를 불러온다."""
    with CONFIG_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_even_sample(path: Path, max_rows: int, random_state: int) -> pd.DataFrame:
    """대용량 전처리 CSV를 메모리에 모두 올리지 않고 균등 표본을 모은다."""
    if not path.is_file():
        raise FileNotFoundError(
            f"전처리 CSV를 찾을 수 없습니다: {path}\n"
            "전처리 스크립트로 data/processed CSV를 생성하거나 --input 경로를 지정하세요."
        )

    # 한 번에 10만 행씩만 읽어 100만 행 데이터도 메모리 부담을 줄인다.
    chunk_size = 100_000
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if not columns:
        raise ValueError(f"CSV 헤더가 비어 있습니다: {path}")
    # 전체 행 수를 먼저 알아야 각 청크에서 뽑을 표본 수를 비율로 정할 수 있다.
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
        # 앞부분에만 표본이 몰리지 않도록, 읽은 비율만큼 표본을 배정한다.
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
    """저장하는 메타데이터에 프로젝트 기준의 짧은 경로를 남긴다."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return f"external source supplied at runtime: {path.name}"


def resolve_input_path(args: argparse.Namespace) -> Path:
    """직접 입력한 CSV가 있으면 우선 사용하고, 없으면 dataset 기본 파일을 쓴다."""
    return args.input if args.input is not None else DEFAULT_INPUTS[args.dataset]


def artifact_paths(dataset: str) -> tuple[str, Path, Path, Path]:
    """입력 피처 세트별 모델·메타데이터·ROC 데이터 경로를 만든다."""
    label = DATASET_LABELS[dataset]
    return (
        label,
        PROJECT_ROOT / "models" / f"logistic_regression_{label}_pipeline.joblib",
        PROJECT_ROOT / "models" / f"logistic_regression_{label}_metadata.json",
        roc_data_path("logistic_regression", label),
    )


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Validation 데이터에서 F1이 가장 높은 이탈 확률 기준을 찾는다.

    예를 들어 반환값이 0.40이면 이탈 확률이 40% 이상인 고객을 이탈로 판단한다.
    Test 데이터는 이 과정에 사용하지 않아 최종 성능을 공정하게 확인할 수 있다.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        return 0.5
    # precision/recall의 마지막 값은 threshold가 없으므로 [:-1]로 길이를 맞춘다.
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def evaluate(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float, elapsed_seconds: float
) -> dict[str, Any]:
    """확률 예측값을 0/1 예측으로 바꾸고 주요 분류 지표를 한 번에 계산한다."""
    # 확률 자체는 AUC 계산에 쓰고, threshold를 넘었는지는 Accuracy/F1 등에 쓴다.
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
    """데이터 로드 → 분할 → 스케일링·학습 → 평가 → 저장 순서로 실험을 실행한다."""
    # 데이터 읽기부터 평가 직전까지 걸린 전체 시간을 결과 JSON에 남기기 위한 시작 시각이다.
    started_at = time.perf_counter()
    args = parse_args()
    config = load_config()
    # model_params.yaml의 logistic_regression: 아래 설정(C, max_iter 등)을 가져온다.
    logistic_config = config["logistic_regression"]
    random_state = int(config["random_state"])
    max_rows = args.max_rows or int(logistic_config["max_rows"])
    input_path = resolve_input_path(args)
    dataset_label, model_path, metadata_path, roc_path = artifact_paths(args.dataset)

    with ProgressIndicator(f"전처리 CSV에서 최대 {max_rows:,}행 표본 추출"):
        frame = load_even_sample(input_path, max_rows, random_state)
    # 설정 파일의 타깃 이름(Churn)과 CSV의 대소문자가 달라도 안전하게 찾는다.
    target_column = resolve_target_column(frame, config["target_column"])

    with ProgressIndicator("전처리 피처 Train/Validation/Test 분할"):
        # X는 모델 입력 피처, y는 맞혀야 할 정답(이탈 여부)이다.
        # Late_Payment_Count는 이번 실험에서 제외하기로 한 피처라서 타깃과 함께 제거한다.
        X = frame.drop(columns=[target_column, "Late_Payment_Count"])
        y = frame[target_column].astype(int)

        # 이 CSV는 앞선 전처리에서 숫자형/원-핫 피처로 만들어졌어야 한다.
        non_numeric_columns = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
        if non_numeric_columns:
            raise ValueError(
                "전처리 CSV에는 숫자형 피처만 있어야 합니다. "
                f"다음 컬럼을 확인하세요: {non_numeric_columns}"
            )
        # Test는 마지막 성능 확인용으로 따로 보관한다. stratify=y는 이탈 비율을 각 세트에 유지한다.
        X_train_validation, X_test, y_train_validation, y_test = train_test_split(
            X,
            y,
            test_size=float(config["test_size"]),
            random_state=random_state,
            stratify=y,
        )
        # 남은 데이터는 실제 학습용 Train과 임계값 선택용 Validation으로 한 번 더 나눈다.
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=float(logistic_config["validation_size"]),
            random_state=random_state,
            stratify=y_train_validation,
        )

    # 피처를 제거했어도 오류가 나지 않도록, 실제 X에 있는 컬럼만 스케일링 목록에 넣는다.
    standard_columns = [column for column in STANDARD_COLUMNS if column in X]
    robust_columns = [column for column in ROBUST_COLUMNS if column in X]
    remaining_columns = [
        column for column in X.columns if column not in standard_columns + robust_columns
    ]
    # ColumnTransformer는 컬럼 종류별로 다른 전처리를 적용한 뒤 결과를 하나로 합친다.
    transformers = [
        (
            "standard_numeric",
            Pipeline(
                steps=[
                    # 결측값은 중앙값으로 채우고, 결측 여부도 별도 피처로 남긴다.
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    # 평균이 0, 표준편차가 1에 가깝도록 맞춰 계수 비교를 안정적으로 만든다.
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
                    # 이상치가 큰 횟수형 피처는 중앙값·사분위수 기준의 RobustScaler를 사용한다.
                    ("scaler", RobustScaler()),
                ]
            ),
            robust_columns,
        ),
        (
            "remaining_numeric",
            # 원-핫 피처처럼 따로 스케일링할 필요가 없는 나머지 숫자형 컬럼이다.
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
                    # C가 클수록 규제가 약해져 학습 데이터에 더 민감해진다.
                    C=float(logistic_config["C"]),
                    # 이탈/유지 비율 차이를 보정해 소수 클래스인 이탈을 놓치지 않도록 돕는다.
                    class_weight=logistic_config["class_weight"],
                    # 최적화 계산의 최대 반복 횟수다. 실제 반복 횟수가 반드시 이 값은 아니다.
                    max_iter=int(logistic_config["max_iter"]),
                    solver=logistic_config["solver"],
                ),
            ),
        ]
    )

    with ProgressIndicator("전처리 Pipeline 및 Logistic Regression 모델 학습"):
        # fit은 Train에만 실행한다. Test 정보가 학습에 섞이면 성능이 과하게 좋아 보일 수 있다.
        pipeline.fit(X_train, y_train)
    with ProgressIndicator("Validation 예측 및 F1 임계값 선택"):
        # predict_proba[:, 1]은 '이탈(1)'일 확률만 꺼낸 것이다.
        validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
        threshold = choose_f1_threshold(y_validation, validation_probabilities)
    with ProgressIndicator("Test 데이터 이탈 확률 예측") as inference_timer:
        # Test는 여기서 처음 사용한다. 최종 성능을 공정하게 확인하기 위해서다.
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_probabilities, threshold, inference_timer.elapsed_seconds)
    total_seconds = time.perf_counter() - started_at

    with ProgressIndicator("모델과 평가 결과 파일 저장"):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        roc_path.parent.mkdir(parents=True, exist_ok=True)
        # 전처리기와 모델을 함께 저장해야 새 고객 데이터에도 같은 방식으로 예측할 수 있다.
        joblib.dump(pipeline, model_path)
        artifacts = {
            "pipeline": str(model_path.relative_to(PROJECT_ROOT)),
            "metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
            "result_data": str(RESULT_DATA_PATH.relative_to(PROJECT_ROOT)),
            "roc_source": str(roc_path.relative_to(PROJECT_ROOT)),
        }
        # 나중에 어떤 데이터·설정으로 학습했는지 추적할 수 있도록 설명 정보를 별도로 저장한다.
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
            "artifacts": artifacts,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        # ROC 곡선 등을 다시 그릴 수 있도록 Test 정답과 이탈 확률을 Parquet로 남긴다.
        pd.DataFrame({"y_true": y_test.to_numpy(), "y_score": test_probabilities}).to_parquet(
            roc_path, index=False
        )
        # result_data.json에서 Logistic Regression/100 또는 /50 항목만 최신 결과로 갱신한다.
        upsert_result(
            model_key="logistic_regression",
            model_name=MODEL_NAME,
            label=dataset_label,
            experiment={
                "comparison_axis": "feature_set",
                "feature_set": args.dataset if args.input is None else "custom",
                "feature_count": int(X.shape[1]),
            },
            metrics=metrics,
            threshold=threshold,
            total_time_sec=total_seconds,
            artifacts=artifacts,
        )

    print_result_summary(dataset_label, len(frame), y_test, threshold, metrics)
    print("\n[저장 파일]")
    print(f"- 모델 Pipeline: {model_path.relative_to(PROJECT_ROOT)}")
    print(f"- 통합 평가 결과: {RESULT_DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"- ROC 원천 데이터: {roc_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
