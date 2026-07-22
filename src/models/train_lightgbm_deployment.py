"""원본 회원 입력을 받는 LightGBM 상위 50% 피처 배포 Pipeline을 학습·저장한다.

대시보드의 개별 고객 입력 폼과 동일한 원본 피처를 입력으로 받아 날짜 파생,
상위 50% 피처 구성, 결측치 대체, LightGBM 예측을 하나의 Pipeline에 담는다.

실행:
    .venv/bin/python src/models/train_lightgbm_deployment.py
    .venv/bin/python src/models/train_lightgbm_deployment.py --max-rows 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.models.model_lightgbm import (
    LEARNING_RATE,
    MAX_DEPTH,
    N_ESTIMATORS,
    NUM_LEAVES,
    RANDOM_STATE,
)
from src.models.lightgbm_deployment_features import (
    DEPLOYMENT_FEATURES,
    RAW_INPUT_FEATURES,
    RawPct50FeatureTransformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=config.RAW_DATA_PATH,
        help="원본 학습 CSV 경로 (기본: data/raw/gym_churn_1M_dataset.csv)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="빠른 점검용 최대 행 수. 최종 배포 모델은 지정하지 않는다.",
    )
    return parser.parse_args()


def load_training_data(path: Path, max_rows: int | None) -> tuple[pd.DataFrame, pd.Series, str]:
    if not path.is_file():
        raise FileNotFoundError(f"원본 데이터를 찾을 수 없습니다: {path}")

    usecols = [*RAW_INPUT_FEATURES, config.TARGET_COLUMN]
    frame = pd.read_csv(path, usecols=usecols, nrows=max_rows)
    if frame.empty:
        raise ValueError("학습 데이터가 비어 있습니다.")

    dates = pd.to_datetime(frame["Membership_Start_Date"], errors="coerce")
    reference_date = dates.max()
    if pd.isna(reference_date):
        raise ValueError("Membership_Start_Date에서 유효한 날짜를 찾지 못했습니다.")
    return frame[RAW_INPUT_FEATURES], frame[config.TARGET_COLUMN].astype(int), reference_date.date().isoformat()


def build_pipeline(reference_date: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", RawPct50FeatureTransformer(reference_date)),
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE,
                    max_depth=MAX_DEPTH,
                    num_leaves=NUM_LEAVES,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def metrics(y_true: pd.Series, probability: np.ndarray) -> dict[str, object]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
        "threshold": 0.5,
    }


def main() -> None:
    args = parse_args()
    X, y, reference_date = load_training_data(args.input, args.max_rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = build_pipeline(reference_date)
    pipeline.fit(X_train, y_train)
    test_probability = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = metrics(y_test, test_probability)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "lightgbm_50_pipeline.joblib"
    metadata_path = config.MODELS_DIR / "lightgbm_50_metadata.json"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model": "LightGBM",
        "dataset_label": "50",
        "purpose": "raw-input deployment pipeline for Streamlit churn prediction",
        "input_mode": "raw",
        "input_features": RAW_INPUT_FEATURES,
        "engineered_features": DEPLOYMENT_FEATURES,
        "target_column": config.TARGET_COLUMN,
        "target_meaning": {"0": "retained", "1": "churned"},
        "reference_date": reference_date,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "threshold_from_validation": 0.5,
        "hyperparameters": pipeline.named_steps["model"].get_params(),
        "test_metrics": test_metrics,
        "artifacts": {
            "pipeline": str(model_path.relative_to(config.PROJECT_ROOT)),
            "metadata": str(metadata_path.relative_to(config.PROJECT_ROOT)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"저장 완료: {model_path.relative_to(config.PROJECT_ROOT)}")
    print(f"메타데이터: {metadata_path.relative_to(config.PROJECT_ROOT)}")
    print(
        "Test - "
        + ", ".join(
            f"{key}={value:.4f}" for key, value in test_metrics.items() if isinstance(value, float)
        )
    )


if __name__ == "__main__":
    main()
