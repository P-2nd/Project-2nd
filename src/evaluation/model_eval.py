# =====================================================
# Config
# =====================================================

from pathlib import Path
import sys
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent.parent

sys.path.insert(
    0,
    str(BASE_DIR)
)

print("BASE_DIR :", BASE_DIR)


# ==========================
# Data
# ==========================

DATA_PATHS = {
    "full": BASE_DIR / "data" / "processed" / "churn_preprocessed_full.csv",
    "50": BASE_DIR / "data" / "processed" / "churn_preprocessed_pct50.csv"
}


# ==========================
# Directory
# ==========================

MODEL_DIR = (
    BASE_DIR
    / "src"
    / "models"
)


MODEL_SAVE_DIR = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "saved_models"
)


PARAM_SAVE_DIR = (
    BASE_DIR
    / "data"
    / "evaluation"
    / "saved_params"
)


RESULT_SAVE_DIR = (
    BASE_DIR
    / "data"
    / "results"
)


ROC_SAVE_DIR = (
    RESULT_SAVE_DIR
    / "roc"
)

MODEL_LIST_PATH = (
    BASE_DIR
    / "data"
    / "model_list.json"
)

for path in [
    MODEL_SAVE_DIR,
    PARAM_SAVE_DIR,
    RESULT_SAVE_DIR,
    ROC_SAVE_DIR
]:
    path.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================
# Parameter
# ==========================

TARGET = "Churn"

TEST_SIZE = 0.2

RANDOM_STATE = 42


# None이면 전체 피처
EXCLUDE_FEATURES = [
    "Late_Payment_Count",
    "PT_Session_Count"
]

EXCLUDE_FILES = {
    "__init__.py",
    "model_registry.py",
}


parser = argparse.ArgumentParser(
    description="학습된 이탈 예측 모델을 데이터셋별로 평가합니다."
)
parser.add_argument(
    "--baseline-only",
    action="store_true",
    help="특성 제외 부가 실험은 건너뛰고 full·50 기본 결과만 생성합니다."
)
ARGS = parser.parse_args()


# =====================================================
# Import
# =====================================================

import json
import time
import joblib
import pandas as pd
import importlib.util


from sklearn.base import clone

from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# =====================================================
# Function
# =====================================================

def load_or_train(
    model_name,
    model,
    X,
    y,
    data_label
):

    feature_sets = [
        (None, X)
    ]


    if EXCLUDE_FEATURES is not None and not ARGS.baseline_only:

        for feature in EXCLUDE_FEATURES:

            if feature not in X.columns:

                print(
                    f"[Skip] {feature}"
                )

                continue


            feature_sets.append(
                (
                    feature,
                    X.drop(
                        columns=[feature]
                    )
                )
            )


    results = {}


    for feature, X_selected in feature_sets:

        print(
            f"\n[{data_label}] {model_name} Feature : {feature}"
        )


        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )


        if feature is None:

            save_name = (
                f"{model_name}_{data_label}"
            )

        else:

            save_name = (
                f"{model_name}_{data_label}_without_{feature}"
            )


        model_path = (
            MODEL_SAVE_DIR
            /
            f"{save_name}_eval.joblib"
        )


        param_path = (
            PARAM_SAVE_DIR
            /
            f"{save_name}_params.json"
        )


        model_instance = clone(model)


        # ==========================
        # Load / Train
        # ==========================

        if model_path.exists():

            print(
                "[LOAD]",
                save_name
            )


            model_instance = joblib.load(
                model_path
            )


        else:

            print(
                "[TRAIN]",
                save_name
            )


            model_instance.fit(
                X_train,
                y_train
            )


            joblib.dump(
                model_instance,
                model_path
            )


        # ==========================
        # Save Parameter
        # ==========================

        with open(
            param_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                model_instance.get_params(),
                f,
                indent=4,
                ensure_ascii=False,
                default=str
            )


        # ==========================
        # Evaluation
        # ==========================

        start = time.perf_counter()


        y_pred = model_instance.predict(
            X_test
        )


        if hasattr(
            model_instance,
            "predict_proba"
        ):

            y_score = (
                model_instance
                .predict_proba(X_test)[:, 1]
            )

        elif hasattr(
            model_instance,
            "decision_function"
        ):

            y_score = (
                model_instance
                .decision_function(X_test)
            )

        else:

            y_score = y_pred


        total_time = (
            time.perf_counter()
            -
            start
        )

        # ==========================
        # Metrics
        # ==========================

        avg_latency_ms = (
            total_time
            * 1000
            /
            len(y_test)
        )


        auc_score = roc_auc_score(
            y_test,
            y_score
        )


        metrics = {

            "accuracy": accuracy_score(
                y_test,
                y_pred
            ),

            "precision": precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "recall": recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "f1": f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

            "roc_auc": auc_score,

            "pr_auc": average_precision_score(
                y_test,
                y_score
            ),

            "confusion_matrix": confusion_matrix(
                y_test,
                y_pred
            ).tolist(),

            "total_samples": len(y_test),

            "average_inference_ms": avg_latency_ms
        }


        # ==========================
        # ROC parquet 저장
        # ==========================

        roc_path = (
            ROC_SAVE_DIR
            /
            f"{save_name}.parquet"
        )


        pd.DataFrame(
            {
                "y_true": y_test,
                "y_score": y_score
            }
        ).to_parquet(
            roc_path,
            index=False
        )


        # ==========================
        # Result JSON 저장
        # ==========================

        result = {

            "model_key": save_name,

            "model_name": save_name.upper(),

            "experiment": {

                "comparison_axis": "feature_set",

                "dataset": data_label,

                "feature_set": (
                    "pct50"
                    if data_label == "50"
                    else "full"
                ),

                "excluded_feature": feature,

                "feature_count": X_selected.shape[1]
            },

            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],

            "total_time_sec": total_time,

            "avg_latency_ms": avg_latency_ms,

            "total_samples": len(y_test),


            "confusion_matrix":
                confusion_matrix(
                    y_test,
                    y_pred
                ).tolist(),


            "auc_score": auc_score,

            "pr_auc": metrics["pr_auc"],

            "threshold": 0.5,


            "parquet": {

                "path": roc_path.relative_to(BASE_DIR).as_posix(),

                "columns": [
                    "y_true",
                    "y_score"
                ]
            }
        }


        result_path = (
            RESULT_SAVE_DIR
            /
            f"{save_name}_results.json"
        )


        with open(
            result_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False,
                default=str
            )


        results[save_name] = {

            "model": model_instance,

            "metrics": metrics
        }


    return results



# =====================================================
# Model Auto Load
# =====================================================

model_files = sorted(
    [
        file
        for file in MODEL_DIR.glob("*.py")
        if file.name.startswith("model_") and file.name not in EXCLUDE_FILES
    ]
)

model_list = {
    "models": [
        file.stem.removeprefix("model_")
        for file in model_files
    ],

    "exclude_features": EXCLUDE_FEATURES
}


with open(
    MODEL_LIST_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_list,
        f,
        indent=4,
        ensure_ascii=False
    )


print(
    f"모델 리스트 저장 완료 : {MODEL_LIST_PATH}"
)


# =====================================================
# Run
# =====================================================

for data_label, data_path in DATA_PATHS.items():

    print("=" * 60)
    print(f"DATASET : {data_label}")
    print("=" * 60)


    df = pd.read_csv(
        data_path
    )


    X = df.drop(
        columns=[TARGET]
    )


    y = df[TARGET]


    for model_file in model_files:


        module_name = model_file.stem


        model_name = (
            module_name
            .removeprefix("model_")
        )


        print("=" * 60)
        print(
            f"{data_label} - {model_name} 실행"
        )
        print("=" * 60)


        spec = importlib.util.spec_from_file_location(
            module_name,
            model_file
        )


        module = importlib.util.module_from_spec(
            spec
        )


        spec.loader.exec_module(
            module
        )


        load_or_train(
            model_name=model_name,
            model=module.model,
            X=X,
            y=y,
            data_label=data_label
        )


print(
    "\n모든 모델 평가 완료"
)
