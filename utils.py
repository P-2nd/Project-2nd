# =====================================================
# utils.py
# =====================================================

from pathlib import Path
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =====================================================
# Path
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_LIST_PATH = BASE_DIR / "data/model_list.json"

RESULT_DIR = BASE_DIR / "data/results"

PARAM_DIR = BASE_DIR / "data/evaluation/saved_params"

PLOT_DIR = BASE_DIR / "data/evaluation/plots"

MODEL_DIR = BASE_DIR / "models"

PROCESSED_DIR = BASE_DIR / "data/processed"

FINAL_MODEL_KEY = "lightgbm_50"

DEPLOYMENT_MODEL_CANDIDATES = (
    MODEL_DIR / "lightgbm_pct50_pipeline.joblib",
    MODEL_DIR / "lightgbm_50_pipeline.joblib",
)

DEPLOYMENT_METADATA_CANDIDATES = (
    MODEL_DIR / "lightgbm_pct50_metadata.json",
    MODEL_DIR / "lightgbm_50_metadata.json",
)


# =====================================================
# Config
# =====================================================

@lru_cache(maxsize=1)
def load_config():

    with open(MODEL_LIST_PATH, encoding="utf-8") as f:

        return json.load(f)


# =====================================================
# Result
# =====================================================

@lru_cache(maxsize=64)
def load_result(model_key):

    result_path = RESULT_DIR / f"{model_key}_results.json"

    if not result_path.exists():

        return None

    with open(result_path, encoding="utf-8") as f:

        result = json.load(f)

    parquet = result.get("parquet")

    if isinstance(parquet, dict) and isinstance(parquet.get("path"), str):

        parquet_path = resolve_artifact_path(parquet["path"])

        if parquet_path is not None:

            parquet["path"] = parquet_path.relative_to(BASE_DIR).as_posix()

    return result


def display_model_name(model_key):
    """Return a Korean-friendly model/experiment label."""

    base_key = model_key.split("_without_", 1)[0]
    dataset = "상위 50% 피처" if base_key.endswith("_50") else "전체 피처"
    model = base_key.rsplit("_", 1)[0].replace("randomforest", "Random Forest")
    model = model.replace("lightgbm", "LightGBM").replace("xgboost", "XGBoost")
    model = model.replace("logistic", "Logistic Regression").replace("knn", "KNN")

    if "_without_" in model_key:
        excluded = model_key.split("_without_", 1)[1]
        return f"{model} · {dataset} · {excluded} 제외"
    return f"{model} · {dataset}"


def load_all_results():

    results = {}

    for file in sorted(RESULT_DIR.glob("*_results.json")):

        model_key = file.stem.replace("_results", "")

        results[model_key] = load_result(model_key)

    return results


# =====================================================
# Parameter
# =====================================================

@lru_cache(maxsize=64)
def load_params(model_key):

    param_path = PARAM_DIR / f"{model_key}_params.json"

    if not param_path.exists():

        return {}

    with open(param_path, encoding="utf-8") as f:

        params = json.load(f)

    return {

        k: v

        for k, v in params.items()

        if v is not None

    }


# =====================================================
# ROC Data
# =====================================================

def resolve_artifact_path(path_value, *, base_dir=BASE_DIR):
    """Resolve both portable and legacy artifact paths inside this project."""

    declared_path = Path(path_value)

    candidates = []

    if declared_path.is_absolute():
        candidates.append(declared_path)
    else:
        candidates.append(base_dir / declared_path)

    filename = path_value.replace("\\", "/").rsplit("/", 1)[-1]

    if filename not in {"", ".", ".."}:
        candidates.append(base_dir / "data" / "results" / "roc" / filename)

    for candidate in candidates:

        if candidate.exists():
            return candidate

    return None


@lru_cache(maxsize=64)
def load_parquet(model_key):

    result = load_result(model_key)

    if result is None:

        return None

    parquet_path = resolve_artifact_path(result["parquet"]["path"])

    if parquet_path is None:

        return None

    return pd.read_parquet(parquet_path)


def calculate_threshold_metrics(roc_df, threshold):
    """Calculate campaign-facing classification metrics at a threshold."""

    y_true = roc_df["y_true"].astype(int).to_numpy()
    y_score = roc_df["y_score"].astype(float).to_numpy()
    y_pred = (y_score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "selected": int(tp + fp),
        "actual_churn": int(tp + fn),
        "total": int(len(y_true)),
    }


def risk_level(probability):
    if probability >= 0.7:
        return "고위험", "red"
    if probability >= 0.4:
        return "주의", "orange"
    return "유지", "green"


def retention_recommendations(customer):
    """Create transparent, rule-based retention actions from customer inputs."""

    recommendations = []
    if customer["Late_Payment_Count"] >= 4:
        recommendations.append(("결제 지원", "연체가 잦습니다. 자동결제 등록과 결제일 사전 알림을 제안하세요."))
    if customer["Monthly_Visits"] <= 6:
        recommendations.append(("방문 활성화", "최근 방문 빈도가 낮습니다. 2주 방문 챌린지나 재방문 쿠폰을 제안하세요."))
    if customer["PT_Session_Count"] == 0:
        recommendations.append(("PT 체험", "PT 이용 이력이 없습니다. 무료 체형 분석 또는 1회 체험권이 적합합니다."))
    if customer["Group_Class_Attendance"] <= 2:
        recommendations.append(("그룹 수업", "그룹 수업 참여가 낮습니다. 선호 시간대의 입문 수업을 추천하세요."))
    if customer["Avg_Equipment_Wait_Time_Min"] >= 18:
        recommendations.append(("혼잡도 개선", "기구 대기 시간이 깁니다. 한산한 시간대와 대체 운동 동선을 안내하세요."))
    if customer["Avg_Workout_Duration_Min"] <= 40:
        recommendations.append(("운동 루틴", "운동 시간이 짧습니다. 30~45분 완성형 루틴을 제공하세요."))
    if not recommendations:
        recommendations.append(("관계 유지", "뚜렷한 행동 위험 신호가 없습니다. 정기 성과 리포트와 멤버십 혜택을 안내하세요."))
    return recommendations


def find_deployment_artifacts():
    model_path = next((path for path in DEPLOYMENT_MODEL_CANDIDATES if path.exists()), None)
    metadata_path = next((path for path in DEPLOYMENT_METADATA_CANDIDATES if path.exists()), None)
    return model_path, metadata_path


@lru_cache(maxsize=2)
def load_deployment_bundle(model_path, metadata_path):
    """Load the externally supplied LightGBM deployment bundle."""

    model = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return model, metadata


def predict_customer(customer, model, metadata):
    """Predict from a raw-input deployment Pipeline following the UI contract."""

    if metadata.get("input_mode", "raw") != "raw":
        raise ValueError("배포 메타데이터의 input_mode는 'raw'여야 합니다.")

    feature_names = metadata.get("input_features") or metadata.get("features")
    if not feature_names:
        raise ValueError("메타데이터에 input_features 또는 features가 필요합니다.")

    missing = [name for name in feature_names if name not in customer]
    if missing:
        raise ValueError(f"입력 폼에 없는 피처입니다: {', '.join(missing)}")

    frame = pd.DataFrame([{name: customer[name] for name in feature_names}])
    probability = float(np.asarray(model.predict_proba(frame))[0, 1])
    threshold = float(
        metadata.get("threshold_from_validation", metadata.get("threshold", 0.5))
    )
    return probability, threshold


def load_feature_importance(model_key):
    """Read lightweight evaluation models only; avoid multi-GB artifacts."""

    model_path = BASE_DIR / "data/evaluation/saved_models" / f"{model_key}_eval.joblib"
    if not model_path.exists() or model_path.stat().st_size > 100 * 1024 * 1024:
        return None

    dataset_path = PROCESSED_DIR / (
        "churn_preprocessed_pct50.csv" if "_50" in model_key else "churn_preprocessed_full.csv"
    )
    features = [column for column in pd.read_csv(dataset_path, nrows=0).columns if column != "Churn"]
    model = joblib.load(model_path)

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)[0])
    else:
        return None

    if len(values) != len(features):
        return None
    return (
        pd.DataFrame({"Feature": features, "Importance": values})
        .sort_values("Importance", ascending=False)
        .head(12)
    )


# =====================================================
# Plot
# =====================================================

def get_plot_paths(model_key):

    return {

        "roc": PLOT_DIR / f"{model_key}_roc.png",

        "cm": PLOT_DIR / f"{model_key}_confusion_matrix.png"

    }
