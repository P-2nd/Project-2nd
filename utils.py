# =====================================================
# utils.py
# =====================================================

from pathlib import Path
import json
import pandas as pd


# =====================================================
# Path
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_LIST_PATH = BASE_DIR / "data/model_list.json"

RESULT_DIR = BASE_DIR / "data/results"

PARAM_DIR = BASE_DIR / "data/evaluation/saved_params"

PLOT_DIR = BASE_DIR / "data/evaluation/plots"


# =====================================================
# Config
# =====================================================

def load_config():

    with open(MODEL_LIST_PATH, encoding="utf-8") as f:

        return json.load(f)


# =====================================================
# Result
# =====================================================

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


def load_all_results():

    results = {}

    for file in sorted(RESULT_DIR.glob("*_results.json")):

        model_key = file.stem.replace("_results", "")

        results[model_key] = load_result(model_key)

    return results


# =====================================================
# Parameter
# =====================================================

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


def load_parquet(model_key):

    result = load_result(model_key)

    if result is None:

        return None

    parquet_path = resolve_artifact_path(result["parquet"]["path"])

    if parquet_path is None:

        return None

    return pd.read_parquet(parquet_path)


# =====================================================
# Plot
# =====================================================

def get_plot_paths(model_key):

    return {

        "roc": PLOT_DIR / f"{model_key}_roc.png",

        "cm": PLOT_DIR / f"{model_key}_confusion_matrix.png"

    }
