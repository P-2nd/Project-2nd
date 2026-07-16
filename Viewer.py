# =====================================================
# Viewer.py
# =====================================================

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# ==========================
# Path
# ==========================

BASE_DIR = Path(__file__).resolve().parent

MODEL_LIST_PATH = BASE_DIR / "data/model_list.json"
RESULT_DIR = BASE_DIR / "data/results"
PARAM_DIR = BASE_DIR / "data/evaluation/saved_params"
ROC_DIR = RESULT_DIR / "roc"

PLOT_DIR = BASE_DIR / "data/evaluation/plots"
PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================
# Load Config
# ==========================

with open(MODEL_LIST_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

models = config["models"]
exclude_features = config.get("exclude_features", [])


# ==========================
# Viewer
# ==========================

while True:

    # --------------------------
    # Model Select
    # --------------------------

    print("\n===== 모델 목록 =====")

    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")

    model_idx = int(input("\n모델 선택 : "))
    model_name = models[model_idx - 1]

    # --------------------------
    # Dataset Select
    # --------------------------

    print("\n===== 데이터 선택 =====")
    print("1. 전체 데이터")
    print("2. 상위 중요도 50%")

    data_choice = int(input("선택 : "))

    if data_choice == 1:
        data_name = "full"
    elif data_choice == 2:
        data_name = "50"
    else:
        print("잘못된 데이터 선택")
        continue

    # --------------------------
    # Feature Select
    # --------------------------

    print("\n===== Feature 선택 =====")
    print("1. 전체 Feature")

    for i, feature in enumerate(exclude_features, start=2):
        print(f"{i}. {feature} 제외")

    feature_choice = int(input("선택 : "))

    if feature_choice == 1:
        save_name = f"{model_name}_{data_name}"

    elif 2 <= feature_choice <= len(exclude_features) + 1:
        feature = exclude_features[feature_choice - 2]
        save_name = f"{model_name}_{data_name}_without_{feature}"

    else:
        print("잘못된 Feature 선택")
        continue

    print(f"\n선택 모델 : {save_name}")

    # --------------------------
    # Result JSON
    # --------------------------

    result_path = (
        RESULT_DIR
        /f"{save_name}_results.json"
    )


    if not result_path.exists():
        print( "결과 파일 없음:", result_path.name)
        continue


    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)


    print("\n========== 결과 ==========")

    print("모델:", result["model_name"])

    if "accuracy" in result:
        print(f"Accuracy : {result['accuracy']:.4f}")

    if "precision" in result:
        print(f"Precision: {result['precision']:.4f}")

    if "recall" in result:
        print(f"Recall   : {result['recall']:.4f}")

    if "f1_score" in result:
        print(f"F1 Score : {result['f1_score']:.4f}")

    print(f"AUC Score: {result['auc_score']:.4f}")
    print(f"실행 시간(sec): {result['total_time_sec']:.6f}")
    print(f"평균 latency(ms): {result['avg_latency_ms']:.6f}")
    print(f"샘플 수: {result['total_samples']}")

    print("\nConfusion Matrix")

    cm = pd.DataFrame(
        result["confusion_matrix"],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    print(cm)


    # --------------------------
    # Parameter
    # --------------------------

    param_path = (
        PARAM_DIR
        /f"{save_name}_params.json"
    )


    if param_path.exists():

        print("\n========== Parameter ==========")

        with open(param_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        for k, v in params.items():

            if v is None:
                continue

            print(f"{k}: {v}")


    # --------------------------
    # ROC
    # --------------------------

    roc_path = (
            ROC_DIR
            / f"{save_name}.parquet"
    )


    if roc_path.exists():

        df = pd.read_parquet(
            roc_path
        )

        fpr, tpr, _ = roc_curve(
            df["y_true"],
            df["y_score"]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.figure(figsize=(6, 5))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC={roc_auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title(f"ROC Curve - {save_name}")
        plt.legend()
        plt.grid()
        roc_plot_path = (
                PLOT_DIR
                /f"{save_name}_roc.png"
        )

        if not roc_plot_path.exists():
            plt.savefig(
                roc_plot_path,
                dpi=300,
                bbox_inches="tight"
            )

        plt.show()

    else:
        print("ROC parquet 없음")

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_plot_path = (
            PLOT_DIR
            /f"{save_name}_confusion_matrix.png"
    )

    if not cm_plot_path.exists():
        plt.savefig(
            cm_plot_path,
            dpi=300,
            bbox_inches="tight"
        )
    plt.show()

    # --------------------------
    # Continue
    # --------------------------

    again = input("\n모델 다시 선택? (y/n): ")

    if again.lower() != "y":

        print("Viewer 종료")
        break