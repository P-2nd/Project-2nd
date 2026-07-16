# =====================================================
# evaluation_plot.py
# =====================================================

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RESULT_DIR = BASE_DIR / "data/results"
ROC_DIR = RESULT_DIR / "roc"
PLOT_DIR = BASE_DIR / "data/evaluation/plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(result_path):

    save_name = result_path.stem.replace("_results", "")

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    # ==========================
    # Confusion Matrix
    # ==========================

    cm_path = PLOT_DIR / f"{save_name}_confusion_matrix.png"

    if not cm_path.exists():

        cm = pd.DataFrame(
            result["confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        plt.figure(figsize=(5,4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues"
        )

        plt.title(f"Confusion Matrix - {save_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        plt.savefig(
            cm_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("저장:", cm_path)

    # ==========================
    # ROC Curve
    # ==========================

    roc_files = list(
        ROC_DIR.glob(f"{save_name}_*.parquet")
    )

    if not roc_files:
        print("ROC 없음:", save_name)
        return

    roc_data = roc_files[0]

    roc_path = PLOT_DIR / f"{save_name}_roc.png"

    if roc_path.exists():
        print("존재:", roc_path)
        return

    df = pd.read_parquet(
        roc_data
    )

    fpr, tpr, _ = roc_curve(
        df["y_true"],
        df["y_score"]
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6,5))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC={roc_auc:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {save_name}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("저장:", roc_path)


# =====================================================
# Run All
# =====================================================

if __name__ == "__main__":

    result_files = RESULT_DIR.glob("*_results.json")

    for result_file in result_files:
        save_plot(result_file)

    print("\n모든 평가 그래프 저장 완료")