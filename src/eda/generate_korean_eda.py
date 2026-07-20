"""원본 EDA 차트의 한국어 설명본을 별도 폴더에 생성한다.

원본 ``data/eda`` 파일은 덮어쓰지 않는다. 이 스크립트는 제목·축·표 헤더와
변수명을 한국어로 바꾼 동일한 성격의 차트를 ``data/eda/korea_desc``에 저장한다.
"""

from __future__ import annotations

import platform
from pathlib import Path

import matplotlib

# GUI가 없는 환경(CI/터미널)에서도 이미지 파일을 만들 수 있도록 설정한다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = ROOT_DIR / "data/raw/gym_churn_1M_dataset.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data/processed/churn_preprocessed_full.csv"
OUTPUT_DIR = ROOT_DIR / "data/eda/korea_desc"


LABELS = {
    "Member_ID": "회원 ID",
    "Age": "나이",
    "Gender": "성별",
    "Membership_Type": "회원권 유형",
    "Membership_Start_Date": "가입 시작일",
    "Monthly_Fee": "월 이용료",
    "Monthly_Visits": "월 방문 횟수",
    "Avg_Workout_Duration_Min": "평균 운동 시간(분)",
    "Peak_Hour_Preference": "선호 시간대",
    "Cardio_Preference": "유산소 운동 선호",
    "Treadmill_Avg_Speed_Kmh": "러닝머신 평균 속도(km/h)",
    "Treadmill_Avg_Incline_Pct": "러닝머신 평균 경사(%)",
    "Group_Class_Attendance": "그룹 수업 참석 횟수",
    "PT_Session_Count": "PT 수업 횟수",
    "Supplement_Usage": "보충제 이용",
    "Avg_Equipment_Wait_Time_Min": "평균 기구 대기 시간(분)",
    "Late_Payment_Count": "연체 횟수",
    "Profile_Type": "회원 프로필 유형",
    "Churn": "이탈 여부",
    "Start_Year": "가입 연도",
    "Start_Month": "가입 월",
    "Start_Weekday": "가입 요일",
    "Membership_Days": "가입 경과 일수",
    "Cardio_Preference_Elliptical": "유산소 선호: 엘립티컬",
    "Cardio_Preference_No Preference": "유산소 선호: 없음",
    "Cardio_Preference_Rowing": "유산소 선호: 로잉",
    "Cardio_Preference_Treadmill": "유산소 선호: 러닝머신",
    "Supplement_Usage_Full Stack": "보충제: 풀 스택",
    "Supplement_Usage_No Protein Supplements": "보충제: 단백질 보충제 없음",
    "Supplement_Usage_Pre-Workout": "보충제: 프리워크아웃",
    "Supplement_Usage_Whey Protein": "보충제: 웨이 프로틴",
    "Profile_Type_Drop-out": "프로필: 중도 이탈형",
    "Profile_Type_Hardcore": "프로필: 하드코어형",
    "Profile_Type_Regular": "프로필: 일반형",
}


def korean_label(column: str) -> str:
    """알려진 변수명은 한글로, 그 외 변수명은 원문 그대로 반환한다."""
    return LABELS.get(column, column)


def configure_korean_font() -> None:
    """운영체제별 기본 한글 폰트를 적용한다."""
    candidates = {
        "Darwin": ["AppleGothic"],
        "Windows": ["Malgun Gothic"],
        "Linux": ["NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"],
    }.get(platform.system(), [])
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for font_name in candidates:
        if font_name in available_fonts:
            plt.rcParams["font.family"] = font_name
            break
    else:
        raise RuntimeError("한글 폰트를 찾지 못했습니다. 시스템에 한글 폰트를 설치해 주세요.")

    plt.rcParams["axes.unicode_minus"] = False


def save_boxplot(raw_data: pd.DataFrame) -> None:
    numeric_columns = raw_data.select_dtypes(include="number").columns
    figure, axis = plt.subplots(figsize=(15, 8))
    sns.boxplot(data=raw_data[numeric_columns], ax=axis)
    axis.set_title("수치형 변수의 이상치 분포")
    axis.set_xlabel("변수")
    axis.set_ylabel("값")
    axis.set_xticks(range(len(numeric_columns)))
    axis.set_xticklabels([korean_label(column) for column in numeric_columns], rotation=45, ha="right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_iqr_table(raw_data: pd.DataFrame) -> None:
    numeric_columns = raw_data.select_dtypes(include="number").columns
    rows = []
    for column in numeric_columns:
        first_quartile = raw_data[column].quantile(0.25)
        third_quartile = raw_data[column].quantile(0.75)
        iqr = third_quartile - first_quartile
        outlier_count = ((raw_data[column] < first_quartile - 1.5 * iqr) | (raw_data[column] > third_quartile + 1.5 * iqr)).sum()
        rows.append([korean_label(column), outlier_count, round(outlier_count / len(raw_data) * 100, 2)])

    iqr_result = pd.DataFrame(rows, columns=["변수", "이상치 개수", "이상치 비율(%)"])
    figure, axis = plt.subplots(figsize=(9, len(iqr_result) * 0.5 + 1.5))
    axis.axis("off")
    axis.set_title("IQR 기준 이상치 현황", pad=14)
    table = axis.table(cellText=iqr_result.values, colLabels=iqr_result.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    figure.savefig(OUTPUT_DIR / "iqr_result.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_numeric_distribution(raw_data: pd.DataFrame) -> None:
    numeric_columns = raw_data.select_dtypes(include="number").columns
    axes = raw_data[numeric_columns].hist(figsize=(18, 12), bins=30)
    for column, axis in zip(numeric_columns, axes.flatten()):
        axis.set_title(korean_label(column))
        axis.set_xlabel("값")
        axis.set_ylabel("빈도")
    figure = plt.gcf()
    figure.suptitle("수치형 변수 분포", y=1.01, fontsize=16)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "numeric_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_categorical_distribution(raw_data: pd.DataFrame) -> None:
    categorical_columns = raw_data.select_dtypes(include=["object", "string"]).columns
    columns_per_row = 3
    row_count = int(np.ceil(len(categorical_columns) / columns_per_row))
    figure, axes = plt.subplots(row_count, columns_per_row, figsize=(6 * columns_per_row, 4 * row_count))
    axes = axes.flatten()

    for index, column in enumerate(categorical_columns):
        axis = axes[index]
        sns.countplot(data=raw_data, x=column, ax=axis)
        axis.set_title(f"{korean_label(column)} 분포")
        axis.set_xlabel(korean_label(column))
        axis.set_ylabel("회원 수")
        axis.tick_params(axis="x", rotation=45)

    for axis in axes[len(categorical_columns):]:
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "categorical_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_raw_correlation_heatmap(raw_data: pd.DataFrame) -> None:
    correlation = raw_data.corr(numeric_only=True)
    correlation = correlation.rename(index=korean_label, columns=korean_label)
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, annot=False, cmap="coolwarm", center=0, square=True, ax=axis)
    axis.set_title("상관관계 히트맵")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_churn_distribution(raw_data: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.countplot(data=raw_data, x="Churn", ax=axis)
    axis.set_title("이탈 여부 분포")
    axis.set_xlabel("이탈 여부 (0=잔존, 1=이탈)")
    axis.set_ylabel("회원 수")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "churn_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_processed_correlation_heatmap() -> None:
    processed_data = pd.read_csv(PROCESSED_DATA_PATH)
    correlation = processed_data.corr(numeric_only=True)
    correlation = correlation.rename(index=korean_label, columns=korean_label)
    figure, axis = plt.subplots(figsize=(16, 14))
    sns.heatmap(correlation, annot=False, cmap="coolwarm", center=0, square=True, ax=axis)
    axis.set_title("전처리 후 상관관계 히트맵")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "correlation_heatmap_2.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_korean_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_data = pd.read_csv(RAW_DATA_PATH)

    save_boxplot(raw_data)
    save_iqr_table(raw_data)
    save_numeric_distribution(raw_data)
    save_categorical_distribution(raw_data)
    save_raw_correlation_heatmap(raw_data)
    save_churn_distribution(raw_data)
    del raw_data

    save_processed_correlation_heatmap()
    print(f"한글 EDA 이미지 7개를 생성했습니다: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
