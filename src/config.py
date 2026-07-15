"""프로젝트 전반에서 공유하는 경로, 데이터 스키마, 기본 실험 상수.

모델별 하이퍼파라미터는 ``configs/model_params.yaml``에 두고, 이 모듈에는
환경에 관계없이 공통으로 사용하는 파일 위치와 컬럼 이름만 둔다.
"""

from __future__ import annotations

from pathlib import Path


# 프로젝트 디렉터리
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EDA_DATA_DIR = DATA_DIR / "eda"
EVALUATION_DATA_DIR = DATA_DIR / "evaluation"
RESULTS_DATA_DIR = DATA_DIR / "results"
RESULT_DATA_PATH = RESULTS_DATA_DIR / "result_data.json"
ROC_DATA_DIR = RESULTS_DATA_DIR / "roc"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"


# 설정·로컬 환경 파일
MODEL_PARAMS_PATH = CONFIGS_DIR / "model_params.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


# 데이터셋 파일
RAW_DATA_FILENAME = "gym_churn_1M_dataset.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
PROCESSED_FULL_DATA_PATH = PROCESSED_DATA_DIR / "churn_preprocessed_full.csv"
PROCESSED_PCT50_DATA_PATH = PROCESSED_DATA_DIR / "churn_preprocessed_pct50.csv"


# 데이터 스키마
ID_COLUMN = "Member_ID"
TARGET_COLUMN = "Churn"
DATE_COLUMN = "Membership_Start_Date"
TARGET_LABELS = {0: "retained", 1: "churned"}

# 원본 데이터에서 모델 입력으로 사용할 수 있는 컬럼이다.
RAW_FEATURE_COLUMNS = (
    "Age",
    "Gender",
    "Membership_Type",
    DATE_COLUMN,
    "Monthly_Fee",
    "Monthly_Visits",
    "Avg_Workout_Duration_Min",
    "Peak_Hour_Preference",
    "Cardio_Preference",
    "Treadmill_Avg_Speed_Kmh",
    "Treadmill_Avg_Incline_Pct",
    "Group_Class_Attendance",
    "PT_Session_Count",
    "Supplement_Usage",
    "Avg_Equipment_Wait_Time_Min",
    "Late_Payment_Count",
    "Profile_Type",
)

# ``DATE_COLUMN``에서 생성하는 파생 피처 이름
DATE_FEATURE_COLUMNS = (
    "Start_Year",
    "Start_Month",
    "Start_Weekday",
    "Membership_Days",
)


# 공통 실험 기본값
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRAINING_FRACTIONS = (1.0, 0.5)
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5


# 학습 횟수 통일 상수 
TRAIN_LOOP_COUNT = 10

__all__ = [
    "CONFIGS_DIR",
    "DATA_DIR",
    "DATE_COLUMN",
    "DATE_FEATURE_COLUMNS",
    "DEFAULT_CLASSIFICATION_THRESHOLD",
    "EDA_DATA_DIR",
    "ENV_PATH",
    "EVALUATION_DATA_DIR",
    "ID_COLUMN",
    "MODELS_DIR",
    "MODEL_PARAMS_PATH",
    "NOTEBOOKS_DIR",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "PROCESSED_FULL_DATA_PATH",
    "PROCESSED_PCT50_DATA_PATH",
    "RANDOM_STATE",
    "RAW_DATA_DIR",
    "RAW_DATA_FILENAME",
    "RAW_DATA_PATH",
    "RAW_FEATURE_COLUMNS",
    "RESULTS_DATA_DIR",
    "RESULT_DATA_PATH",
    "ROC_DATA_DIR",
    "SRC_DIR",
    "TARGET_COLUMN",
    "TARGET_LABELS",
    "TESTS_DIR",
    "TEST_SIZE",
    "TRAIN_LOOP_COUNT",
    "TRAINING_FRACTIONS",
]
