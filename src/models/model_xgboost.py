# ==========================
# Import
# ==========================
from xgboost import XGBClassifier


# ==========================
# Hyper Parameter
# ==========================
RANDOM_STATE = 42

N_ESTIMATORS = 200
MAX_DEPTH = 5
LEARNING_RATE = 0.05
SUBSAMPLE = 0.8


# ==========================
# Model
# ==========================
model = XGBClassifier(

    random_state=RANDOM_STATE,

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    learning_rate=LEARNING_RATE,

    subsample=SUBSAMPLE,

    eval_metric="logloss"

)