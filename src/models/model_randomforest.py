# ==========================
# Import
# ==========================
from sklearn.ensemble import RandomForestClassifier


# ==========================
# Hyper Parameter
# ==========================
RANDOM_STATE = 42

N_ESTIMATORS = 200
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2


# ==========================
# Model
# ==========================
model = RandomForestClassifier(

    random_state=RANDOM_STATE,

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_split=MIN_SAMPLES_SPLIT,

    n_jobs=-1

)