# ==========================
# Import
# ==========================
from sklearn.ensemble import RandomForestClassifier


# ==========================
# Hyper Parameter
# ==========================
RANDOM_STATE = 42

N_ESTIMATORS = 200
MAX_DEPTH = 7
MIN_SAMPLES_SPLIT = 100
MIN_SAMPLES_LEAF = 50
MAX_FEATURES = "sqrt"


# ==========================
# Model
# ==========================
model = RandomForestClassifier(

    random_state=RANDOM_STATE,

    n_estimators=N_ESTIMATORS,

    max_depth=MAX_DEPTH,

    min_samples_split=MIN_SAMPLES_SPLIT,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    max_features=MAX_FEATURES,

    n_jobs=-1

)
