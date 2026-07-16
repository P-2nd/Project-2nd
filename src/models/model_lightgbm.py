# ==========================
# Import
# ==========================
from lightgbm import LGBMClassifier


# ==========================
# Hyper Parameter
# ==========================
RANDOM_STATE = 42

N_ESTIMATORS = 200
LEARNING_RATE = 0.05
MAX_DEPTH = -1
NUM_LEAVES = 31


# ==========================
# Model
# ==========================
model = LGBMClassifier(

    random_state=RANDOM_STATE,

    n_estimators=N_ESTIMATORS,

    learning_rate=LEARNING_RATE,

    max_depth=MAX_DEPTH,

    num_leaves=NUM_LEAVES

)