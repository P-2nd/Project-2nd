# ==========================
# Import
# ==========================
from sklearn.linear_model import LogisticRegression

# ==========================
# Hyper Parameter
# ==========================
RANDOM_STATE = 42
MAX_ITER = 1000
C = 1.0
SOLVER = "lbfgs"

# ==========================
# Model
# ==========================
model = LogisticRegression(
    random_state=RANDOM_STATE,
    max_iter=MAX_ITER,
    C=C,
    solver=SOLVER,
)