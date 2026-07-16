# ==========================
# Import
# ==========================
from sklearn.neighbors import KNeighborsClassifier


# ==========================
# Hyper Parameter
# ==========================
N_NEIGHBORS = 10
WEIGHTS = "uniform"
METRIC = "minkowski"


# ==========================
# Model
# ==========================
model = KNeighborsClassifier(
    n_neighbors=N_NEIGHBORS,
    weights=WEIGHTS,
    metric=METRIC,
)