"""
Shared Fisher-forecast configuration  —  EDIT THESE THREE KNOBS IN ONE PLACE.

Both `fisher_forecast.py` and `fisher_analysis.ipynb` import this module, so you
no longer keep flags in sync across two files. Everything else is derived below.
"""

# Data-vector representation used to build the Fisher matrix:
#   "log"    : ln P  (default) — Gaussianises the chi^2 band powers, makes the
#              covariance ~parameter-independent, and is well-conditioned.
#   "linear" : band power P.
DATA_SPACE = "log"

# Which Fisher terms to include (and, if both, how dC/dtheta is obtained):
#   "mean_only"     : signal term only                                  (term 1)
#   "both_analytic" : + covariance term via the Gaussian cosmic-variance model
#                     (identically zero in log space, i.e. the same as "mean_only" there)
#   "both_numeric"  : + covariance term via re-simulated dC/dtheta (Step 2b; expensive)
FISHER = "both_analytic"

# Add the pw priors from config.yml to the Fisher matrix (False = data-only forecast)
WITH_PRIOR = True

# --- derived flag (do not edit) ---
COMPUTE_COV_DERIV = FISHER == "both_numeric"            # fisher_forecast.py: run Step 2b?

assert DATA_SPACE in {"linear", "log"}, f"DATA_SPACE={DATA_SPACE!r}"
assert FISHER in {"mean_only", "both_analytic", "both_numeric"}, f"FISHER={FISHER!r}"

# --- pw priors, read from config.yml (graph.pw simulator); order = theta order ---
#   [log10(A), n1, n2, log10(k_break)]
from pathlib import Path
import numpy as np
import yaml

_PRIORS = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yml").read_text()
                         )["graph"]["pw"]["simulator"]["priors"]
assert all(p[0] == "uniform" for p in _PRIORS), f"only uniform priors supported: {_PRIORS}"
PRIOR_BOUNDS = [(float(lo), float(hi)) for _, lo, hi in _PRIORS]

# Gaussian approximation of each uniform prior (variance = width^2 / 12), added to F
F_PRIOR = np.diag([12.0 / (hi - lo) ** 2 for lo, hi in PRIOR_BOUNDS])
