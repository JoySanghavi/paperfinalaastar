"""
Fisher-matrix assembly (STEP 3), shared by fisher_forecast.py and fisher_analysis.ipynb,
so the forecast stored in fisher_results.npz and the one shown in the notebook are computed
by the same code.

Input `res`: the Step-1/2 products of fisher_forecast.py, as a dict with the keys of
fisher_results.npz (pk_cov, pk_realizations, pk_grad, pk_grad_log, pk_model_fid and, for
FISHER = "both_numeric", pk_cov_grad_lin / pk_cov_grad_log).

    F         = G (h C^-1) G^T  [+ 1/2 Tr(C^-1 dC_a C^-1 dC_b)]  [+ F_prior]
    param_cov = m1 F^-1

  y, C, G : data vector (P or ln P), its covariance, and d<y>/dtheta (n_par, n_data)
  h       : Hartlap et al. (2007) debiasing of the precision matrix
  m1      : Dodelson & Schneider (2013) / Percival et al. (2014) factor on the parameter
            covariance — complementary to h; apply each exactly once.
"""

import numpy as np

COND_THRESHOLD = 1e12        # cond(C) above this -> pseudo-inverse (with a warning)


def safe_inv(C, name="matrix", threshold=COND_THRESHOLD):
    """inv(C), or the Moore-Penrose pseudo-inverse with a warning if cond(C) > threshold."""
    cond = np.linalg.cond(C)
    if cond > threshold:
        print(f"[warn] cond({name}) = {cond:.2e} > {threshold:.0e} -- using pinv")
        return np.linalg.pinv(C)
    return np.linalg.inv(C)


def hartlap_factor(n_real, n_data):
    """Hartlap factor h: unbiased precision matrix = h * inv(sample covariance)."""
    return (n_real - n_data - 2) / (n_real - 1)


def percival_m1(n_real, n_data, n_par):
    """Dodelson-Schneider / Percival m1 >= 1 on the parameter covariance."""
    nu = n_real - n_data
    a = 2.0 / ((nu - 1) * (nu - 4))
    b = (nu - 2) / ((nu - 1) * (nu - 4))
    return (1.0 + b * (n_data - n_par)) / (1.0 + a + b * (n_par + 1))


def data_vector(res, space):
    """(C, G) of the data vector in `space`: "linear" (y = P) or "log" (y = ln P)."""
    if space == "linear":
        return res["pk_cov"], res["pk_grad"]                  # Cov(P), d<P>/dtheta
    if space == "log":                                        # Cov(ln P), d<ln P>/dtheta (Jensen)
        return np.cov(np.log(res["pk_realizations"].astype(np.float64)), rowvar=False), res["pk_grad_log"]
    raise ValueError(f"space={space!r}, expected 'linear' or 'log'")


def cov_fisher_term(Cinv, dCov):
    """Covariance term 1/2 Tr(C^-1 dC_a C^-1 dC_b) for dCov (n_par, n_data, n_data)."""
    M = [Cinv @ d for d in dCov]
    return np.array([[0.5 * np.trace(Ma @ Mb) for Mb in M] for Ma in M])


def analytic_cov_deriv(res, space):
    """
    dC/dtheta of the Gaussian cosmic-variance model Cov(P)_ij ∝ P_i P_j, in `space`.

    linear: dC_a = C (g_ai + g_aj),  g_a = d ln<P> / dtheta_a.
    log:    delta method, Cov(ln P) = C / (mu_i mu_j), whose derivative is identically zero
            for this model — so with DATA_SPACE = "log", FISHER = "both_analytic" gives the
            same forecast as "mean_only". It is computed here, not assumed; use
            "both_numeric" for the re-simulated (non-zero) estimate.
    """
    C, mu = res["pk_cov"], res["pk_model_fid"]
    g = res["pk_grad"] / mu
    dC_lin = np.stack([C * (ga[:, None] + ga[None, :]) for ga in g])
    if space == "linear":
        return dC_lin
    if space == "log":
        JJ = np.outer(1.0 / mu, 1.0 / mu)
        C_log = C * JJ
        return np.stack([d * JJ - ga[:, None] * C_log - C_log * ga[None, :] for d, ga in zip(dC_lin, g)])
    raise ValueError(f"space={space!r}, expected 'linear' or 'log'")


def assemble(res, space, fisher, f_prior=None):
    """
    Fisher matrix and forecast errors.

    space:   "linear" | "log"                                  (fisher_config.DATA_SPACE)
    fisher:  "mean_only" | "both_analytic" | "both_numeric"    (fisher_config.FISHER)
    f_prior: (n_par, n_par) prior Fisher matrix added to F, or None (data only)

    Returns a dict: C, G, hartlap, m1, F_mean, F_cov, F_data, F, param_cov, sigma (with the
    prior), sigma_data (data only).
    """
    n_real, n_data = res["pk_realizations"].shape
    C, G = data_vector(res, space)
    n_par = G.shape[0]

    h = hartlap_factor(n_real, n_data)
    Cinv = h * safe_inv(C, f"C ({space})")
    F_mean = G @ Cinv @ G.T                                   # term 1 (mean / signal)

    if fisher == "mean_only":                                 # term 2 excluded by definition
        F_cov = np.zeros((n_par, n_par))
    elif fisher == "both_analytic":
        F_cov = cov_fisher_term(Cinv, analytic_cov_deriv(res, space))
    elif fisher == "both_numeric":
        key = "pk_cov_grad_log" if space == "log" else "pk_cov_grad_lin"
        if key not in res:
            raise KeyError(f"{key} not in the results: run fisher_forecast.py with "
                           f"FISHER = 'both_numeric' (Step 2b) first")
        F_cov = cov_fisher_term(Cinv, res[key])
    else:
        raise ValueError(f"fisher={fisher!r}")

    F_data = F_mean + F_cov
    F = F_data if f_prior is None else F_data + f_prior
    m1 = percival_m1(n_real, n_data, n_par)
    param_cov = m1 * safe_inv(F, "F")
    return dict(
        C=C, G=G, hartlap=h, m1=m1, F_mean=F_mean, F_cov=F_cov, F_data=F_data, F=F,
        param_cov=param_cov, sigma=np.sqrt(np.diag(param_cov)),
        sigma_data=np.sqrt(np.diag(m1 * safe_inv(F_data, "F_data"))),
    )
