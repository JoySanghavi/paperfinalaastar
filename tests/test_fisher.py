"""
Checks of the Fisher forecast in fisher/ (fast; numpy only). Run from the project root:

    python -m unittest discover -s tests -v

  TestAssembly   fisher_assembly maths on a synthetic data set with a known Fisher matrix:
                 Hartlap / Percival factors, prior, log-space analytic covariance term = 0,
                 pseudo-inverse fallback, missing Step-2b data
  TestConfig     fisher_config: priors = config.yml, derived flags
  TestBundle     fisher/fisher_results.npz: complete, consistent with settings.py and config.yml,
                 converged, and reproduced exactly by fisher_assembly (the notebook's code path)
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fisher"))
import fisher_assembly as fa        # noqa: E402
import fisher_config as fcfg        # noqa: E402
import settings                     # noqa: E402

RESULTS = ROOT / "fisher" / "fisher_results.npz"


def toy_bundle(n_real=20000, n_data=6, n_par=2, seed=0):
    """
    Synthetic Step-1/2 products with a known answer: y ~ N(mu + G^T theta, Σ), Σ diagonal.
    Returns (res, G, Σ): the exact linear-space Fisher matrix is G Σ^-1 G^T.
    """
    rng = np.random.default_rng(seed)
    mu = np.linspace(10.0, 20.0, n_data)
    G = rng.normal(size=(n_par, n_data))
    sd = np.linspace(0.5, 1.0, n_data)
    real = mu + rng.normal(size=(n_real, n_data)) * sd
    res = dict(pk_cov=np.cov(real, rowvar=False), pk_realizations=real, pk_grad=G,
               pk_grad_log=G / mu, pk_model_fid=mu)
    return res, G, np.diag(sd ** 2)


def quiet(f, *args, **kwargs):
    """Call f with its prints (safe_inv warnings) suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


class TestAssembly(unittest.TestCase):
    def test_hartlap_factor(self):
        self.assertAlmostEqual(fa.hartlap_factor(5000, 42), (5000 - 42 - 2) / (5000 - 1))
        self.assertLess(fa.hartlap_factor(100, 50), 1.0)

    def test_hartlap_debiases_small_surveys(self):
        # 60 realisations of 6 band powers: h = 0.88, so inv(sample cov) alone overestimates F
        # by ~13 %. Averaged over many such surveys, assemble's F_mean must be unbiased (±1.3 %
        # over 8 seeds; fewer realisations make inv(sample cov) too heavy-tailed to average).
        n_real, trials = 60, 2000
        res, G, cov = toy_bundle()                      # fixed model; only the noise is redrawn
        rng = np.random.default_rng(1)
        F = []
        for _ in range(trials):
            real = res["pk_model_fid"] + rng.normal(size=(n_real, len(cov))) * np.sqrt(np.diag(cov))
            small = dict(res, pk_realizations=real, pk_cov=np.cov(real, rowvar=False))
            F.append(fa.assemble(small, "linear", "mean_only")["F_mean"])
        F_true = G @ np.linalg.inv(cov) @ G.T
        F_avg = np.mean(F, axis=0)
        np.testing.assert_allclose(np.diag(F_avg), np.diag(F_true), rtol=0.04)
        h = fa.hartlap_factor(n_real, 6)
        self.assertTrue(np.all(np.diag(F_avg) / h > 1.08 * np.diag(F_true)))  # without h: biased

    def test_percival_m1_at_least_one_and_tends_to_one(self):
        for n_real in (200, 1000, 5000):
            self.assertGreaterEqual(fa.percival_m1(n_real, 42, 4), 1.0)
        self.assertAlmostEqual(fa.percival_m1(10 ** 7, 42, 4), 1.0, places=4)
        # Percival et al. (2014) eq. 18 by hand, n_s = 50, n_b = 10, n_p = 4:
        # A = 2 / (39 * 36), B = 38 / (39 * 36), m1 = (1 + 6 B) / (1 + A + 5 B)
        self.assertAlmostEqual(fa.percival_m1(50, 10, 4), 1.0225564, places=6)

    def test_recovers_known_fisher_matrix(self):
        res, G, cov = toy_bundle()
        out = fa.assemble(res, "linear", "mean_only")
        np.testing.assert_allclose(out["F"], G @ np.linalg.inv(cov) @ G.T, rtol=0.05)
        # param_cov = m1 F^-1, sigma = sqrt(diag)
        np.testing.assert_allclose(out["param_cov"], out["m1"] * np.linalg.inv(out["F"]), rtol=1e-12)
        np.testing.assert_allclose(out["sigma"], np.sqrt(np.diag(out["param_cov"])), rtol=1e-12)

    def test_prior_only_tightens(self):
        res, _, _ = toy_bundle()
        out = fa.assemble(res, "linear", "mean_only", f_prior=np.eye(2))
        self.assertTrue(np.all(out["sigma"] <= out["sigma_data"]))
        np.testing.assert_allclose(out["F"] - out["F_data"], np.eye(2), atol=1e-12)

    def test_analytic_covariance_term_zero_in_log_space(self):
        res, _, _ = toy_bundle()
        dC = fa.analytic_cov_deriv(res, "log")
        scale = np.abs(res["pk_cov"] / np.outer(res["pk_model_fid"], res["pk_model_fid"])).max()
        self.assertLess(np.abs(dC).max() / scale, 1e-12)
        out = fa.assemble(res, "log", "both_analytic")
        np.testing.assert_allclose(out["F"], fa.assemble(res, "log", "mean_only")["F"], rtol=1e-10)

    def test_analytic_covariance_term_positive_in_linear_space(self):
        res, _, _ = toy_bundle()
        out = fa.assemble(res, "linear", "both_analytic")
        self.assertTrue(np.all(np.diag(out["F_cov"]) > 0))

    def test_both_numeric_needs_step_2b(self):
        res, _, _ = toy_bundle()
        with self.assertRaises(KeyError):
            fa.assemble(res, "log", "both_numeric")

    def test_unknown_knobs_rejected(self):
        res, _, _ = toy_bundle()
        with self.assertRaises(ValueError):
            fa.assemble(res, "sqrt", "mean_only")
        with self.assertRaises(ValueError):
            fa.assemble(res, "log", "all_terms")

    def test_safe_inv(self):
        C = np.array([[2.0, 0.5], [0.5, 1.0]])
        np.testing.assert_allclose(fa.safe_inv(C), np.linalg.inv(C))
        singular = np.diag([1.0, 1e-14])                        # cond 1e14 > threshold -> pinv
        np.testing.assert_allclose(quiet(fa.safe_inv, singular), np.linalg.pinv(singular))


class TestConfig(unittest.TestCase):
    def test_prior_bounds_are_config_yml_priors(self):
        priors = yaml.safe_load(settings.CONFIG_PATH.read_text())["graph"]["pw"]["simulator"]["priors"]
        self.assertEqual(fcfg.PRIOR_BOUNDS, [(float(lo), float(hi)) for _, lo, hi in priors])

    def test_gaussian_prior_variance(self):
        widths = np.array([hi - lo for lo, hi in fcfg.PRIOR_BOUNDS])
        np.testing.assert_allclose(np.diag(fcfg.F_PRIOR), 12.0 / widths ** 2)

    def test_derived_flag(self):
        self.assertEqual(fcfg.COMPUTE_COV_DERIV, fcfg.FISHER == "both_numeric")

    def test_truth_inside_priors(self):
        for name, v, (lo, hi) in zip(settings.PW_NAMES, settings.TRUE_PW, fcfg.PRIOR_BOUNDS):
            with self.subTest(param=name):
                self.assertTrue(lo <= v <= hi)


@unittest.skipUnless(RESULTS.exists(), "fisher/fisher_results.npz not made yet (python fisher/fisher_forecast.py)")
class TestBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(RESULTS) as z:
            cls.R = {k: z[k] for k in z.files}

    def test_complete(self):
        need = {"pk_cov", "pk_mean_data", "pk_realizations", "k_valid", "subset_valid", "valid_mask",
                "true_k", "true_pw",
                "pk_grad", "pk_grad_log", "pk_model_fid", "fd_steps", "fd_convergence",
                "fisher_matrix", "param_cov", "fisher_sigma", "fisher_sigma_data", "fisher_prior",
                "prior_bounds", "with_prior", "data_space", "fisher_mode"}
        self.assertEqual(need - set(self.R), set())

    def test_shapes_consistent(self):
        R = self.R
        n_real, n_valid = R["pk_realizations"].shape
        self.assertEqual(int(R["valid_mask"].sum()), n_valid)
        self.assertEqual(R["k_valid"].shape, (n_valid,))
        self.assertEqual(R["pk_cov"].shape, (n_valid, n_valid))
        self.assertEqual(R["pk_grad"].shape, (len(settings.PW_NAMES), n_valid))
        # data vector = the band powers of every UV subset, concatenated (as pk_obs)
        n_sub, n_bins = settings.N_SUBSETS, len(R["true_k"])
        self.assertEqual(R["valid_mask"].shape, (n_sub * n_bins,))
        np.testing.assert_allclose(R["k_valid"], np.tile(R["true_k"], n_sub)[R["valid_mask"]])
        np.testing.assert_array_equal(R["subset_valid"],
                                      np.repeat(np.arange(n_sub), n_bins)[R["valid_mask"]])
        self.assertEqual(set(R["subset_valid"].tolist()), set(range(n_sub)))

    def test_matches_settings_and_config(self):
        np.testing.assert_allclose(self.R["true_pw"][0], settings.TRUE_PW)
        np.testing.assert_allclose(self.R["prior_bounds"], fcfg.PRIOR_BOUNDS)

    def test_covariance_positive_definite(self):
        self.assertTrue(np.all(np.linalg.eigvalsh(self.R["pk_cov"]) > 0))

    def test_gradient_converged_and_linear_in_A(self):
        R = self.R
        self.assertTrue(np.all(R["fd_convergence"][np.isfinite(R["fd_convergence"])] < 0.01))
        # band power is linear in A, so d<P>/d log10(A) = ln(10) <P>
        rel = np.abs(R["pk_grad"][0] - np.log(10) * R["pk_model_fid"]) / np.abs(np.log(10) * R["pk_model_fid"])
        self.assertLess(np.median(rel), 1e-2)
        # the forecast uses the log-space gradient: ln P is exactly linear in log10(A), so
        # d<ln P>/d log10(A) = ln(10) in every bin (paired seeds: no Monte-Carlo scatter)
        np.testing.assert_allclose(R["pk_grad_log"][0], np.log(10), rtol=1e-4)
        if "fd_convergence_log" in R:
            conv = R["fd_convergence_log"]
            self.assertTrue(np.all(conv[np.isfinite(conv)] < 0.01))

    def test_assembly_reproduces_stored_forecast(self):
        R = self.R
        knobs = (fcfg.DATA_SPACE, fcfg.FISHER, fcfg.WITH_PRIOR)
        if "data_space" in R and (str(R["data_space"]), str(R["fisher_mode"]), bool(R["with_prior"])) != knobs:
            self.skipTest("fisher_config knobs differ from the stored forecast; re-run fisher_forecast.py")
        out = quiet(fa.assemble, R, fcfg.DATA_SPACE, fcfg.FISHER, fcfg.F_PRIOR if fcfg.WITH_PRIOR else None)
        np.testing.assert_allclose(out["F"], R["fisher_matrix"], rtol=1e-10)
        np.testing.assert_allclose(out["sigma"], R["fisher_sigma"], rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
