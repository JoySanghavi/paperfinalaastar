"""
Fisher forecast — STEP 1 (covariance) + STEP 2 (band-power gradient).

We forecast the broken-power-law parameters theta = [log10(A), n1, n2, log10(k_break)]
at FIXED true gains (the gains are NOT forecast parameters). The data vector is
the band-power spectrum measured from the science observation, for each UV subset,
concatenated — the pk_obs the SBI estimator sees:

    d = [P_0(k), P_1(k)]   measured from the dirty images obsx   (length n_subsets * n_bins)

The two subsets image the same sky through different UV coverage, so their band powers are
correlated; the covariance below carries that. Bins with no sampled UV mode in a subset (FFT
round-off only, the rule of the SBI embedding) are dropped for that subset.

The instrument, the truth and the thermal noise are those of the SBI pipeline: simulators
from src/model.py, numbers from src/settings.py (NOISE_STD > 0 adds the same Hermitian
noise per UV pixel as model.ThermalNoiseUV; the "both_analytic" covariance derivative
assumes cosmic variance only, so prefer "both_numeric" when the noise is significant).

STEP 1 — Covariance C.
    Draw N_REAL independent Gaussian density-field realizations from the SAME true
    power spectrum, push each through the SAME fixed-gain pipeline, measure the band
    power, and take the sample covariance. Because the gains are fixed, the UV
    sampling maps model.uv_maps(true_gains) and model.UNIT_UV are identical in every
    realization, so they are computed once outside the loop.

STEP 2 — Gradient dmu/dtheta of the model mean band power.
    For each parameter, central finite difference of the MC-mean band power:
        dmu/dtheta_i = ( mu(theta + h_i e_i) - mu(theta - h_i e_i) ) / (2 h_i)
    Both legs reuse the SAME white-noise realizations (common random numbers), so
    the field changes only through sqrt(P(k;theta)) and the sample variance cancels
    in the difference — a low-variance derivative even at modest realization counts.

Produces a SINGLE bundled archive  fisher/fisher_results.npz  (written once at
the end, so a failed run leaves no half-written / mismatched outputs). Its keys:
  Step 1:
    pk_cov           — (n_valid, n_valid) sample covariance of band powers
    pk_mean_data     — (n_valid,)         MC-mean band power (the data mean)
    pk_realizations  — (N_REAL, n_valid)  all realizations (for Hartlap etc.)
    k_valid          — (n_valid,)         k-centres of the kept (finite) bins
    subset_valid     — (n_valid,)         UV subset of each kept bin
    valid_mask       — (n_subsets * n_bins,) bool  which raw bins were kept, subset-major
    true_pw          — (1, 4) true broken-power-law parameters (reference)
    true_k           — (n_bins,) all bin centres of one subset (reference)
  Step 2:
    pk_grad          — (4, n_valid) d<P>/dtheta, rows ordered as theta
    pk_grad_log      — (4, n_valid) d<ln P>/dtheta (Jensen-correct log-space gradient)
    pk_model_fid     — (n_valid,)   model mean band power at the fiducial theta
    fd_steps         — (4,)         finite-difference steps used per parameter
    fd_convergence   — (4,)         ||grad(h/2)-grad(h)||/||grad(h)|| per parameter
                                    (NaN if CHECK_FD_CONVERGENCE=False)
    fd_convergence_log — (4,)       the same for pk_grad_log (the gradient used with
                                    DATA_SPACE = "log")
  Step 2b (only if COMPUTE_COV_DERIV=True):
    pk_cov_grad_lin  — (4, n_valid, n_valid) numerical dCov(P)/dtheta
    pk_cov_grad_log  — (4, n_valid, n_valid) numerical dCov(ln P)/dtheta
  Step 3 (fisher_assembly.assemble; knobs DATA_SPACE / FISHER from fisher_config.py):
    fisher_matrix    — (4, 4)  Fisher matrix F (data + prior if WITH_PRIOR)
    param_cov        — (4, 4)  parameter covariance (Hartlap + Percival m1)
    fisher_sigma     — (4,)    1-sigma marginalized forecast errors
    fisher_sigma_data— (4,)    the same without the prior
    data_space, fisher_mode    the knobs this forecast was assembled with

Forecast knobs live in fisher_config.py (shared with fisher_analysis.ipynb, set ONCE):
  DATA_SPACE : "log" (default; ln P) or "linear" (band power P).
  FISHER     : "mean_only" | "both_analytic" | "both_numeric".  "both_numeric"
               triggers Step 2b and uses the re-simulated covariance derivative.

Device: model.device (the GPU if there is one, else the CPU). The forecast is a Monte-Carlo
estimate, and torch draws different random numbers on CPU and GPU for the same seeds, so a run
on the other device gives sigmas that differ at the Monte-Carlo level (a few per cent), not a bug.

Run from the project root (paths are relative to this file, so any directory works):
    python fisher/fisher_forecast.py [output.npz]      # default: fisher/fisher_results.npz
"""

import sys
import numpy as np
import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE      = Path(__file__).resolve().parent          # fisher/
ROOT      = HERE.parent                              # project root
SRC       = ROOT / "src"                             # simulators, settings, padded tensor
OUT_PATH  = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fisher_results.npz"

sys.path.insert(0, str(SRC))
import functionlistnew as func
import model
import settings

# ---------------------------------------------------------------------------
# Forecast / Monte-Carlo settings
# ---------------------------------------------------------------------------
N_REAL     = 5000          # number of field realizations for the covariance
BATCH      = 250           # realizations per batch (memory/speed tradeoff)
MC_SEED    = 1234          # master seed for the realization draws (reproducible)

# --- Step 2 (gradient) settings ---
DERIV_N_REAL = 2000        # realizations averaged for each model-mean band power
DERIV_BATCH  = 250         # batch size for the derivative evaluations
DERIV_SEED0  = 7000        # base seed for the common-random-number batches
# Central finite-difference steps, in the SAME order as theta:
#   [log10(A), n1, n2, log10(k_break)]   (log-space steps for A and k_break)
# k_break: P(k) has a kink there, so a mode crossing it within ±h makes the difference step-
# dependent; 0.0125 keeps grad(h) vs grad(h/2) below FD_CONV_TOL.
FD_STEPS     = np.array([0.025, 0.025, 0.025, 0.0125])
# Convergence check: also evaluate the gradient at h/2 and compare. A central
# difference has O(h^2) truncation error, so a large grad(h) vs grad(h/2) change flags
# an unconverged step. Costs an extra 4*2 model-mean evals; set False to skip.
CHECK_FD_CONVERGENCE = True
FD_CONV_TOL          = 0.01      # warn if ||grad(h/2)-grad(h)|| / ||grad(h)|| > this

# --- Step 3 (Fisher assembly) configuration ---
# The two forecast knobs DATA_SPACE and FISHER live in fisher_config.py (set ONCE
# there, shared with fisher_analysis.ipynb); COMPUTE_COV_DERIV is derived from FISHER.
# The assembly itself is fisher_assembly.assemble (also used by the notebook).
sys.path.insert(0, str(HERE))
from fisher_config import (DATA_SPACE, FISHER, COMPUTE_COV_DERIV,
                           WITH_PRIOR, F_PRIOR, PRIOR_BOUNDS)
import fisher_assembly

# --- Step 2b (numerical covariance derivative) tuning ---
N_COV_DERIV     = 2000       # realizations per leg per parameter (CRN-paired legs)
COV_DERIV_SEED0 = 20000      # base seed for the covariance-derivative CRN batches

device = model.device

# ---------------------------------------------------------------------------
# True parameters and simulators — shared with the SBI pipeline
# ---------------------------------------------------------------------------
# True gains are FIXED here (not forecast): same true_z as scripts/gen_mock_data.py.
true_z = settings.true_z(model.N_UNKNOWN)                     # (1, 2 N_UNKNOWN)
true_log_A, true_n1, true_n2, true_log_k_break = settings.TRUE_PW

# Fiducial (and the fixed true gains) must lie inside the config.yml priors
settings.check_truth_inside_priors(true_z[0].tolist())

print(f"Padded tensor shape : {tuple(model.padded.shape)}  (nant={model.NANT})")

PW_sim      = model.density_sim
BT_model    = model.brightness
PK_measurer = model.pk_measurer
NOISE_STD   = model.NOISE_STD                      # mK units (converted from Jy if NOISE_UNIT = 'Jy')
foreground  = settings.SCIENCE_FOREGROUND_FLUX * model.BEAM_CENTRE_GAUSSIAN   # (H, W)

# ---------------------------------------------------------------------------
# FIXED gains  ->  UV sampling maps are constant across all realizations
# ---------------------------------------------------------------------------
true_gains = func.theta_to_gains(true_z.to(device), settings.KNOWN_ANT)   # (1, 2*nant)
stacked_uv = model.uv_maps(true_gains)                   # (1, 2, 128, 128), fixed
unit_uv    = model.UNIT_UV                               # (1, 2, 128, 128), fixed

# True power-spectrum parameters as a theta vector [log10(A), n1, n2, log10(k_break)].
theta_true = np.array([true_log_A, true_n1, true_n2, true_log_k_break])


def unpack_theta(theta: np.ndarray, batch_size: int = 1):
    """theta = [log10(A), n1, n2, log10(k_break)] -> the four (A, n1, n2, k_break)
    tensors of length batch_size on `device`, ready for PW_sim.sample_field; the
    conversion is func.pw_to_params, as in model.DensityFromPw (float32, as falcon)."""
    pw = torch.as_tensor(theta, dtype=torch.float32, device=device).expand(batch_size, -1)
    return func.pw_to_params(pw)


def cov_t(x: torch.Tensor) -> torch.Tensor:
    """Sample covariance of x (n_obs, n_var) over rows, ddof=1 -- matches
    np.cov(x, rowvar=False). Computed in float64 (band powers span ~10 decades) and
    kept in torch so the reduction stays on the GPU."""
    x  = x.double()
    xc = x - x.mean(dim=0, keepdim=True)
    return (xc.T @ xc) / (x.shape[0] - 1)


@torch.no_grad()
def band_powers(delta_batch: torch.Tensor) -> torch.Tensor:
    """
    Push a batch of density fields through the FIXED-gain instrument pipeline and
    return the band-power data vector (the 2 UV subsets concatenated, as pk_obs).

    delta_batch : (b, 128, 128) real density fields
    returns     : (b, n_subsets * n_bins)   measured P(k), subset-major
    """
    Tb       = BT_model(delta_batch)                     # (b, 128, 128)
    image    = Tb + foreground                           # as model.SkyImageFromBrightness
    uvimage  = func.image_to_uv(image)                   # (b, 128, 128) complex, FFT order

    tracked  = func.apply_uv_sampling(stacked_uv, uvimage)             # (b, 2, H, W)
    # Normalise by the unit (all-gain) UV map. Unsampled cells have unit_uv == 0;
    # guard the division so no NaN/inf is ever created (cleaner than .nan_to_num after).
    sampled  = unit_uv.abs() > 0
    safe_uv  = torch.where(sampled, unit_uv, torch.ones_like(unit_uv))
    normed   = torch.where(sampled, tracked / safe_uv, torch.zeros_like(tracked))
    if NOISE_STD > 0:                                    # as model.ThermalNoiseUV
        # (skipped at zero noise so the RNG stream — and the forecast — stay unchanged)
        white  = torch.randn(normed.shape, device=device)
        normed = normed + torch.fft.fft2(white, norm="ortho") * (NOISE_STD * model.UV_NOISE_SCALE)
    obsx     = func.uv_to_image(normed)                                # (b, 2, H, W)

    b, s, H, W = obsx.shape
    _, pk_mean, _ = PK_measurer(obsx.real.reshape(b * s, H, W))
    # Bins with no sampled UV mode hold only FFT round-off: set them NaN (same rule as the
    # SBI embedding); valid_mask below drops them.
    pk = func.drop_roundoff_bins(pk_mean.reshape(b, s, -1))
    return pk.reshape(b, -1)                              # (b, s * n_bins) subset-major


# ---------------------------------------------------------------------------
# Monte-Carlo loop  (gains fixed; only the sky realization changes)
# ---------------------------------------------------------------------------
torch.manual_seed(MC_SEED)
rows = []
done = 0
with torch.no_grad():                                    # no autograd graph in the MC loop
    while done < N_REAL:
        b = min(BATCH, N_REAL - done)
        delta = PW_sim.sample_field(*unpack_theta(theta_true, b), seed=None)  # fresh draw
        rows.append(band_powers(delta))                  # (b, n_bins) torch, on device
        done += b
        print(f"  realizations: {done}/{N_REAL}")

D = torch.cat(rows, dim=0)                               # (N_REAL, n_bins) torch

# ---------------------------------------------------------------------------
# Keep only finite bins (empty k-bins are NaN and identical every realization).
# Reductions stay in torch on the GPU; we move only the small results to numpy.
# valid_mask_t (torch) is reused by Step 2 / Step 2b to mask their realizations.
# ---------------------------------------------------------------------------
valid_mask_t = torch.isfinite(D).all(dim=0)              # (n_bins,) bool torch
D_valid_t    = D[:, valid_mask_t]                        # (N_REAL, n_valid) torch
n_valid      = int(valid_mask_t.sum())

pk_mean_data = D_valid_t.double().mean(dim=0).cpu().numpy()    # (n_valid,)
pk_cov       = cov_t(D_valid_t).cpu().numpy()                  # (n_valid, n_valid)
valid_mask   = valid_mask_t.cpu().numpy()                      # (n_subsets * n_bins,)
k_all        = PK_measurer.k_centers.detach().cpu().numpy()    # (n_bins,)
n_sub        = settings.N_SUBSETS
k_valid      = np.tile(k_all, n_sub)[valid_mask]
subset_valid = np.repeat(np.arange(n_sub), len(k_all))[valid_mask]
D_valid      = D_valid_t.cpu().numpy()                         # (N_REAL, n_valid) for storage

# Hartlap factor for the later unbiased inverse  C^-1_unbiased = h * inv(C)
hartlap = fisher_assembly.hartlap_factor(N_REAL, n_valid)

# ---------------------------------------------------------------------------
# Collect Step 1 products. Everything is bundled into ONE archive
# (fisher_results.npz) at the very end, so a run either writes all arrays or none
# -- no half-written / mismatched directory if it fails partway.
# ---------------------------------------------------------------------------
results = dict(
    pk_cov=pk_cov, pk_mean_data=pk_mean_data, pk_realizations=D_valid,
    k_valid=k_valid, subset_valid=subset_valid, valid_mask=valid_mask, true_k=k_all,
    true_pw=np.array([[true_log_A, true_n1, true_n2, true_log_k_break]]),
)

print("\n=== Step 1: data covariance built ===")
print(f"Realizations            : {N_REAL}")
print(f"Bins (raw -> finite)    : {n_sub} x {len(k_all)} -> {n_valid}  "
      f"(per subset: {np.bincount(subset_valid, minlength=n_sub).tolist()})")
print(f"k range (finite bins)   : [{k_valid.min():.4f}, {k_valid.max():.4f}]")
print(f"Covariance shape        : {pk_cov.shape}")
print(f"Condition number        : {np.linalg.cond(pk_cov):.3e}")
print(f"Hartlap factor (N>p+2)  : {hartlap:.4f}")


# ===========================================================================
# STEP 2 — Gradient of the model mean band power, dmu/dtheta
# ===========================================================================
# theta = [log10(A), n1, n2, log10(k_break)].  We perturb in THIS space, so the
# forecast errors will be on these (log-)parameters directly.
theta0      = theta_true                        # fiducial point (defined once, near the top)
PARAM_NAMES = list(settings.PW_NAMES)            # ["log10_A", "n1", "n2", "log10_k_break"]

# Fixed list of (seed, batch_size) pairs -> common random numbers. The SAME list
# is reused for every theta evaluation (both + and - legs), so paired fields
# share white noise and only sqrt(P(k;theta)) differs.
deriv_seeds = []
done = 0
s = DERIV_SEED0
while done < DERIV_N_REAL:
    b = min(DERIV_BATCH, DERIV_N_REAL - done)
    deriv_seeds.append((s, b))
    s += 1
    done += b


@torch.no_grad()
def model_band_power(theta: np.ndarray):
    """
    MC-mean band power at parameters theta, on the kept (valid) bins.

    Uses the FIXED-gain `band_powers` pipeline and the shared `deriv_seeds`
    (common random numbers), so differences between two `theta` values are
    dominated by the parameter change, not by Monte-Carlo scatter.

    theta   : (4,) [log10(A), n1, n2, log10(k_break)]
    returns : (mean_P, mean_lnP), each (n_valid,) -- the linear mean <P> and the
              log mean <ln P>. The log mean is needed for a rigorous log-space
              forecast: <ln P> != ln<P> (Jensen), so the log-space gradient must be
              d<ln P>/dtheta, not d ln<P>/dtheta.
    """
    rows = []
    for seed, b in deriv_seeds:
        delta = PW_sim.sample_field(*unpack_theta(theta, b), seed=seed)
        rows.append(band_powers(delta))                     # (b, n_bins) torch
    allreal = torch.cat(rows, dim=0)[:, valid_mask_t].double()   # (DERIV_N_REAL, n_valid)
    return (allreal.mean(dim=0).cpu().numpy(),
            torch.log(allreal).mean(dim=0).cpu().numpy())


# Fiducial model band power (paired with the same seeds for consistency).
pk_model_fid, pk_logmean_fid = model_band_power(theta0)    # (n_valid,) each

# Central finite differences, one parameter at a time, in BOTH linear and log space.
pk_grad     = np.zeros((len(theta0), n_valid))             # d<P>/dtheta
pk_grad_log = np.zeros((len(theta0), n_valid))             # d<ln P>/dtheta (Jensen-correct)
for i in range(len(theta0)):
    tp = theta0.copy(); tp[i] += FD_STEPS[i]
    tm = theta0.copy(); tm[i] -= FD_STEPS[i]
    Pp, Lp = model_band_power(tp)
    Pm, Lm = model_band_power(tm)
    pk_grad[i]     = (Pp - Pm) / (2.0 * FD_STEPS[i])
    pk_grad_log[i] = (Lp - Lm) / (2.0 * FD_STEPS[i])
    print(f"  d mu / d {PARAM_NAMES[i]:13s} done  "
          f"(|grad| = {np.linalg.norm(pk_grad[i]):.3e})")

# Finite-difference convergence check: re-evaluate the gradient at h/2 (same CRN
# seeds, so the difference reflects the step size, not MC scatter) and compare. A
# central difference has O(h^2) truncation error, so a large grad(h) vs grad(h/2)
# change means h is too big (or too small -> noise-dominated).
fd_convergence = np.full(len(theta0), np.nan)
fd_convergence_log = np.full(len(theta0), np.nan)
if CHECK_FD_CONVERGENCE:
    print("Finite-difference convergence (grad at h vs h/2):")
    for i in range(len(theta0)):
        tp = theta0.copy(); tp[i] += 0.5 * FD_STEPS[i]
        tm = theta0.copy(); tm[i] -= 0.5 * FD_STEPS[i]
        Pp, Lp = model_band_power(tp)
        Pm, Lm = model_band_power(tm)
        grad_half = (Pp - Pm) / (2.0 * 0.5 * FD_STEPS[i])      # central diff at h/2
        grad_half_log = (Lp - Lm) / (2.0 * 0.5 * FD_STEPS[i])
        fd_convergence[i] = (np.linalg.norm(grad_half - pk_grad[i])
                             / max(np.linalg.norm(pk_grad[i]), 1e-30))
        fd_convergence_log[i] = (np.linalg.norm(grad_half_log - pk_grad_log[i])
                                 / max(np.linalg.norm(pk_grad_log[i]), 1e-30))
        worst = max(fd_convergence[i], fd_convergence_log[i])
        flag = "  <-- NOT converged" if worst > FD_CONV_TOL else ""
        print(f"  {PARAM_NAMES[i]:13s}  ||grad(h/2)-grad(h)||/||grad(h)|| = "
              f"{fd_convergence[i]:.3e} (P), {fd_convergence_log[i]:.3e} (ln P){flag}")
    if max(np.nanmax(fd_convergence), np.nanmax(fd_convergence_log)) > FD_CONV_TOL:
        print(f"  WARNING: a finite difference changed by > {FD_CONV_TOL:.0%} between h and "
              f"h/2 -- the step is not converged; adjust FD_STEPS for that parameter.")

# Analytic cross-check: band power is linear in A, so d mu / d log10(A) = ln(10) * mu.
logA_check = np.log(10.0) * pk_model_fid
rel_err = np.median(np.abs(pk_grad[0] - logA_check) / np.abs(logA_check))

results.update(pk_grad=pk_grad, pk_grad_log=pk_grad_log, pk_model_fid=pk_model_fid,
               fd_steps=FD_STEPS, fd_convergence=fd_convergence,
               fd_convergence_log=fd_convergence_log)

print("\n=== Step 2: band-power gradient built ===")
print(f"Realizations per leg    : {DERIV_N_REAL}")
print(f"FD steps                : {dict(zip(PARAM_NAMES, FD_STEPS.tolist()))}")
print(f"Gradient shape          : {pk_grad.shape}  (rows = {PARAM_NAMES})")
print(f"log10(A) deriv vs ln(10)*mu  median rel. err : {rel_err:.2e}  "
      f"(should be ~0; band power is linear in A)")


# ===========================================================================
# STEP 2b — Numerical covariance derivative dC/dtheta  (for FISHER="both_numeric")
# ===========================================================================
# Re-estimate the band-power covariance at theta +/- h_i and central-difference it.
# Both legs reuse the SAME white-noise seeds (common random numbers), so the paired
# covariances differ mainly through the parameter change rather than MC scatter --
# essential, since a raw difference of two noisy sample covariances is very noisy.
# Saved in BOTH spaces so the notebook can pick by DATA_SPACE.
if COMPUTE_COV_DERIV:
    cov_seeds = []
    done = 0
    s = COV_DERIV_SEED0
    while done < N_COV_DERIV:
        b = min(DERIV_BATCH, N_COV_DERIV - done)
        cov_seeds.append((s, b)); s += 1; done += b

    @torch.no_grad()
    def realizations_at(theta: np.ndarray) -> torch.Tensor:
        """(N_COV_DERIV, n_valid) band-power realizations at theta, on CRN seeds (torch)."""
        rows = []
        for seed, b in cov_seeds:
            delta = PW_sim.sample_field(*unpack_theta(theta, b), seed=seed)
            rows.append(band_powers(delta))                 # (b, n_bins) torch
        return torch.cat(rows, dim=0)[:, valid_mask_t]      # (N_COV_DERIV, n_valid) torch

    pk_cov_grad_lin = np.zeros((len(theta0), n_valid, n_valid))
    pk_cov_grad_log = np.zeros((len(theta0), n_valid, n_valid))
    for i in range(len(theta0)):
        tp = theta0.copy(); tp[i] += FD_STEPS[i]
        tm = theta0.copy(); tm[i] -= FD_STEPS[i]
        Dp = realizations_at(tp)
        Dm = realizations_at(tm)
        pk_cov_grad_lin[i] = ((cov_t(Dp) - cov_t(Dm)) / (2.0 * FD_STEPS[i])).cpu().numpy()
        pk_cov_grad_log[i] = ((cov_t(torch.log(Dp)) - cov_t(torch.log(Dm)))
                              / (2.0 * FD_STEPS[i])).cpu().numpy()
        print(f"  dC/d {PARAM_NAMES[i]:13s} done  "
              f"(|dC_lin|={np.linalg.norm(pk_cov_grad_lin[i]):.2e}, "
              f"|dC_log|={np.linalg.norm(pk_cov_grad_log[i]):.2e})")

    results.update(pk_cov_grad_lin=pk_cov_grad_lin, pk_cov_grad_log=pk_cov_grad_log,
                   n_cov_deriv=np.array(N_COV_DERIV))    # realizations per leg (provenance)
    print(f"\n=== Step 2b: numerical dC/dtheta built ({N_COV_DERIV} CRN realizations/leg) ===")
else:
    print("\nStep 2b skipped (COMPUTE_COV_DERIV=False); FISHER='both_numeric' needs it.")


# ===========================================================================
# STEP 3 — Assemble the Fisher matrix and forecast errors
# ===========================================================================
# Controlled by DATA_SPACE and FISHER (fisher_config.py); the same function builds the
# forecast in fisher_analysis.ipynb. Priors: config.yml pw priors, Gaussian approximation.
fc = fisher_assembly.assemble(results, DATA_SPACE, FISHER, F_PRIOR if WITH_PRIOR else None)

results.update(fisher_matrix=fc["F"], param_cov=fc["param_cov"], fisher_sigma=fc["sigma"],
               fisher_sigma_data=fc["sigma_data"], fisher_prior=F_PRIOR,
               prior_bounds=np.array(PRIOR_BOUNDS), with_prior=np.array(WITH_PRIOR),
               data_space=np.array(DATA_SPACE), fisher_mode=np.array(FISHER))

print("\n=== Step 3: Fisher forecast assembled ===")
print(f"DATA_SPACE = {DATA_SPACE!r}   FISHER = {FISHER!r}   WITH_PRIOR = {WITH_PRIOR}")
print(f"cond(C) = {np.linalg.cond(fc['C']):.2e}   Hartlap = {fc['hartlap']:.5f}   m1 = {fc['m1']:.5f}")
for nm, t, s, sd, (lo, hi) in zip(PARAM_NAMES, theta0, fc["sigma"], fc["sigma_data"], PRIOR_BOUNDS):
    print(f"  {nm:14s}  fiducial = {t:+.4f}   sigma = {s:.4f}   "
          f"(data only {sd:.4f}; prior U[{lo:g}, {hi:g}] -> {(hi - lo) / np.sqrt(12):.4f})")


# ===========================================================================
# SAVE — one bundled archive (all-or-nothing -> no half-written directory)
# ===========================================================================
np.savez_compressed(OUT_PATH, **results)
print(f"\nSaved {len(results)} arrays -> {OUT_PATH}")
print(f"  keys: {', '.join(sorted(results))}")
