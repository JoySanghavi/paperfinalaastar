"""
Shared settings — the ONE place to change the simulation set-up.

Every script reads its numbers from here (src/model.py, scripts/gen_mock_data.py,
scripts/plot_pipeline.py, scripts/plot_reference.py, fisher/fisher_forecast.py),
so the simulator used for training, the mock observation and the Fisher forecast cannot
drift apart. Only the priors live elsewhere (config.yml, read by falcon); the scripts check
the true values below against them.

This module has no heavy imports, so notebooks can use it cheaply:
    import sys; sys.path.insert(0, "../src"); import settings     # from notebooks/
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = Path(__file__).resolve().parent
ROOT_DIR    = SRC_DIR.parent
DATA_DIR    = ROOT_DIR / "data"          # observation + truth (*.npy), from scripts/gen_mock_data.py
FIGURES_DIR = ROOT_DIR / "figures"       # figures of scripts/ and notebooks/
CONFIG_PATH = ROOT_DIR / "config.yml"

# ---------------------------------------------------------------------------
# Instrument  —  SKA-Low AA*, 4 h track at 30 s integrations, z = 9, 128 px, 512 Mpc
# ---------------------------------------------------------------------------
# Baseline tracks on the UV grid, (n_baselines, 3, L): [x (col), y (row), t] per track
# sample, in FFT order (DC at [0, 0]); t = number of INT_TIME integrations in that pixel.
# The file name encodes the set-up: AA*, 4 h, 30 s, z = 9, 128 px, 512 Mpc.
PADDED_PATH  = SRC_DIR / "paddedAAstar_4_30_9_128_512.pth"
TOTAL_TIME_H = 4.0      # length of ONE track — fixed by PADDED_PATH (see N_DAYS for repeats)
INT_TIME_S   = 30.0     # documentation only — fixed by PADDED_PATH

N_PIX     = 128         # image / UV grid side [px]
BOX_SIZE  = 512.0       # image side [Mpc]  →  pixel size BOX_SIZE / N_PIX = 4 Mpc
REDSHIFT  = 9.0
N_SUBSETS = 2           # baselines are split into this many UV subsets (one UV map each)

# Cosmology — Planck 2018 (Paper VI, TT,TE,EE+lowE+lensing; flat ΛCDM, Ω_Λ = 1 − Ω_m), used for
# the 21-cm brightness temperature, the comoving distance (mK <-> Jy) and notebooks/noise.ipynb.
HUBBLE_H = 0.674        # H0 / (100 km/s/Mpc)
OMEGA_M  = 0.315        # matter density
OMEGA_B  = 0.0492       # baryon density (Ω_b h² = 0.02237)
HELIUM_Y = 0.2454       # primordial helium mass fraction Y_p
# The tracks (PADDED_PATH) were made with tools21cm 2.4.3 with its constants set to these same values
# (not shipped here; see README "Cosmology and the tracks file"): they are gridded with
# D_c(9) = 9400.70 Mpc, as functionlistnew.comoving_distance gives.

# Antenna layout: [reference g_0 = 1] + [unknown antennas, from z] + [KNOWN_ANT with g = 1]
#   → z has 2 * (n_ant - 1 - KNOWN_ANT) parameters (212 for AA*: 307 antennas, 106 unknown).
# Changing KNOWN_ANT: the z prior list in config.yml must have 2 * (n_ant - 1 - KNOWN_ANT)
# entries (checked by scripts/gen_mock_data.py and the tests).
KNOWN_ANT = 200

# ---------------------------------------------------------------------------
# Thermal noise
# ---------------------------------------------------------------------------
# Std of the complex noise on ONE visibility (one baseline, one INT_TIME integration),
# E|n|^2 = NOISE_STD^2. Averaging n integrations keeps the signal and gives NOISE_STD / sqrt(n)
# (V_ref: n per baseline; UV maps: n per pixel).
# NOISE_UNIT "mK": already in the pipeline's visibility units (the ortho-FFT of the sky in mK).
#            "Jy": a physical visibility noise (e.g. from the radiometer equation); src/model.py
#                  converts it once with functionlistnew.jansky_to_mK(..., "uv") and the
#                  simulator adds it in mK, like everything else.
NOISE_STD  = 0.0
NOISE_UNIT = "mK"
# Observing time: the 4 h track (PADDED_PATH) is observed on N_DAYS days with the same UV coverage
# (it repeats every sidereal day), so every baseline / UV pixel has N_DAYS times the integrations
# and src/model.py divides the noise by sqrt(N_DAYS). The signal and the gains are the same every day.
#   e.g. N_DAYS = 250  →  T_OBS_H = 1000 h  →  noise / sqrt(250)
N_DAYS  = 1
T_OBS_H = N_DAYS * TOTAL_TIME_H      # total observing time [h] (derived)

# ---------------------------------------------------------------------------
# Skies
# ---------------------------------------------------------------------------
# Reference calibrator (Stage 1): a Gaussian of CALIBRATOR_SIGMA px and peak CALIBRATOR_FLUX
# at the beam / phase centre, pixel (N_PIX // 2, N_PIX // 2). Narrow and on the phase centre
# → the same S_ref ≈ CALIBRATOR_FLUX / N_PIX on every observed baseline.
CALIBRATOR_FLUX  = 2.7   # [mK]
CALIBRATOR_SIGMA = 0.3   # [px]

# Science sky (Stage 2): 21-cm brightness temperature + this times the calibrator-shaped
# Gaussian (a bright source at the beam centre). 0 = no foreground in the science field.
SCIENCE_FOREGROUND_FLUX = 0.0   # [mK]

# ---------------------------------------------------------------------------
# Truth for the mock observation (scripts/gen_mock_data.py and the Fisher forecast)
# ---------------------------------------------------------------------------
TRUE_Z_SEED  = 99       # true gain perturbations: N(0, TRUE_Z_SCALE^2) per Re / Im
TRUE_Z_SCALE = 0.3
# Broken power law [log10(A), n1, n2, log10(k_break)]:
#   P(k) = A (k / k_break)^n1  (k <= k_break),  A (k / k_break)^n2  (k > k_break)
# k_break ≈ 0.16 Mpc^-1 = 0.20 k_Nyquist; k-bins with data span k ≈ 0.013-1.06 Mpc^-1,
# so there are bins on both sides of the break. This truth gives a pixel rms σ_δ = 0.31,
# σ_T = 8.2 mK (δ < -1 in 0.06 % of pixels): the upper end of a z ≈ 9 21-cm slice at 4 Mpc.
TRUE_PW         = (1.0, -0.5, -2.0, -0.8)
TRUE_FIELD_SEED = 54    # seed of the true density-field realisation
PW_NAMES        = ("log10_A", "n1", "n2", "log10_k_break")


def true_z(n_unknown_ant: int):
    """
    True gain perturbations (1, 2 * n_unknown_ant), float32 tensor on the CPU.

    Drawn from a private generator, so the global torch RNG is left untouched; identical to
    `torch.manual_seed(TRUE_Z_SEED); torch.randn(1, 2 * n_unknown_ant) * TRUE_Z_SCALE`.
    The draw depends on its length: changing KNOWN_ANT gives different true gains for every
    antenna, not a truncation of the previous ones.
    """
    import torch
    gen = torch.Generator().manual_seed(TRUE_Z_SEED)
    return torch.randn(1, 2 * n_unknown_ant, generator=gen) * TRUE_Z_SCALE


def check_truth_inside_priors(z_values) -> None:
    """Assert that TRUE_PW and the true z lie inside the uniform priors of config.yml."""
    import yaml
    graph = yaml.safe_load(CONFIG_PATH.read_text())["graph"]
    for node, values in (("pw", list(TRUE_PW)), ("z", list(z_values))):
        priors = graph[node]["simulator"]["priors"]
        assert len(priors) == len(values), \
            f"{node}: config.yml has {len(priors)} priors, the true value has {len(values)}"
        for k, ((_, lo, hi), v) in enumerate(zip(priors, values)):
            assert lo <= v <= hi, f"true {node}[{k}] = {v:.4f} outside config.yml prior [{lo}, {hi}]"
