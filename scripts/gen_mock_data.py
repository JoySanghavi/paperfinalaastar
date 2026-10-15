"""
Generate the mock observation (and truth) that falcon conditions on.

The observation is produced by the SAME simulator nodes falcon trains on (src/model.py),
with the truth from src/settings.py, so the mock data and the training simulations cannot
drift apart. Re-run this whenever src/settings.py changes.

Writes to data/:
  Observed by falcon (config.yml `observed:`)
    pk_obs.npy            (2, n_bins)      P(k) of the dirty image per UV subset    [Stage 2]
    V_ref_obs.npy         (N, N) complex   reference visibilities G S_ref G† + N    [Stage 1]
  Intermediate products (analysis / notebooks)
    x_obs.npy             (2, H, W) complex  normalised UV maps incl. noise
    obsx_obs.npy          (2, H, W) real     dirty images [mK]
    image.npy             (H, W)             true science sky [mK]
    S_ref.npy             (N, N) complex     reference sky matrix (identical to model.S_REF)
    S_obs.npy             (N, N) complex     science sky matrix
    V_dense_obs.npy       (N, N) complex     science visibilities G S G† (noise-free)
  Truth
    true_z.npy            (1, 2 N_unknown)   gain perturbations
    true_pw.npy           (1, 4)             [log10(A), n1, n2, log10(k_break)]
    true_delta.npy        (1, H, W)          density contrast
    true_k.npy            (n_bins,)          k-bin centres [Mpc^-1]
    true_pk_measured.npy  (n_bins,)          measured P(k) of true_delta
  Provenance
    mock_device.npy       ()  str            device the truth was drawn on: "cpu" or "cuda"

Runs on the GPU if there is one, else on the CPU (model.device). torch draws different random
numbers on CPU and GPU for the same seed, so the true field (TRUE_FIELD_SEED) depends on the
device: it is recorded in mock_device.npy, and the checks that regenerate the truth
(scripts/plot_pipeline.py, the notebooks) compare against data/ on that device. A falcon run
conditions on the data/ it was trained with — do not regenerate data/ for an existing run.

Run from the project root:
    python scripts/gen_mock_data.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent                # project root
sys.path.insert(0, str(ROOT / "src"))
import functionlistnew as func                               # noqa: E402
import model                                                 # noqa: E402
import settings                                              # noqa: E402


def to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


# ---------------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------------
true_z  = settings.true_z(model.N_UNKNOWN)                       # (1, 2 N_UNKNOWN) CPU tensor
true_pw = np.array([settings.TRUE_PW])                           # (1, 4)
settings.check_truth_inside_priors(true_z[0].tolist())

print(f"Padded tensor : {tuple(model.padded.shape)}  (nant = {model.NANT}, "
      f"unknown = {model.N_UNKNOWN}, known = {settings.KNOWN_ANT})")
print(f"Grid          : {settings.N_PIX} px, {settings.BOX_SIZE} Mpc;  NOISE_STD = {settings.NOISE_STD};  "
      f"device = {model.device}")

# ---------------------------------------------------------------------------
# Forward model — the falcon graph of config.yml, node by node (batch of 1)
# ---------------------------------------------------------------------------
gains = model.GainsFromZ().simulate_batch(1, to_numpy(true_z))

# df: the DensityFromPw node with a fixed seed for the true realisation
pw    = torch.tensor(true_pw, dtype=torch.float32, device=model.device)        # (1, 4)
delta = to_numpy(model.density_sim.sample_field(*func.pw_to_params(pw),
                                                seed=settings.TRUE_FIELD_SEED))   # (1, H, W)

bt         = model.BrightnessFromDensity().simulate_batch(1, delta)
image      = model.SkyImageFromBrightness().simulate_batch(1, bt)
uvimage    = model.UVSkyFromImage().simulate_batch(1, image)
stacked_uv = model.UVTracksFromGains().simulate_batch(1, gains)
tracked_uv = model.TrackedUVFromSky().simulate_batch(1, uvimage, stacked_uv)
normed_uv  = model.NormalisedUV().simulate_batch(1, tracked_uv)
noiseimage = model.ThermalNoiseUV().simulate_batch(1, uvimage)
x_obs      = model.VisibilitiesWithNoise().simulate_batch(1, normed_uv, noiseimage)
obsx_obs   = model.DirtyImageFromUV().simulate_batch(1, x_obs)
pk_obs     = model.PkFromDirtyImage().simulate_batch(1, obsx_obs)
V_ref      = model.ReferenceVisFromGains().simulate_batch(1, gains)

# Dense science path (auxiliary, noise-free)
S_obs   = model.SkyMatrixFromUVSky().simulate_batch(1, uvimage)
V_dense = model.DenseVisFromGains().simulate_batch(1, gains, S_obs)

# Measured P(k) of the true density field
k_centers, pk_true, _ = model.pk_measurer(torch.tensor(delta, device=model.device))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
outputs = {
    # observed by falcon
    "pk_obs":           pk_obs[0],
    "V_ref_obs":        V_ref[0],
    # intermediate products
    "x_obs":            x_obs[0],
    "obsx_obs":         obsx_obs[0],
    "image":            image[0],
    "S_ref":            to_numpy(model.S_REF),
    "S_obs":            S_obs[0],
    "V_dense_obs":      V_dense[0],
    # truth
    "true_z":           to_numpy(true_z),
    "true_pw":          true_pw,
    "true_delta":       delta,
    "true_k":           to_numpy(k_centers),
    "true_pk_measured": to_numpy(pk_true[0]),
    # provenance: seeded draws differ between CPU and GPU
    "mock_device":      np.array(torch.device(model.device).type),
}
for name, arr in outputs.items():
    np.save(settings.DATA_DIR / f"{name}.npy", arr)
    print(f"Saved {name + '.npy':22s} shape {tuple(arr.shape)}  {arr.dtype}")

print(f"\nTrue pw : {dict(zip(settings.PW_NAMES, settings.TRUE_PW))}")
print("Done. Mock data saved to data/")
