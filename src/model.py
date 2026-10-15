"""
falcon simulator nodes for two-stage calibration + power-spectrum inference (SKA-Low AA*).

Every class below is one node of the graph in config.yml (`_target_: model.<Class>`).
falcon calls `simulate_batch(batch_size, *parents)` with numpy arrays and stores every
node's numpy output per sample. Settings come from src/settings.py.

Stage 1 — gain calibration       q(z | V_ref)
    A known calibrator at the beam centre is observed; V_ref = G S_ref G† + N.
Stage 2 — science inference      q(pw | pk_obs, V_ref)
    The 21-cm sky is observed through the same gains; the network sees the P(k) of the
    dirty image (pk_obs) and V_ref (to marginalise over the gains).

Graph (config.yml node names; [observed] = conditioned on data/*.npy)

    z ──► gains ──────────────────────────────────────────────► V_ref          [observed]
                │                                                 (G S_ref G† + N)
                └──► stacked_uv ──┐
                                  ├──► tracked_uv ──► normed_tracked_uv ──┐
    pw ──► df ──► bt ──► image ──► uvimage ──┘                            ├──► x ──► obsx ──► pk_obs  [observed]
           └──► pk_df              └──► noiseimage ──────────────────────┘
                                         (thermal noise, NOISE_STD / sqrt(n) per UV pixel)

    Optional dense science path (disabled in config.yml):
        uvimage ──► S_full;   gains + S_full ──► V_dense = G S_full G†

Units: images in mK (brightness temperature); UV quantities are their ortho-FFTs.
"""

import torch

import functionlistnew as func
import settings

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===========================================================================
# Set-up (all numbers from settings.py)
# ===========================================================================
KNOWN_ANT               = settings.KNOWN_ANT
SCIENCE_FOREGROUND_FLUX = settings.SCIENCE_FOREGROUND_FLUX

padded    = torch.load(settings.PADDED_PATH, map_location=device, weights_only=True)
NANT      = func.nant_from_padded(padded)          # 46971 baselines → 307 antennas
N_UNKNOWN = NANT - 1 - KNOWN_ANT                   # antennas with free gains (z has 2 × this)

config = func.SimulationConfig(
    shape=settings.N_PIX,
    BoxSize=settings.BOX_SIZE,
    sigma=settings.CALIBRATOR_SIGMA,
    prefactor=settings.CALIBRATOR_FLUX,
    z=settings.REDSHIFT,
    nant=NANT,
    padded=padded,
    subsetindex=func.split_indices(padded.shape[0], settings.N_SUBSETS),
    device=device,
)

# Thermal noise in the pipeline's units (mK, ortho-FFT), for the whole observation:
#   settings.NOISE_STD   one 30 s integration, in settings.NOISE_UNIT ("mK" or "Jy" → converted here)
#   / sqrt(N_DAYS)       the 4 h track repeated on N_DAYS days: N_DAYS × the integrations per
#                        baseline / UV pixel (the nodes divide further by sqrt(n) of ONE track)
assert settings.NOISE_UNIT in ("mK", "Jy"), f"NOISE_UNIT={settings.NOISE_UNIT!r}, expected 'mK' or 'Jy'"
assert settings.N_DAYS > 0, f"N_DAYS={settings.N_DAYS} must be > 0"
NOISE_STD_30S = (settings.NOISE_STD if settings.NOISE_UNIT == "mK"
                 else func.jansky_to_mK(settings.NOISE_STD, "uv", config))
NOISE_STD = NOISE_STD_30S / settings.N_DAYS ** 0.5

# Simulators (built once per process)
density_sim = func.DensityField2D(config)
brightness  = func.BrightnessTemperature(config)
uv_maps     = func.UVMapGenerator(config)
sky_matrix  = func.SkyMatrixGenerator(padded, config.shape, NANT)
pk_measurer = func.PowerSpectrumMeasurer(
    shape=(config.shape, config.shape), BoxSize=config.BoxSize, device=device,
)

# UV-path normalisation: unit-gain UV maps = integrations per pixel (1, S, H, W), and the
# thermal-noise scale 1 / sqrt(n) (0 on unsampled pixels). n is symmetric under
# (u, v) → (-u, -v), so the scaled noise stays Hermitian.
UNIT_UV        = func.UnitUVMapGenerator(config)()
UV_COUNT       = UNIT_UV.real
UV_NOISE_SCALE = torch.where(UV_COUNT > 0, UV_COUNT.clamp(min=1e-30).rsqrt(),
                             torch.zeros_like(UV_COUNT))

# Unit-peak Gaussian at the beam centre: the calibrator shape (and optional science foreground)
BEAM_CENTRE_GAUSSIAN = func.gaussian_2d(size=config.shape, sigma=config.sigma, device=device)

# Stage-1 reference sky matrix S_ref (N, N), known and fixed: the calibrator image through
# the dense path. Computed here from settings, so it always matches the current set-up;
# scripts/gen_mock_data.py saves the same matrix to data/S_ref.npy for notebooks.
# (To calibrate on an external sky model instead, load it here.)
S_REF = sky_matrix(func.image_to_uv(settings.CALIBRATOR_FLUX * BEAM_CENTRE_GAUSSIAN[None]))[0]


class _NoiseNode:
    """
    Base of the two thermal-noise nodes. noise_std: std of the complex noise on one visibility in
    mK units; None (falcon builds the nodes without arguments) = the module's NOISE_STD, read at
    call time. Pass 0.0 for a noise-free run without touching the module.
    """
    def __init__(self, noise_std=None):
        self.noise_std = noise_std

    @property
    def std(self):
        return NOISE_STD if self.noise_std is None else self.noise_std


def _tensor(x, dtype=None):
    return torch.tensor(x, device=device, dtype=dtype)


def _numpy(t):
    return t.detach().cpu().numpy()


# ===========================================================================
# Gains
# ===========================================================================

class GainsFromZ:
    """gains: z (B, 2 N_UNKNOWN) perturbations → (B, 2 NANT) interleaved antenna gains."""
    def simulate_batch(self, batch_size, z):
        z = _tensor(z)
        assert z.size(1) == 2 * N_UNKNOWN, \
            f"z has {z.size(1)} params, expected 2*(nant-1-KNOWN_ANT) = {2 * N_UNKNOWN}"
        return _numpy(func.theta_to_gains(z, KNOWN_ANT))


class UVTracksFromGains:
    """stacked_uv: gain-weighted UV maps (B, S, H, W), Σ_track t · g_i conj(g_j) per pixel."""
    def simulate_batch(self, batch_size, gains):
        return _numpy(uv_maps(_tensor(gains)))


# ===========================================================================
# Science sky
# ===========================================================================

class DensityFromPw:
    """df: density contrast δ (B, H, W) from pw = [log10(A), n1, n2, log10(k_break)]."""
    def simulate_batch(self, batch_size, pw):
        return _numpy(density_sim.sample_field(*func.pw_to_params(_tensor(pw))))


class BrightnessFromDensity:
    """bt: 21-cm brightness temperature (B, H, W) [mK]."""
    def simulate_batch(self, batch_size, df):
        return _numpy(brightness(_tensor(df)))


class PkFromDensity:
    """pk_df: azimuthally averaged P(k) of δ, (B, n_bins) — diagnostic, not used by estimators."""
    def simulate_batch(self, batch_size, df):
        _, pk_mean, _ = pk_measurer(_tensor(df))
        return _numpy(pk_mean)


class SkyImageFromBrightness:
    """image: science sky (B, H, W) [mK] = bt + SCIENCE_FOREGROUND_FLUX × beam-centre Gaussian."""
    def simulate_batch(self, batch_size, bt):
        bt = _tensor(bt)
        foreground = BEAM_CENTRE_GAUSSIAN.unsqueeze(0).expand_as(bt)
        return _numpy(bt + SCIENCE_FOREGROUND_FLUX * foreground)


class UVSkyFromImage:
    """uvimage: Fourier sky (B, H, W) complex, FFT order, phase centre = beam centre."""
    def simulate_batch(self, batch_size, image):
        return _numpy(func.image_to_uv(_tensor(image)))


# ===========================================================================
# Science observation (UV-map path)
# ===========================================================================

class TrackedUVFromSky:
    """tracked_uv: gain-weighted sampled sky (B, S, H, W) = stacked_uv × uvimage."""
    def simulate_batch(self, batch_size, uvimage, stacked_uv):
        return _numpy(func.apply_uv_sampling(_tensor(stacked_uv), _tensor(uvimage)))


class NormalisedUV:
    """normed_tracked_uv: tracked_uv / integrations per pixel (mean over integrations); 0 unsampled."""
    def simulate_batch(self, batch_size, tracked_uv):
        normed = _tensor(tracked_uv) / UNIT_UV
        return _numpy(normed.nan_to_num(nan=0.0))


class ThermalNoiseUV(_NoiseNode):
    """
    noiseimage: gain-independent thermal noise on the normalised UV map, (B, S, H, W) complex.

    fft2 (ortho) of real white noise → N(-u, -v) = N*(u, v), E|N|² = 1 per pixel, so iFFT2(N)
    is real; scaled by noise std / sqrt(n) (n = integrations per pixel), independent per
    subset. `uvimage` only sets the batch size. ThermalNoiseUV(noise_std=0.0): noise-free.
    """
    def simulate_batch(self, batch_size, uvimage):
        white = torch.randn((len(uvimage), *UV_NOISE_SCALE.shape[1:]), device=device)
        return _numpy(torch.fft.fft2(white, norm="ortho") * (self.std * UV_NOISE_SCALE))


class VisibilitiesWithNoise:
    """x: observed normalised UV map (B, S, H, W) = normed_tracked_uv + noiseimage."""
    def simulate_batch(self, batch_size, normed_tracked_uv, noiseimage):
        return _numpy(_tensor(normed_tracked_uv) + _tensor(noiseimage))


class DirtyImageFromUV:
    """
    obsx: dirty image (B, S, H, W) real [mK], pixel-aligned with `image`.

    x is Hermitian (UV maps gridded with conjugates + Hermitian noise), so the imaginary part
    is FFT round-off (~1e-7 of the real part) and is dropped: half the memory.
    """
    def simulate_batch(self, batch_size, x):
        return _numpy(func.uv_to_image(_tensor(x)).real)


class PkFromDirtyImage:
    """pk_obs: azimuthally averaged P(k) of each subset's dirty image, (B, S, n_bins)."""
    def simulate_batch(self, batch_size, obsx):
        obsx = _tensor(obsx, dtype=torch.float32)
        B, S, H, W = obsx.shape
        _, pk_mean, _ = pk_measurer(obsx.reshape(B * S, H, W))
        return _numpy(pk_mean.reshape(B, S, -1))


# ===========================================================================
# Calibration observation (dense path)
# ===========================================================================

class ReferenceVisFromGains(_NoiseNode):
    """
    V_ref: reference visibilities (B, N, N) = G S_ref G† + NOISE_STD · N.

    V[i, j] = g_i S_ref[i, j] conj(g_j) (outer product, no (B, N, N) gain matrices needed);
    N is Hermitian with std 1 / sqrt(n_ij), zero on unobserved baselines and the diagonal.
    ReferenceVisFromGains(noise_std=0.0): noise-free.
    """
    def simulate_batch(self, batch_size, gains):
        g = func.gains_to_complex(_tensor(gains, dtype=torch.float32))            # (B, N)
        V_ref = g[:, :, None] * S_REF[None] * g.conj()[:, None, :]
        V_ref = V_ref + self.std * sky_matrix.noise(len(g))
        return _numpy(V_ref)


class SkyMatrixFromUVSky:
    """S_full: science sky matrix (B, N, N), each baseline averaged over its track."""
    def simulate_batch(self, batch_size, uvimage):
        return _numpy(sky_matrix(_tensor(uvimage, dtype=torch.complex64)))


class DenseVisFromGains:
    """V_dense: science visibilities (B, N, N) = G S G† (noise-free)."""
    def simulate_batch(self, batch_size, gains, S):
        gains = _tensor(gains, dtype=torch.float32)
        S     = _tensor(S, dtype=torch.complex64)
        return _numpy(func.visibilities_from_gains(gains, S))
