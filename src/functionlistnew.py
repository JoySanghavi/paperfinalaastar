"""
Simulation building blocks: sky model, instrument (UV-map and dense paths), P(k) estimator.

Pure library — nothing here depends on falcon. src/model.py wires these into falcon nodes;
scripts/gen_mock_data.py, the plotting scripts and the Fisher forecast reuse them directly.

Conventions (used everywhere in this project)
---------------------------------------------
* Arrays are indexed array[..., y, x]: y = row (down), x = column (right).
* Images have the beam / phase centre at pixel (N // 2, N // 2), as displayed.
* UV arrays are in FFT order: DC at [0, 0], negative u / v wrapped to the end — the same
  order as the padded baseline tracks and torch.fft.fftfreq.
* image_to_uv / uv_to_image are the ONLY image <-> UV transforms for skies and UV maps
  (orthonormal FFTs, so values keep their mK units and P(k) needs no extra factors).
* Gains are real tensors (B, 2 * n_ant), interleaved [Re g_0, Im g_0, Re g_1, ...];
  baseline (i, j), i < j, carries g_i conj(g_j), in torch.triu_indices order.

Sections
--------
  1. Configuration          SimulationConfig, nant_from_padded, split_indices
  2. Gains                  theta_to_gains, baseline_gains, visibilities_from_gains
  3. Image <-> UV           image_to_uv, uv_to_image
  4. Sky model              gaussian_2d, power_spectrum_2d, DensityField2D,
                            mean_brightness_temperature, BrightnessTemperature
  5. Instrument: UV maps    UVMapGenerator, UnitUVMapGenerator, apply_uv_sampling
  6. Instrument: dense      SkyMatrixGenerator  (S[i, j] per baseline, V = G S G†)
  7. Measurement            PowerSpectrumMeasurer, drop_roundoff_bins
  8. Units                  comoving_distance, mK_per_jansky, mK_to_jansky, jansky_to_mK
"""

import warnings

import numpy as np
import torch

import settings

# Kept for reproducibility of existing notebooks and runs, which import this module and
# rely on the global RNG state it sets.
torch.manual_seed(42)


# ===========================================================================
# 1. Configuration
# ===========================================================================

class SimulationConfig:
    """
    Grid, cosmology and instrument settings shared by the simulators.

    Defaults come from src/settings.py, so SimulationConfig() describes the project set-up.

    Grid:        shape [px], BoxSize [Mpc]  →  dx = dy = BoxSize / shape
    Calibrator:  sigma [px] width and prefactor [mK] peak of the Gaussian source used as
                 the Stage-1 reference sky (gaussian_2d at the image centre).
    Cosmology:   inputs of BrightnessTemperature and comoving_distance (Planck 2018, settings.py).
    Instrument:  nant (default: from padded; None without tracks), padded (baseline tracks),
                 subsetindex (baseline indices per UV subset),
                 conjugate (also grid baseline (j, i) at (-u, -v) → Hermitian UV maps),
                 drop_dc (drop samples on the DC pixel → zero-mean images).
    """
    def __init__(
        self,
        shape=settings.N_PIX,
        BoxSize=settings.BOX_SIZE,
        sigma=settings.CALIBRATOR_SIGMA,
        prefactor=settings.CALIBRATOR_FLUX,
        ionizationfraction=1.0,
        z=settings.REDSHIFT,
        TCMB=0.0,
        TS=1.0,
        omegab=settings.OMEGA_B,
        h=settings.HUBBLE_H,
        omegam=settings.OMEGA_M,
        yp=settings.HELIUM_Y,
        nant=None,
        padded=None,
        subsetindex=None,
        device=None,
        conjugate=True,
        drop_dc=True,
    ):
        # Grid
        self.shape = shape
        self.BoxSize = BoxSize
        self.dx = BoxSize / shape
        self.dy = BoxSize / shape

        # Calibrator source (Gaussian at the image centre)
        self.sigma = sigma
        self.prefactor = prefactor

        # Cosmology
        self.ionizationfraction = ionizationfraction
        self.z = z
        self.TCMB = TCMB
        self.TS = TS
        self.omegab = omegab
        self.h = h
        self.omegam = omegam
        self.yp = yp

        # Antennas / UV tracks: nant follows from the tracks (K = N (N - 1) / 2 baselines)
        if padded is not None:
            n_from_tracks = nant_from_padded(padded)
            assert nant in (None, n_from_tracks), \
                f"padded has {padded.shape[0]} baselines ({n_from_tracks} antennas) but nant={nant}"
            nant = n_from_tracks
        self.nant = nant
        self.padded = padded
        self.subsetindex = subsetindex
        self.conjugate = conjugate
        self.drop_dc = drop_dc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")


def nant_from_padded(padded: torch.Tensor) -> int:
    """Number of antennas N such that padded.shape[0] == N (N - 1) / 2."""
    nbl = padded.shape[0]
    nant = int(round((1 + np.sqrt(1 + 8 * nbl)) / 2))
    assert nant * (nant - 1) // 2 == nbl, f"{nbl} is not N choose 2 for any N"
    return nant


def split_indices(n: int, k: int, device=None) -> list[torch.Tensor]:
    """Split [0, 1, ..., n-1] into k contiguous, nearly equal parts (the UV subsets)."""
    if k <= 0:
        raise ValueError("Number of splits k must be > 0")
    if k > n:
        warnings.warn(f"Splitting {n} elements into {k} parts, some will be empty.")
    indices = torch.arange(n, device=device)
    base_size, remainder = divmod(n, k)
    sizes = [base_size + 1 if i < remainder else base_size for i in range(k)]
    return list(torch.split(indices, sizes))


# ===========================================================================
# 2. Gains
# ===========================================================================

def theta_to_gains(theta: torch.Tensor, nknown: int = 0) -> torch.Tensor:
    """
    Gain perturbations θ → antenna gains, (B, 2 (n - 1 - nknown)) → (B, 2 n).

    Antenna order: [reference g_0 = 1] + [unknown antennas, g = 1 + θ] + [nknown known
    antennas, g = 1]. θ holds interleaved (Re, Im) perturbations of the unknown antennas.
    """
    assert theta.ndim == 2 and theta.size(1) % 2 == 0, \
        f"Expected shape (batch, 2*(n-1-nknown)), got {tuple(theta.shape)}"
    if nknown:
        theta = torch.cat([theta, theta.new_zeros(len(theta), 2 * nknown)], dim=1)
    offset = theta.new_zeros(theta.shape)
    offset[:, 0::2] = 1.0
    reference = theta.new_tensor([[1.0, 0.0]]).expand(len(theta), 2)
    return torch.cat((reference, theta + offset), dim=1)


def gains_to_complex(gains: torch.Tensor) -> torch.Tensor:
    """(B, 2 N) interleaved real gains → (B, N) complex gains."""
    assert gains.ndim == 2 and gains.size(1) % 2 == 0, \
        f"Expected shape (B, 2*N), got {tuple(gains.shape)}"
    return torch.view_as_complex(gains.reshape(len(gains), -1, 2).contiguous())


def baseline_gains(gainlists: torch.Tensor) -> torch.Tensor:
    """
    Per-baseline gain products g_i conj(g_j) for all pairs i < j.

    gainlists: (B, 2 N) interleaved real gains → (B, N (N - 1) / 2) complex, in
    torch.triu_indices order (the order of the padded baseline tracks).
    """
    g = gains_to_complex(gainlists)
    idx_i, idx_j = torch.triu_indices(g.shape[1], g.shape[1], offset=1, device=gainlists.device)
    return g[:, idx_i] * torch.conj(g[:, idx_j])


def gains_to_diagonal_matrix(gains: torch.Tensor) -> torch.Tensor:
    """(B, 2 N) interleaved real gains → (B, N, N) complex diagonal gain matrices G."""
    assert gains.shape[-1] % 2 == 0, \
        f"gains.shape[-1] must be even (interleaved re/im), got {gains.shape[-1]}"
    return torch.diag_embed(gains_to_complex(gains))


def visibilities_from_gains(gains: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """
    Visibility matrix V = G S G†  (V[i, j] = g_i S[i, j] conj(g_j)).

    gains: (B, 2 N) interleaved real gains;  S: (B, N, N) sky matrix  →  V: (B, N, N).
    """
    assert S.dim() == 3 and S.shape[1] == S.shape[2], \
        f"S must be (B, N, N), got {tuple(S.shape)}"
    assert gains.shape[0] == S.shape[0] and gains.shape[-1] // 2 == S.shape[1], \
        f"nant mismatch: gains implies nant={gains.shape[-1]//2}, S has N={S.shape[1]}"
    G = gains_to_diagonal_matrix(gains)
    return torch.bmm(torch.bmm(G, S), G.conj().transpose(-2, -1))


# ===========================================================================
# 3. Image <-> UV transforms
# ===========================================================================
# ifftshift moves the image centre to [0, 0] before fft2, so a source at the beam centre has
# zero phase on every baseline; fftshift undoes it after ifft2, so
# uv_to_image(image_to_uv(img)) == img pixel for pixel. White noise and P(k) estimates do
# not need the shifts (both are invariant to circular shifts).

def image_to_uv(image: torch.Tensor) -> torch.Tensor:
    """(..., H, W) image, centre at (H//2, W//2) → (..., H, W) UV map, FFT order, ortho."""
    return torch.fft.fft2(torch.fft.ifftshift(image, dim=(-2, -1)), norm="ortho")


def uv_to_image(uv: torch.Tensor) -> torch.Tensor:
    """(..., H, W) UV map, FFT order → (..., H, W) image, centre at (H//2, W//2), ortho."""
    return torch.fft.fftshift(torch.fft.ifft2(uv, norm="ortho"), dim=(-2, -1))


# ===========================================================================
# 4. Sky model
# ===========================================================================

def gaussian_2d(size=128, sigma=0.3, device="cpu", center=None):
    """
    (size, size) Gaussian with unit peak, array[y, x].

    center: same for x and y. The default size // 2 is the image (beam / phase) centre and
    lies ON a pixel, so a sub-pixel sigma stays a single-pixel source with ~flat |FFT| and
    zero phase. Pass (size - 1) / 2 for a between-pixel centre (split over 4 pixels,
    |FFT| → 0 at Nyquist).
    """
    coords = torch.arange(size, dtype=torch.float32, device=device)
    x, y = torch.meshgrid(coords, coords, indexing="xy")      # x along columns, y along rows
    cx = cy = size // 2 if center is None else center
    return torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))


def make_kgrid(shape, BoxSize, device, rfft=False):
    """
    |k| = sqrt(kx² + ky²) [Mpc⁻¹] on the FFT grid, array[y, x].

    shape = (H, W). kx (columns) uses rfftfreq if rfft else fftfreq → (H, W//2+1) or (H, W);
    ky (rows) always uses fftfreq. FFT order, matching torch.fft.(r)fft2 output.
    """
    H, W = shape
    dy, dx = BoxSize / H, BoxSize / W
    freq_fn = torch.fft.rfftfreq if rfft else torch.fft.fftfreq
    kx = freq_fn(W, d=dx, device=device) * 2 * np.pi
    ky = torch.fft.fftfreq(H, d=dy, device=device) * 2 * np.pi
    kx, ky = torch.meshgrid(kx, ky, indexing="xy")             # kx varies along axis 1
    return torch.sqrt(kx ** 2 + ky ** 2)


def power_spectrum_2d(k, A, n1, n2, k_break):
    """
    Broken power law, continuous at k_break; P(0) = 0 (zero-mean field):

        P(k) = A (k / k_break)^n1   for k <= k_break
        P(k) = A (k / k_break)^n2   for k >  k_break

    A is the amplitude AT the break [Mpc²].
    k: (...) grid;  A, n1, n2, k_break: (B,)  →  (B, ...) broadcast over leading k dims.
    """
    mask_zero = k < 1e-12
    k_safe    = torch.where(mask_zero, torch.ones_like(k), k)
    kb        = k_break[:, None, None]
    ratio     = k_safe / kb
    low       = A[:, None, None] * ratio ** n1[:, None, None]
    high      = A[:, None, None] * ratio ** n2[:, None, None]
    pk        = torch.where(k_safe <= kb, low, high)
    return torch.where(mask_zero, torch.zeros_like(pk), pk)


def pw_to_params(pw: torch.Tensor):
    """
    pw (B, 4) = [log10(A), n1, n2, log10(k_break)] → (A, n1, n2, k_break), each (B,), the
    arguments of power_spectrum_2d / DensityField2D.sample_field. The one place this
    conversion is done (model.DensityFromPw, gen_mock_data.py, the Fisher forecast).
    """
    return torch.pow(10, pw[:, 0]), pw[:, 1], pw[:, 2], torch.pow(10, pw[:, 3])


def generate_hermitian_noise(B, H, W, device, seed=None):
    """
    Unit Hermitian Gaussian noise on the rfft grid, (B, H, W//2+1) complex.

    rfft2 (ortho) of real unit white noise: exactly Hermitian, E|N|² = 1 on every mode
    (Re, Im each N(0, 1/2); self-conjugate modes real N(0, 1)). `seed` reseeds the GLOBAL
    torch RNG (used for reproducible / common-random-number draws).
    """
    if seed is not None:
        torch.manual_seed(seed)
    white = torch.randn((B, H, W), device=device)
    return torch.fft.rfft2(white, norm="ortho")


class DensityField2D:
    """Gaussian random density fields δ with a broken power-law P(k)."""
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.k = make_kgrid((config.shape, config.shape), config.BoxSize, config.device, rfft=True)

    def sample_field(self, A, n1, n2, k_break, seed: int | None = None) -> torch.Tensor:
        """
        A, n1, n2, k_break: (B,) — see power_spectrum_2d  →  δ: (B, shape, shape) real.

        Hermitian unit noise coloured by sqrt(P / (dx dy)), then irfft2 (ortho):
        <|δ̃_ortho|²> = P / (dx dy), the inverse of PowerSpectrumMeasurer's P = |δ̃_ortho|² dx dy,
        so the measured P(k) equals the analytic one at any BoxSize.
        """
        cfg = self.config
        pk = power_spectrum_2d(self.k, A, n1, n2, k_break) / (cfg.dx * cfg.dy)
        noise = generate_hermitian_noise(len(A), cfg.shape, cfg.shape, cfg.device, seed=seed)
        return torch.fft.irfft2(noise * torch.sqrt(pk), s=(cfg.shape, cfg.shape), norm="ortho")


def mean_brightness_temperature(z: float, h: float, omegab: float, omegam: float, yp: float) -> float:
    """
    T0 [mK]: 21-cm brightness temperature of fully neutral gas at mean density, T_S >> T_CMB,
    optically thin, no peculiar velocities (Furlanetto, Oh & Briggs 2006):

        T0 = 3 h_P c³ A10 n_H(z) / (32 π k_B ν21² (1 + z) H(z)),
        n_H(z) = (1 − Y_p) Ω_b ρ_crit,0 (1 + z)³ / m_p,   H(z): flat ΛCDM, radiation neglected
                                                          (as comoving_distance).

    Planck 2018 at z = 9: 26.60 mK. Furlanetto's approximation 27 (Ω_b h² / 0.023)
    (0.15 / Ω_m h² (1 + z) / 10)^½ mK is this formula for their reference cosmology (26.96 mK).
    A10_HI = 2.85e-15 s⁻¹ is the value used there and in 21cmFAST; the magnetic-dipole value
    ≈ 2.876e-15 s⁻¹ would raise T0 by 0.9 %.
    """
    H0 = 100e3 * h / MPC                                                     # [1 / s]
    rho_crit = 3.0 * H0 ** 2 / (8.0 * np.pi * G_NEWTON)                      # [kg / m³]
    n_H = (1.0 - yp) * omegab * rho_crit / M_PROTON * (1.0 + z) ** 3         # [1 / m³]
    H_z = H0 * np.sqrt(omegam * (1.0 + z) ** 3 + 1.0 - omegam)               # [1 / s]
    T0_K = (3.0 * H_PLANCK * C_LIGHT ** 3 * A10_HI * n_H
            / (32.0 * np.pi * K_BOLTZMANN * NU_21CM ** 2 * (1.0 + z) * H_z))
    return T0_K * 1e3


class BrightnessTemperature:
    """
    21-cm brightness temperature δT_b = T0 x_HI (1 + δ) (1 − T_CMB / T_S) [mK] from the density
    contrast δ, T0 = mean_brightness_temperature of the config's cosmology and redshift.
    Defaults: x_HI = 1 (`ionizationfraction` is the NEUTRAL fraction), T_S >> T_CMB (TCMB = 0,
    TS = 1); Planck 2018 at z = 9 gives δT_b = 26.60 (1 + δ) mK.
    """
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.T0 = mean_brightness_temperature(config.z, config.h, config.omegab, config.omegam, config.yp)

    def __call__(self, density_field: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        return self.T0 * cfg.ionizationfraction * (1 - cfg.TCMB / cfg.TS) * (1 + density_field)


# ===========================================================================
# 5. Instrument: UV-map path
# ===========================================================================

class SubsetUVMapGenerator:
    """
    Grid one subset of baselines onto the UV plane, weighted by gains and time.

    padded[:, 0 / 1 / 2, :] = x (col) / y (row) / t (integrations in that pixel), FFT order.
    UV map[y, x] = Σ_baselines Σ_track t · gain, with gain = g_i conj(g_j) (mode "norm") or
    |g_i g_j|² (mode "square"). With unit gains this is the number of integrations per pixel.

    conjugate=True: each baseline (i, j) at (y, x) is also gridded as (j, i) at (-y, -x) with
    conj(gain), so the map is Hermitian and its iFFT (for a real sky) is real.
    drop_dc=True: samples on the DC pixel get zero weight → the image carries no mean.
    """
    def __init__(self, paddedtensor: torch.Tensor, subsetindices: torch.Tensor, shape: int,
                 conjugate: bool = True, drop_dc: bool = True):
        self.device = paddedtensor.device
        self.shape = shape
        self.subsetindices = subsetindices
        self.conjugate = conjugate
        selected = paddedtensor[subsetindices]
        x = selected[:, 0, :].long()
        y = selected[:, 1, :].long()
        self.t = selected[:, 2, :]
        if drop_dc:
            self.t = self.t * ((x != 0) | (y != 0))
        self.flat_idx_template = (y * shape + x).reshape(-1)                         # array[y, x]
        # (-u, -v) in FFT order; padded (t = 0) entries at (0, 0) map to (0, 0)
        self.conj_flat_idx_template = (((-y) % shape) * shape + (-x) % shape).reshape(-1)

    def __call__(self, baselinegain: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """baselinegain: (B, n_baselines) complex → (B, shape, shape) complex UV map."""
        baselinegain = baselinegain.to(self.device)
        batch_size = len(baselinegain)
        bg = baselinegain[:, self.subsetindices]
        if mode == "norm":
            gains = bg
        elif mode == "square":
            gains = bg * bg.conj()
        else:
            raise ValueError(f"Unknown mode '{mode}', expected 'norm' or 'square'")
        t = self.t.to(gains.dtype)
        weighted_t = (gains.unsqueeze(-1) * t).reshape(batch_size, -1)
        flat_idx = self.flat_idx_template
        if self.conjugate:
            weighted_t_conj = (gains.conj().unsqueeze(-1) * t).reshape(batch_size, -1)
            weighted_t = torch.cat([weighted_t, weighted_t_conj], dim=1)
            flat_idx = torch.cat([flat_idx, self.conj_flat_idx_template])
        flat_idx = flat_idx.unsqueeze(0).expand(batch_size, -1)
        output = torch.zeros(batch_size, self.shape * self.shape, device=self.device,
                             dtype=baselinegain.dtype)
        output.scatter_add_(1, flat_idx, weighted_t)
        return output.view(batch_size, self.shape, self.shape)


class UVMapGenerator:
    """Gain-weighted UV maps of every subset: gains (B, 2 N) → (B, n_subsets, shape, shape)."""
    def __init__(self, config: SimulationConfig):
        self.nbl = config.padded.shape[0]
        self.uvgens = [SubsetUVMapGenerator(config.padded, subset, config.shape,
                                            conjugate=config.conjugate, drop_dc=config.drop_dc)
                       for subset in config.subsetindex]

    def __call__(self, gainlist: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        bg = baseline_gains(gainlist)
        assert bg.shape[1] == self.nbl, \
            f"gains give {bg.shape[1]} baselines, padded has {self.nbl}"
        return torch.stack([uvgen(bg, mode) for uvgen in self.uvgens], dim=1)


class UnitUVMapGenerator:
    """
    UV maps for unit gains, computed once: (1, n_subsets, shape, shape) complex with zero
    imaginary part = number of integrations per UV pixel (the normalisation of the UV path).
    """
    def __init__(self, config: SimulationConfig):
        unittheta = torch.zeros((1, 2 * (config.nant - 1)), device=config.device)
        self.stacked = UVMapGenerator(config)(theta_to_gains(unittheta))

    def __call__(self) -> torch.Tensor:
        return self.stacked


def apply_uv_sampling(stacked: torch.Tensor, fourimag: torch.Tensor) -> torch.Tensor:
    """
    Gain-weighted sampling of the Fourier sky: (B, n_subsets, H, W) UV maps × (B, H, W) sky
    → (B, n_subsets, H, W).
    """
    assert fourimag.dim() == 3, f"Expected fourimag shape (B, H, W), got {tuple(fourimag.shape)}"
    assert stacked.dim() == 4, \
        f"Expected stacked shape (B, num_subsets, H, W), got {tuple(stacked.shape)}"
    return stacked * fourimag.unsqueeze(1)


# ===========================================================================
# 6. Instrument: dense (per-baseline) path
# ===========================================================================

class SkyMatrixGenerator:
    """
    Dense sky matrix S (B, N, N): one value per baseline, averaged over its whole track.

        S[i, j] = Σ_track t · uv_sky[+u_ij] / n_ij,   n_ij = Σ_track t
        S[j, i] = Σ_track t · uv_sky[−u_ij] / n_ij    (baseline j–i: the mirrored track, u_ji = −u_ij)

    i.e. the mean over the baseline's integrations: the signal does not depend on how long
    the baseline was observed, while noise() falls as 1 / sqrt(n_ij). Each ordered pair is
    tracked on its own track: (i, j) samples the sky at +u, (j, i) at −u, exactly as the UV-map
    path grids baseline (j, i) at (−u, −v). For a real sky uv_sky[−u] = conj(uv_sky[u]), so S
    comes out Hermitian (to FFT round-off) without being forced to. Gains are constant over the
    observation, so G S G† equals averaging the gain-corrupted visibilities. Baselines that
    never land on the grid (n_ij = 0) give S = 0; diagonal 0 (no autocorrelations).

    paddedtensor: (K, 3, L) baseline tracks as in SubsetUVMapGenerator (FFT order).
    drop_dc: zero weight on the DC pixel, as in the UV-map path.
    """
    def __init__(self, paddedtensor: torch.Tensor, shape: int, nant: int, drop_dc: bool = True):
        self.device = paddedtensor.device
        self.nant = nant
        self.shape = shape

        i_idx, j_idx = torch.triu_indices(nant, nant, offset=1)
        self.i_idx = i_idx.to(self.device)
        self.j_idx = j_idx.to(self.device)

        self.x = paddedtensor[:, 0, :]                            # (K, L) column
        self.y = paddedtensor[:, 1, :]                            # (K, L) row
        self.t = paddedtensor[:, 2, :]                            # (K, L) integrations
        self.flat_idx = (self.y * self.shape + self.x).long()    # samples array[y, x]: (i, j) at +u
        # baseline (j, i): the same track mirrored to (−u, −v), FFT order (DC stays at DC)
        self.flat_idx_conj = ((-self.y) % self.shape * self.shape + (-self.x) % self.shape).long()
        if drop_dc:
            self.t = self.t * (self.flat_idx != 0)

        # Integrations per baseline n_ij (K,), 1 / n_ij and the noise scale 1 / sqrt(n_ij)
        self.counts = self.t.sum(dim=-1)
        self.inv_counts = torch.where(self.counts > 0, 1.0 / self.counts.clamp(min=1e-30),
                                      torch.zeros_like(self.counts))
        self.noise_scale = self.inv_counts.sqrt()

    def _to_matrix(self, vals_ij: torch.Tensor, vals_ji: torch.Tensor) -> torch.Tensor:
        """(B, K) values of baselines (i, j) and (j, i), i < j → (B, N, N) with zero diagonal."""
        M = torch.zeros(len(vals_ij), self.nant, self.nant, device=self.device, dtype=vals_ij.dtype)
        M[:, self.i_idx, self.j_idx] = vals_ij
        M[:, self.j_idx, self.i_idx] = vals_ji
        return M

    def noise(self, batch_size: int) -> torch.Tensor:
        """
        Unit thermal noise on the dense visibility matrix, (B, N, N) complex Hermitian:
        N[i, j] ~ CN(0, 1 / n_ij) (Re, Im each N(0, 1 / (2 n_ij))); zero on the diagonal and
        on unobserved baselines. Multiply by the per-integration noise std.

        N[j, i] = conj(N[i, j]), NOT an independent draw: the correlator forms one product per
        station pair, and V_ji is that same measurement conjugated (as the UV pixel at −u holds
        the conjugates of the samples at +u). Independent noise on (j, i) would count each
        measurement twice.
        """
        vals = torch.randn(batch_size, len(self.counts), dtype=torch.complex64, device=self.device)
        vals = vals * self.noise_scale
        return self._to_matrix(vals, vals.conj())

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """uv_sky: (B, H, W) or (B, H*W) complex, FFT order (image_to_uv output) → S (B, N, N)."""
        image = image.to(self.device)
        image = image.reshape(image.shape[0], -1)
        vals_ij = (image[:, self.flat_idx] * self.t).sum(dim=-1) * self.inv_counts        # +u track
        vals_ji = (image[:, self.flat_idx_conj] * self.t).sum(dim=-1) * self.inv_counts   # −u track
        return self._to_matrix(vals_ij, vals_ji)


# ===========================================================================
# 7. Measurement
# ===========================================================================

class PowerSpectrumMeasurer:
    """
    Azimuthally averaged power spectrum of batched 2D images (log-spaced k-bins).

    P(k) = |δ̃_ortho(k)|² dx dy, δ̃_ortho = rfft2(image - mean, norm="ortho"); by Parseval the
    mean of P(k) / (dx dy) over the full FFT grid equals the pixel variance. The k-grid, bins
    and mode counts are set up once in __init__; each call is one rfft2 + two scatter_add_.
    """

    def __init__(self, shape: tuple[int, int], BoxSize: float, n_bins: int = 50, device=None):
        """shape: (H, W) of the images;  BoxSize [Mpc];  n_bins log-spaced bins."""
        H, W = shape
        device = device or torch.device("cpu")
        dx, dy = BoxSize / W, BoxSize / H
        self.dx_dy  = dx * dy
        self.n_bins = n_bins
        self.N      = H * (W // 2 + 1)

        k      = make_kgrid((H, W), BoxSize, device, rfft=True)      # (H, W//2+1)
        k_flat = k.reshape(self.N)
        k_min  = k_flat[k_flat > 0].min().item()
        k_max  = k_flat.max().item()
        bin_edges = torch.logspace(np.log10(k_min), np.log10(k_max), steps=n_bins + 1, device=device)
        self.k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])      # (n_bins,)

        bin_idx = torch.bucketize(k_flat, bin_edges) - 1              # bins are (e_b, e_b+1]
        # guard: logspace round-off can put e_0 just above k_min (not on the 128 px / 512 Mpc
        # grid), which would give the k_min ring -1 like DC; put it in bin 0 (DC stays -1 = invalid)
        bin_idx[(k_flat > 0) & (bin_idx < 0)] = 0
        valid = (bin_idx >= 0) & (bin_idx < n_bins)
        self.valid_f         = valid.float()
        self.bin_idx_clamped = bin_idx.clamp(0, n_bins - 1)

        counts = torch.zeros(n_bins, device=device)
        counts.scatter_add_(0, self.bin_idx_clamped, self.valid_f)
        self.safe_counts = counts.clamp(min=1).unsqueeze(0)           # (1, n_bins)
        self.empty       = counts < 1                                  # (n_bins,)

    def __call__(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        images: (B, H, W) real (mean-subtracted internally)
        → k_centers (n_bins,), pk_mean (B, n_bins), pk_std (B, n_bins); empty bins are NaN.
        """
        B = images.shape[0]
        device = images.device
        valid_f         = self.valid_f.to(device)
        bin_idx_clamped = self.bin_idx_clamped.to(device)
        safe_counts     = self.safe_counts.to(device)

        images  = images - images.mean(dim=(-2, -1), keepdim=True)
        delta_k = torch.fft.rfft2(images, norm="ortho")
        p_valid = (delta_k.abs() ** 2 * self.dx_dy).reshape(B, self.N) * valid_f

        idx_exp = bin_idx_clamped.unsqueeze(0).expand(B, self.N)
        pk_sum  = torch.zeros(B, self.n_bins, device=device, dtype=p_valid.dtype)
        pk_sum2 = torch.zeros(B, self.n_bins, device=device, dtype=p_valid.dtype)
        pk_sum.scatter_add_(1, idx_exp, p_valid)
        pk_sum2.scatter_add_(1, idx_exp, p_valid ** 2)

        pk_mean = pk_sum / safe_counts
        pk_std  = (pk_sum2 / safe_counts - pk_mean ** 2).clamp(min=0).sqrt()

        empty = self.empty.to(device)
        pk_mean[:, empty] = float("nan")
        pk_std[:, empty]  = float("nan")
        return self.k_centers.to(device), pk_mean, pk_std


def drop_roundoff_bins(pk: torch.Tensor, rtol: float = 1e-11, fill: float = float("nan")) -> torch.Tensor:
    """
    Set k-bins that hold only FFT round-off (no sampled UV mode) to `fill`; NaN bins too.

    RELATIVE to the peak along the last (bin) axis, round-off bins measure <= 4e-15 and bins
    with data >= 1.5e-6 (pk_obs, 1024 prior draws with random gains, current set-up), so
    rtol = 1e-11 separates them by > 3 decades either way. Thermal noise
    does not change this: it is zero on unsampled UV pixels.
    pk: (..., n_bins).
    """
    peak = torch.nan_to_num(pk, nan=0.0).amax(dim=-1, keepdim=True)
    return torch.where(pk > rtol * peak, pk, torch.full_like(pk, fill))


# ===========================================================================
# 8. Units: mK <-> Jy
# ===========================================================================
# The pipeline works in mK: images are brightness temperatures [mK] and visibilities are their
# ortho-FFT (image_to_uv), i.e. Σ_pix T e^{-2πi u·θ} / N. A real interferometer measures
# V = Σ_pix S_pix e^{-2πi u·θ} in Jy, with the flux of a pixel S = 2 k_B T Ω_pix / λ²
# (Rayleigh-Jeans). Hence, per pixel and per visibility,
#
#     T [mK]     = S [Jy]  × mK_per_jansky("image")       = S λ² / (2 k_B Ω_pix) × 1e3 × 1e-26
#     F [mK]     = V [Jy]  × mK_per_jansky("uv")          = the image factor / N
#
# with λ = λ_21 (1 + z) and Ω_pix = (pixel size / D_c(z))². The two domains differ by N (the
# ortho-FFT normalisation), so the domain must always be given.

K_BOLTZMANN = 1.380649e-23      # [J / K]
C_LIGHT     = 2.99792458e8      # [m / s]
NU_21CM     = 1420.405751768e6  # [Hz]
JANSKY      = 1e-26             # [W m^-2 Hz^-1]
# also used by mean_brightness_temperature (section 4)
H_PLANCK    = 6.62607015e-34    # [J s]
A10_HI      = 2.85e-15          # [1 / s] 21-cm spontaneous emission rate (Furlanetto et al. 2006)
M_PROTON    = 1.67262192369e-27 # [kg]
G_NEWTON    = 6.67430e-11       # [m³ / (kg s²)]
MPC         = 3.0856775814913673e22  # [m]


def comoving_distance(z: float, h: float = settings.HUBBLE_H, omegam: float = settings.OMEGA_M,
                      n_steps: int = 100_000) -> float:
    """
    Comoving distance [Mpc] to redshift z in flat ΛCDM (radiation neglected):
    D_c = (c / H0) ∫_0^z dz' / sqrt(Ω_m (1 + z')³ + 1 − Ω_m), trapezoid rule.

    Defaults: Planck 2018 (settings.py), D_c(9) = 9400.70 Mpc — the distance the UV tracks file
    (settings.PADDED_PATH, made with tools21cm 2.4.3 set to the same cosmology) was gridded with.
    """
    zz = np.linspace(0.0, z, n_steps + 1)
    f = 1.0 / np.sqrt(omegam * (1.0 + zz) ** 3 + 1.0 - omegam)
    integral = (f[:-1] + f[1:]).sum() * 0.5 * (zz[1] - zz[0])
    return C_LIGHT / 1e3 / (100.0 * h) * integral


def mK_per_jansky(domain: str, config: SimulationConfig | None = None) -> float:
    """
    Conversion factor [mK per Jy] for this grid and redshift.

    domain: "image" — brightness temperature [mK] of one pixel holding 1 Jy
            "uv"    — pipeline visibility (ortho-FFT of an mK image) per Jy of visibility
                      (= the "image" factor / N); thermal noise and V_ref live here
    config: grid (shape, BoxSize) and cosmology (z, h, omegam); default SimulationConfig(),
            i.e. src/settings.py.
    """
    if domain not in ("image", "uv"):
        raise ValueError(f"domain={domain!r}, expected 'image' or 'uv'")
    cfg = config or SimulationConfig()
    lam = C_LIGHT / (NU_21CM / (1.0 + cfg.z))                               # [m]
    omega_pix = (cfg.dx / comoving_distance(cfg.z, cfg.h, cfg.omegam)) ** 2    # [sr]
    per_pixel = lam ** 2 / (2.0 * K_BOLTZMANN * omega_pix) * JANSKY * 1e3      # [mK / Jy]
    return per_pixel if domain == "image" else per_pixel / cfg.shape


def mK_to_jansky(x, domain: str, config: SimulationConfig | None = None):
    """x in the pipeline's mK units → Jy (per pixel for "image", per visibility for "uv").
    Works on tensors, arrays and scalars; keeps dtype (float / complex) and device."""
    return x / mK_per_jansky(domain, config)


def jansky_to_mK(x, domain: str, config: SimulationConfig | None = None):
    """x in Jy (per pixel for "image", per visibility for "uv") → the pipeline's mK units.
    Works on tensors, arrays and scalars; keeps dtype (float / complex) and device."""
    return x * mK_per_jansky(domain, config)
