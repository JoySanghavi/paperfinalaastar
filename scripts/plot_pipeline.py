"""
Stage-by-stage figures of the simulator with value checks, noise = 0, using the truth of
src/settings.py and the node classes of src/model.py (so the plots test the real code).

  01_science_sky.png   δ, Tb (true sky), |uvimage|, measured vs analytic P(k)
  02_uv_and_gains.png  UV coverage per subset, true gains, gain-weighted UV maps
  03_observed_sky.png  true sky (no DC) vs dirty images (unit / true gains), PSF, pk_obs
  04_reference.png     calibrator, S_ref, V_ref = G S_ref G†, first-row checks

Conventions: images have the phase (beam) centre at pixel (N//2, N//2); UV arrays are in
FFT order (DC at [0, 0]) and are fftshift-ed ONLY for display, so the plotted UV axes are
offsets from DC. Run from the project root:  python scripts/plot_pipeline.py
Figures go to figures/pipeline_plots/. Runs on the GPU if there is one, else on the CPU. The true
field drawn here equals data/true_delta.npy only on the device data/ was made on
(data/mock_device.npy; torch's random numbers differ between CPU and GPU) — the check says so.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import functionlistnew as func
import model
import settings

OUT = settings.FIGURES_DIR / "pipeline_plots"
OUT.mkdir(parents=True, exist_ok=True)
dev = model.device
N, L = model.config.shape, model.config.BoxSize
dx = L / N
knownant = model.KNOWN_ANT
nant = model.NANT

# True parameters — identical to scripts/gen_mock_data.py
true_z = settings.true_z(model.N_UNKNOWN).numpy()
true_pw = np.array([settings.TRUE_PW], dtype=np.float32)
FIELD_SEED = settings.TRUE_FIELD_SEED

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BAD = "#0b0b0b", "#52514e", "#e4e3df", "#e8e8e8"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
    "legend.frameon": False, "legend.fontsize": 8,
})
SEQ, DIV, CYC = "Blues", "RdBu_r", "twilight"


def cmap(name):
    c = plt.get_cmap(name).copy()
    c.set_bad(BAD)
    return c


def cbar(fig, im, ax, label):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(label, fontsize=8, color=INK2)
    cb.outline.set_visible(False)


IMG_EXT = [0, L, L, 0]                                   # Mpc, origin top-left (array[y, x])
UV_EXT = [-N // 2 - 0.5, N // 2 - 0.5, N // 2 - 0.5, -N // 2 - 0.5]   # UV px offset from DC
shift = np.fft.fftshift


def show_img(fig, ax, a, title, label, norm=None, cm=SEQ, ext=IMG_EXT, unit="Mpc"):
    im = ax.imshow(a, cmap=cmap(cm), norm=norm, extent=ext, interpolation="nearest")
    ax.set_title(title, loc="left")
    ax.set_xlabel(f"x [{unit}]"); ax.set_ylabel(f"y [{unit}]")
    cbar(fig, im, ax, label)
    return im


def show_uv(fig, ax, a, title, label, norm=None, cm=SEQ):
    im = ax.imshow(shift(a), cmap=cmap(cm), norm=norm, extent=UV_EXT, interpolation="nearest")
    ax.set_title(title, loc="left")
    ax.set_xlabel("u [px from DC]"); ax.set_ylabel("v [px from DC]")
    cbar(fig, im, ax, label)
    return im


checks = []


def check(name, value, expect, ok, status=None):
    checks.append((name, value, expect, status or ("PASS" if ok else "FAIL")))


def np_(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else t


# ===========================================================================
# Run the pipeline with model.py nodes (B = 1, noise = 0)
# ===========================================================================
gains = model.GainsFromZ().simulate_batch(1, true_z)
torch.manual_seed(FIELD_SEED)
df = model.DensityFromPw().simulate_batch(1, true_pw)                  # δ
bt = model.BrightnessFromDensity().simulate_batch(1, df)               # Tb [mK]
image = model.SkyImageFromBrightness().simulate_batch(1, bt)           # sky image [mK]
uvimage = model.UVSkyFromImage().simulate_batch(1, image)              # FFT order
stacked = model.UVTracksFromGains().simulate_batch(1, gains)
tracked = model.TrackedUVFromSky().simulate_batch(1, uvimage, stacked)
normed = model.NormalisedUV().simulate_batch(1, tracked)
noiseimg = model.ThermalNoiseUV(noise_std=0.0).simulate_batch(1, uvimage)    # noise-free figures
x = model.VisibilitiesWithNoise().simulate_batch(1, normed, noiseimg)
obsx = model.DirtyImageFromUV().simulate_batch(1, x)                   # (1, 2, N, N) real [mK]
pk_obs = model.PkFromDirtyImage().simulate_batch(1, obsx)              # (1, 2, n_bins)

# Same chain with unit gains = perfectly calibrated instrument
unit_gains = np_(func.theta_to_gains(torch.zeros(1, 2 * (nant - 1 - knownant)), knownant))
st_u = model.UVTracksFromGains().simulate_batch(1, unit_gains)
x_u = model.NormalisedUV().simulate_batch(1, model.TrackedUVFromSky().simulate_batch(1, uvimage, st_u))
obsx_u = model.DirtyImageFromUV().simulate_batch(1, x_u)
pk_u = model.PkFromDirtyImage().simulate_batch(1, obsx_u)

unit_cnt = np_(model.UV_COUNT)[0]                                      # (2, N, N) integrations / px
sampled = unit_cnt > 0
g = np_(func.gains_to_complex(torch.tensor(gains)))[0]

delta, Tb, img, uv = df[0], bt[0], image[0], uvimage[0]

# ---- value checks: science path ----
T0 = float(model.brightness(torch.zeros(1, device=dev)))                # T_b of δ = 0 [mK]
check(f"Tb = T0 (1+δ), T0 = {T0:.3f} mK (Planck 2018, z = 9)", np.abs(Tb - T0 * (1 + delta)).max(), "≈ 0",
      np.abs(Tb - T0 * (1 + delta)).max() < 1e-3)
check("uvimage DC = Σ image / N (ortho)", abs(uv[0, 0] - img.sum() / N), "≈ 0", abs(uv[0, 0] - img.sum() / N) < 1e-2)
rt = np_(func.uv_to_image(torch.tensor(uv)).real)
check("uv_to_image(image_to_uv(image)) = image", np.abs(rt - img).max(), "≈ 0", np.abs(rt - img).max() < 1e-3)
err_u = np.abs(x_u[0] - uv[None] * sampled).max()
check("unit gains: x = uvimage on sampled px, 0 elsewhere", err_u, "≈ 0", err_u < 1e-4)
check("x is Hermitian: max |x(-k) - x*(k)|",
      np.abs(np.roll(x[0][..., ::-1, ::-1], 1, axis=(-2, -1)) - x[0].conj()).max(), "≈ 0",
      np.abs(np.roll(x[0][..., ::-1, ::-1], 1, axis=(-2, -1)) - x[0].conj()).max() < 1e-4)
td = np.load(settings.DATA_DIR / "true_delta.npy")[0]                   # the mock observation's field
md_path = settings.DATA_DIR / "mock_device.npy"
mock_dev = str(np.load(md_path)) if md_path.exists() else "unknown"
run_dev = torch.device(dev).type
same_field = np.allclose(delta, td, rtol=0, atol=1e-6)
check(f"δ = data/true_delta.npy (data/ made on {mock_dev}, this run on {run_dev})",
      np.abs(delta - td).max(), "≈ 0", same_field,
      status=None if same_field or mock_dev == run_dev else "NOTE: other device, seeded draw differs (expected)")
check("obsx mean (DC dropped)", np.abs(obsx[0].mean(axis=(-2, -1))).max(), "≈ 0", np.abs(obsx[0].mean(axis=(-2, -1))).max() < 1e-4)

# ===========================================================================
# Figure 1 — science sky
# ===========================================================================
fig, ax = plt.subplots(1, 4, figsize=(21, 4.8), constrained_layout=True)
m = np.abs(delta).max()
show_img(fig, ax[0], delta, "1a  density contrast δ", "δ", norm=TwoSlopeNorm(0, -m, m), cm=DIV)
show_img(fig, ax[1], Tb, "1b  TRUE SKY: brightness temperature", "Tb [mK]")
ax[1].plot(L / 2, L / 2, "+", color=ORANGE, ms=12, mew=2)
ax[1].annotate("beam / phase centre", (L / 2, L / 2), (L * 0.52, L * 0.42), color=INK, fontsize=8)
show_uv(fig, ax[2], np.abs(uv), "1c  |uvimage| = |FFT(Tb)| (DC centred for display)", "|V| [mK]",
        norm=LogNorm(np.abs(uv)[np.abs(uv) > 0].min() * 10, np.abs(uv).max()))
pkm = func.PowerSpectrumMeasurer((N, N), L, device="cpu")
# many realisations for the P(k) check (single field is noisy)
torch.manual_seed(0)
B = 300
pwb = torch.tensor(true_pw).repeat(B, 1)
pwd = pwb.to(dev)                                  # the simulator's k-grid lives on model.device
db = model.density_sim.sample_field(*func.pw_to_params(pwd)).cpu()
k, pk_many, _ = pkm(db)
_, pk_true, _ = pkm(torch.tensor(delta)[None])
pk_ana = np_(func.power_spectrum_2d(k.cpu(), 10 ** pwb[:1, 0], pwb[:1, 1], pwb[:1, 2], 10 ** pwb[:1, 3])).reshape(-1)
k = np_(k)
a = ax[3]
a.loglog(k, pk_ana, color=BLUE, label="analytic P(k)")
a.loglog(k, np_(pk_many.mean(0)), "o", color=ORANGE, ms=4, label=f"measured, mean of {B} fields")
a.loglog(k, np_(pk_true[0]), "o-", color=AQUA, lw=1, ms=2.5, label="measured, the true field")   # markers: empty bins are NaN
a.axvline(10 ** true_pw[0, 3], color=INK2, lw=1, ls=":")
a.text(10 ** true_pw[0, 3] * 1.05, pk_ana[np.isfinite(pk_ana)].max() * 0.5, "k_break", color=INK2, fontsize=8)
a.set_xlabel("k [Mpc⁻¹]"); a.set_ylabel("P(k) [Mpc²]"); a.set_title("1d  P(k) of δ", loc="left")
a.grid(True, which="major", color=GRID, lw=0.6); a.legend()
fig.savefig(OUT / "01_science_sky.png", dpi=110)
plt.close(fig)
ratio = np_(pk_many.mean(0)) / pk_ana
check("measured/analytic P(k) of δ (median over bins)", np.nanmedian(ratio), "≈ 1", abs(np.nanmedian(ratio) - 1) < 0.05)

# ===========================================================================
# Figure 2 — UV coverage and gains
# ===========================================================================
fig, ax = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)
for s in range(2):
    c = np.where(sampled[s], unit_cnt[s], np.nan)
    show_uv(fig, ax[0, s], c, f"2{'ab'[s]}  UV coverage, subset {s}  ({sampled[s].mean():.1%} of px)",
            "30 s integrations per px", norm=LogNorm(1, np.nanmax(unit_cnt)))
idx = np.arange(nant)
groups = [(idx == 0, "reference g₀ = 1", INK), ((idx >= 1) & (idx < nant - knownant), "unknown (from z)", BLUE),
          (idx >= nant - knownant, f"known g = 1 ({knownant})", ORANGE)]
for msk, lab, col in groups:
    ax[0, 2].plot(idx[msk], np.abs(g[msk]), "o", ms=3, color=col, label=lab)
    ax[0, 3].plot(idx[msk], np.angle(g[msk]), "o", ms=3, color=col, label=lab)
ax[0, 2].set_title("2c  true gain amplitude |gᵢ|", loc="left"); ax[0, 2].set_xlabel("antenna"); ax[0, 2].set_ylabel("|g|")
ax[0, 3].set_title("2d  true gain phase arg gᵢ", loc="left"); ax[0, 3].set_xlabel("antenna"); ax[0, 3].set_ylabel("rad")
for a in ax[0, 2:]:
    a.grid(True, color=GRID, lw=0.6); a.legend(loc="upper right")
eff = np.where(sampled, stacked[0] / np.where(sampled, unit_cnt, 1), np.nan)   # Σ t gᵢgⱼ* / Σ t
for s in range(2):
    show_uv(fig, ax[1, 2 * s], np.abs(eff[s]), f"2{'eg'[s]}  |gain-weighted UV map| subset {s}",
            "|Σ t gᵢgⱼ* / Σ t|", norm=LogNorm(1e-2, max(1, np.nanmax(np.abs(eff)))))
    show_uv(fig, ax[1, 2 * s + 1], np.angle(eff[s]), f"2{'fh'[s]}  phase of gain-weighted UV map, subset {s}",
            "rad", cm=CYC)
    ax[1, 2 * s + 1].images[0].set_clim(-np.pi, np.pi)
fig.savefig(OUT / "02_uv_and_gains.png", dpi=110)
plt.close(fig)

# ===========================================================================
# Figure 3 — observed sky
# ===========================================================================
fig, ax = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)
sky0 = img - img.mean()
m = np.abs(sky0).max()
nrm = TwoSlopeNorm(0, -m, m)
show_img(fig, ax[0, 0], sky0, "3a  true sky − mean (an interferometer has no DC)", "Tb [mK]", norm=nrm, cm=DIV)
show_img(fig, ax[0, 1], obsx_u[0, 0], "3b  dirty image, UNIT gains, subset 0", "Tb [mK]", norm=nrm, cm=DIV)
show_img(fig, ax[0, 2], obsx[0, 0], "3c  OBSERVED SKY obsx, true gains, subset 0", "Tb [mK]",
         norm=TwoSlopeNorm(0, -np.abs(obsx).max(), np.abs(obsx).max()), cm=DIV)
show_img(fig, ax[0, 3], obsx[0, 1], "3d  OBSERVED SKY obsx, true gains, subset 1", "Tb [mK]",
         norm=TwoSlopeNorm(0, -np.abs(obsx).max(), np.abs(obsx).max()), cm=DIV)
# PSF: point source at the centre through the unit-gain instrument
pt = torch.zeros(1, N, N, device=dev); pt[0, N // 2, N // 2] = 1.0
uv_pt = model.UVSkyFromImage().simulate_batch(1, np_(pt))
psf = model.DirtyImageFromUV().simulate_batch(1, model.NormalisedUV().simulate_batch(
    1, model.TrackedUVFromSky().simulate_batch(1, uv_pt, st_u)))[0]
z = 16
c0 = N // 2
zext = [(c0 - z) * dx, (c0 + z + 1) * dx, (c0 + z + 1) * dx, (c0 - z) * dx]   # same pixel edges as IMG_EXT
for s in range(2):
    p = psf[s, c0 - z:c0 + z + 1, c0 - z:c0 + z + 1] / psf[s].max()
    show_img(fig, ax[1, s], p, f"3{'ef'[s]}  dirty beam (PSF), subset {s}, peak at px {tuple(int(v) for v in np.unravel_index(psf[s].argmax(), (N, N)))}",
             "normalised", norm=TwoSlopeNorm(0, -1, 1), cm=DIV, ext=zext)
    check(f"PSF subset {s} peaks at beam centre ({c0},{c0})", np.unravel_index(psf[s].argmax(), (N, N)), f"({c0}, {c0})",
          np.unravel_index(psf[s].argmax(), (N, N)) == (c0, c0))
# pk_obs: per subset, unit vs true gains, against 27² P_δ(true field) × sampled fraction
a = ax[1, 2]
pk_sky = np_(pkm(torch.tensor(Tb)[None])[1][0])          # P(k) of the true Tb field
cols = [BLUE, ORANGE]
# bins with no sampled UV mode hold only FFT round-off (~1e-12): hidden, as the embedding drops them
show = lambda pk: np_(func.drop_roundoff_bins(torch.tensor(pk)))
for s in range(2):
    a.loglog(k, show(pk_u[0, s]), "-", color=cols[s], label=f"pk_obs unit gains, subset {s}")
    a.loglog(k, show(pk_obs[0, s]), "o", ms=3, color=cols[s], label=f"pk_obs true gains, subset {s}")
a.loglog(k, pk_sky, "-", color=INK2, lw=1, label="P(k) of true Tb (all modes)")
a.set_xlabel("k [Mpc⁻¹]"); a.set_ylabel("P(k) [mK² Mpc²]")
a.set_title("3g  pk_obs (what the network sees; round-off bins hidden)", loc="left")
a.grid(True, color=GRID, lw=0.6); a.legend(fontsize=7)
# sampled fraction per k bin → expected pk_obs(unit gains) / P_sky
a = ax[1, 3]
kk = np_(func.make_kgrid((N, N), L, "cpu", rfft=False))
edges = np.logspace(np.log10(kk[kk > 0].min()), np.log10(kk.max()), 51)
for s in range(2):
    num = np.histogram(kk[sampled[s]], edges, weights=np.abs(uv[sampled[s]]) ** 2)[0]
    den = np.histogram(kk[kk > 0], edges, weights=np.abs(uv[kk > 0]) ** 2)[0]
    frac = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    a.semilogx(k, frac, "-", color=cols[s], label=f"expected, subset {s}")
    a.semilogx(k, pk_u[0, s] / pk_sky, "o", ms=3, color=cols[s], label=f"pk_obs / P_sky, subset {s}")
a.set_ylim(-0.05, 1.05); a.set_xlabel("k [Mpc⁻¹]"); a.set_ylabel("fraction of power kept")
a.set_title("3h  unit gains: pk_obs = P_sky × sampled fraction", loc="left")
a.grid(True, color=GRID, lw=0.6); a.legend(fontsize=7)
fig.savefig(OUT / "03_observed_sky.png", dpi=110)
plt.close(fig)

# ===========================================================================
# Figure 4 — reference calibrator, S_ref, V_ref
# ===========================================================================
D = model.sky_matrix
gauss = np_(model.BEAM_CENTRE_GAUSSIAN)
cal = settings.CALIBRATOR_FLUX * gauss                                 # calibrator image [mK]
uv_cal = np_(func.image_to_uv(torch.tensor(cal, device=dev)[None]))[0]
S_ref = np_(D(torch.tensor(uv_cal, device=dev)[None]))[0]              # fresh, current convention
check("model.S_REF = calibrator through the dense path", np.abs(np_(model.S_REF) - S_ref).max(), "0",
      np.abs(np_(model.S_REF) - S_ref).max() == 0)
V_ref = model.ReferenceVisFromGains(noise_std=0.0).simulate_batch(1, gains)[0]

cnt = np_(D.counts)
iu, ju = np_(D.i_idx), np_(D.j_idx)
obs = np.zeros((nant, nant), bool); obs[iu, ju] = cnt > 0; obs |= obs.T
A_exp = settings.CALIBRATOR_FLUX * gauss.sum() / N        # flat-spectrum |S_ref| = Σ calibrator / N [mK]
Sv = S_ref[iu, ju][cnt > 0]
# 0.3 px Gaussian on a pixel grid: |FFT| = (p/N)(1 + 2w(cos 2πu/N + cos 2πv/N) + ...),
# w = exp(-1/(2σ²)) → 3% lower at the Nyquist corner than at DC. A track average must lie inside.
lo, hi = np.abs(uv_cal).min(), np.abs(uv_cal).max()
check("|S_ref| inside [min, max] of |FFT(calibrator)| (σ = 0.3 px → 3% spread)",
      f"{np.abs(Sv).min():.5f}..{np.abs(Sv).max():.5f}", f"{lo:.5f}..{hi:.5f}",
      (np.abs(Sv).min() >= lo * (1 - 1e-5)) and (np.abs(Sv).max() <= hi * (1 + 1e-5)))
check("S_ref phase on observed baselines", f"{np.abs(np.angle(Sv)).max():.1e}", "0", np.abs(np.angle(Sv)).max() < 1e-4)
check("S_ref = 0 on unobserved baselines and diagonal", np.abs(S_ref[~obs]).max(), "0", np.abs(S_ref[~obs]).max() == 0)
Vexp = g[:, None] * S_ref * g.conj()[None, :]
check("V_ref (model node) = gᵢ S_ref gⱼ*", np.abs(V_ref - Vexp).max(), "≈ 0", np.abs(V_ref - Vexp).max() < 1e-6)
check("V_ref Hermitian", np.abs(V_ref - V_ref.conj().T).max(), "≈ 0", np.abs(V_ref - V_ref.conj().T).max() < 1e-6)

fig, ax = plt.subplots(2, 4, figsize=(21, 9.5), constrained_layout=True)
z = 4
p = cal[c0 - z:c0 + z + 1, c0 - z:c0 + z + 1]
zext = [(c0 - z) * dx, (c0 + z + 1) * dx, (c0 + z + 1) * dx, (c0 - z) * dx]   # same pixel edges as IMG_EXT
show_img(fig, ax[0, 0], p, f"4a  calibrator at beam centre px ({c0},{c0}), zoom", "Tb [mK]", ext=zext)
show_uv(fig, ax[0, 1], np.angle(uv_cal), "4b  phase of FFT(calibrator): 0 everywhere", "rad", cm=CYC)
ax[0, 1].images[0].set_clim(-np.pi, np.pi)
Sm = np.where(obs, np.abs(S_ref), np.nan)
im = ax[0, 2].imshow(Sm, cmap=cmap(SEQ), vmin=0, vmax=A_exp * 1.05, interpolation="nearest")
ax[0, 2].set_title("4c  |S_ref[i,j]|  (grey = baseline never on the grid)", loc="left")
ax[0, 2].set_xlabel("antenna j"); ax[0, 2].set_ylabel("antenna i"); cbar(fig, im, ax[0, 2], "[mK]")
a = ax[0, 3]
a.hist(np.abs(Sv) / A_exp, np.linspace(lo / A_exp - 0.005, hi / A_exp + 0.005, 41), color=BLUE)
for v in (lo, hi):
    a.axvline(v / A_exp, color=INK2, lw=1, ls=":")
a.set_yscale("log"); a.set_xlabel("|S_ref| / (Σ calibrator / N)"); a.set_ylabel("observed baselines")
a.set_title(f"4d  |S_ref| on the {len(Sv)} observed baselines", loc="left")
a.text(0.03, 0.95, "dotted: min / max of |FFT(calibrator)|", transform=a.transAxes, fontsize=8, color=INK2, va="top")
a.grid(True, color=GRID, lw=0.6)
Vm = np.where(obs, V_ref, np.nan)
im = ax[1, 0].imshow(np.abs(Vm), cmap=cmap(SEQ), interpolation="nearest")
ax[1, 0].set_title("4e  |V_ref| = |gᵢ| |S_ref| |gⱼ|", loc="left"); cbar(fig, im, ax[1, 0], "[mK]")
im = ax[1, 1].imshow(np.angle(Vm), cmap=cmap(CYC), vmin=-np.pi, vmax=np.pi, interpolation="nearest")
ax[1, 1].set_title("4f  arg V_ref = arg gᵢ − arg gⱼ", loc="left"); cbar(fig, im, ax[1, 1], "rad")
for a in ax[1, :2]:
    a.set_xlabel("antenna j"); a.set_ylabel("antenna i")
a = ax[1, 2]
row = V_ref[0, 1:][cnt[:nant - 1] > 0]                                  # first row (VRefFirstRowEmbedding input)
gg = (g[0] * g[1:].conj())[cnt[:nant - 1] > 0]
a.plot(np.angle(gg), np.angle(row), "o", ms=3, color=BLUE)
a.plot([-np.pi, np.pi], [-np.pi, np.pi], color=INK2, lw=1)
a.set_xlabel("arg(g₀ gⱼ*)  [true gains]"); a.set_ylabel("arg V_ref[0, j]")
a.set_title(f"4g  first row of V_ref: {len(row)} observed baselines", loc="left"); a.grid(True, color=GRID, lw=0.6)
a = ax[1, 3]
a.plot(np.abs(gg) * A_exp, np.abs(row), "o", ms=3, color=BLUE)
lim = [0, max(np.abs(row).max(), (np.abs(gg) * A_exp).max()) * 1.05]
a.plot(lim, lim, color=INK2, lw=1)
a.set_xlabel("|g₀ gⱼ*| × Σ calibrator / N  [mK]"); a.set_ylabel("|V_ref[0, j]| [mK]")
a.set_title("4h  amplitude check", loc="left"); a.grid(True, color=GRID, lw=0.6)
fig.savefig(OUT / "04_reference.png", dpi=110)
plt.close(fig)

# ===========================================================================
print(f"\nFigures in {OUT}\n")
w = max(len(c[0]) for c in checks)
for name, val, exp, st in checks:
    print(f"  [{st}] {name:<{w}}  got {val}  expected {exp}")
