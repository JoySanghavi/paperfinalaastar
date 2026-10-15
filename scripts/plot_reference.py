"""
How the Stage-1 reference calibrator looks to the instrument.

Builds a Gaussian reference sky of any width and follows it through both paths,
using only the simulators in src/functionlistnew.py and the set-up in src/settings.py:

  UV-map path :  image_ref -> image_to_uv -> x UV sampling (conjugates, no DC) -> uv_to_image = dirty image
  Dense path  :  image_ref -> image_to_uv -> SkyMatrixGenerator -> S_ref -> V_ref = G S_ref G^H

Use from a notebook (see notebooks/reference_calibrator.ipynb):
    view = reference_view(sigma=3.0)
    plot_reference_view(view)
    summarize(view)

Or from the command line (project root), which saves figures/ref_calibrator_view.png:
    python scripts/plot_reference.py [sigma]
"""

import sys
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SRC  = HERE.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import functionlistnew as func
import settings

device = "cuda" if torch.cuda.is_available() else "cpu"
SIGMA_REF = settings.CALIBRATOR_SIGMA   # calibrator width of the pipeline [px]
PREFACTOR = settings.CALIBRATOR_FLUX    # calibrator peak of the pipeline [mK]
KNOWNANT  = settings.KNOWN_ANT          # known (unit-gain) antennas
DYN_RANGE = 1e-8       # log colour scales span [max * DYN_RANGE, max]


def _np(t):
    """tensor (any device) -> numpy"""
    return t.detach().cpu().numpy()


@lru_cache(maxsize=1)
def instrument():
    """Simulators + gains shared by every view (built once)."""
    padded = torch.load(settings.PADDED_PATH, map_location=device, weights_only=True)
    nant   = func.nant_from_padded(padded)
    config = func.SimulationConfig(nant=nant, padded=padded, device=device,
                                   subsetindex=func.split_indices(padded.shape[0], settings.N_SUBSETS))
    # Same true gains as scripts/gen_mock_data.py; the last KNOWNANT antennas have unit gain
    true_z = settings.true_z(nant - 1 - KNOWNANT)
    i, j = torch.triu_indices(nant, nant, offset=1)
    dense = func.SkyMatrixGenerator(padded, config.shape, nant)
    return dict(
        config=config, nant=nant, i=i.numpy(), j=j.numpy(),
        observed=_np(dense.counts > 0),                # (n_baselines,) observed, DC excluded as in the pipeline
        UV_gen=func.UVMapGenerator(config),
        unit_uv=func.UnitUVMapGenerator(config)(),          # (1, 2, N, N)
        dense=dense,
        true_gains=func.theta_to_gains(true_z, KNOWNANT).to(device),
        unit_gains=func.theta_to_gains(torch.zeros(1, 2 * (nant - 1), device=device)),
    )


def reference_view(sigma=SIGMA_REF, prefactor=PREFACTOR, center=None):
    """Push a Gaussian reference blob of width `sigma` (px) through the instrument.

    center: pixel of the blob centre; default N//2 (on a pixel, as gaussian_2d does).
    Pass (N-1)/2 for a between-pixel centre.
    """
    ins = instrument()
    N = ins["config"].shape
    center = N // 2 if center is None else center               # gaussian_2d's default

    image_ref   = prefactor * func.gaussian_2d(N, sigma, device, center=center)  # (N, N)
    uvimage_ref = func.image_to_uv(image_ref.unsqueeze(0))                     # (1, N, N)

    def mask(gains):
        return torch.nan_to_num(ins["UV_gen"](gains) / ins["unit_uv"], nan=0.0)[0]  # (2, N, N)

    def dirty(gains):
        """Dirty image, averaged over the 2 subsets: ifft2(M * uvimage)."""
        return _np(func.uv_to_image(mask(gains) * uvimage_ref).real.mean(0))

    S_ref = ins["dense"](uvimage_ref)                                  # (1, nant, nant)
    V_ref = _np(func.visibilities_from_gains(ins["true_gains"], S_ref)[0])
    return dict(
        sigma=sigma, center=center,
        image=_np(image_ref),
        uvimage=_np(uvimage_ref[0]),
        coverage=_np(ins["unit_uv"][0].abs().sum(0)),              # 30-s integrations per cell, both subsets
        sampled=_np((mask(ins["true_gains"]) * uvimage_ref).abs().sum(0)),
        dirty_unit=dirty(ins["unit_gains"]),
        dirty_true=dirty(ins["true_gains"]),
        S_ref=_np(S_ref[0]),
        V_ref=V_ref,
    )


def summarize(view, rel_threshold=1e-3):
    """Key numbers: how much of the calibrator the instrument actually sees."""
    ins = instrument()
    i, j, obs = ins["i"], ins["j"], ins["observed"]
    s = np.abs(view["S_ref"][i, j])
    usable = s > rel_threshold * s.max() if s.max() > 0 else np.zeros_like(obs)
    P = np.abs(view["uvimage"]) ** 2
    P[0, 0] = 0.0                                                      # DC is never sampled
    out = {
        "sigma (px)":                          view["sigma"],
        "observed baselines":                  int(obs.sum()),
        f"usable baselines (|S|>{rel_threshold:g} max)": int((usable & obs).sum()),
        f"usable first-row entries (of {ins['nant'] - 1})": int(usable[i == 0].sum()),
        "non-DC sky power on sampled cells":   float(P[view["coverage"] > 0].sum() / max(P.sum(), 1e-300)),
        "sky peak":                            float(view["image"].max()),
        "dirty peak (unit gains)":             float(view["dirty_unit"].max()),
        "rel. change of dirty image from gains": float(
            np.linalg.norm(view["dirty_true"] - view["dirty_unit"])
            / max(np.linalg.norm(view["dirty_unit"]), 1e-300)),
    }
    return out


def _lognorm(m):
    pos = m[m > 0]
    if pos.size == 0:
        return None
    vmax = pos.max()
    return LogNorm(vmin=max(pos.min(), vmax * DYN_RANGE), vmax=vmax)


def plot_reference_view(view, zoom=None, save=None):
    """2x4 figure: sky, |FFT|, UV sampling, sampled |V|; dirty (unit / true gains), |S_ref|, |V_ref|."""
    nant = instrument()["nant"]
    N = view["image"].shape[0]
    c = int(round(view["center"]))
    zoom = int(zoom or max(12, np.ceil(4 * view["sigma"])))
    zoom = min(zoom, c, N - 1 - c)
    zs = slice(c - zoom, c + zoom + 1)
    ext_z = [-zoom - 0.5, zoom + 0.5, zoom + 0.5, -zoom - 0.5]     # px offsets from centre
    k0 = N // 2
    ext_k = [-k0 - 0.5, k0 - 0.5, k0 - 0.5, -k0 - 0.5]             # UV px offsets from DC
    shift = np.fft.fftshift
    seq, div, grey = "Blues", "RdBu_r", "#e8e8e8"

    fig, ax = plt.subplots(2, 4, figsize=(19, 9.5))

    def cbar(im, a, label):
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.03).set_label(label, fontsize=9)

    def logpanel(a, m, title, label, extent=None, **kw):
        m = np.where(m > 0, m, np.nan)
        im = a.imshow(m, cmap=seq, norm=_lognorm(np.nan_to_num(m)), extent=extent, **kw)
        a.set_facecolor(grey)
        a.set_title(title, fontsize=11)
        cbar(im, a, label)

    # Row 1: sky -> Fourier -> sampling -> sampled visibilities
    im = ax[0, 0].imshow(view["image"][zs, zs], cmap=seq, extent=ext_z)
    ax[0, 0].set_title(rf"Reference sky  ($\sigma$={view['sigma']:g} px, zoom $\pm${zoom})", fontsize=11)
    cbar(im, ax[0, 0], "T_b [mK]")
    logpanel(ax[0, 1], shift(np.abs(view["uvimage"])), "|FFT(sky)|", "|V| (log)", ext_k)
    cov = shift(view["coverage"])
    logpanel(ax[0, 2], cov, f"UV sampling ({int((cov > 0).sum())}/{N * N} cells)",
             "30-s integrations per cell (log)", ext_k)
    logpanel(ax[0, 3], shift(view["sampled"]), "|sampled visibilities|, true gains",
             "|V| summed over subsets (log)", ext_k)
    ax[0, 0].set_xlabel("x  (px from centre)")
    ax[0, 0].set_ylabel("y  (px from centre)")
    for a in ax[0, 1:]:
        a.set_xlabel("u  (px from DC)")
        a.set_ylabel("v  (px from DC)")

    # Row 2: image-domain path (dirty images) and dense path (S_ref, V_ref)
    v = max(np.abs(view["dirty_unit"][zs, zs]).max(), np.abs(view["dirty_true"][zs, zs]).max())
    for a, img, title in [(ax[1, 0], view["dirty_unit"], "Dirty image, unit gains (= PSF * sky)"),
                          (ax[1, 1], view["dirty_true"], "Dirty image, true gains")]:
        im = a.imshow(img[zs, zs], cmap=div, vmin=-v, vmax=v, extent=ext_z)
        a.set_title(title, fontsize=11)
        a.set_xlabel("x  (px from centre)")
        a.set_ylabel("y  (px from centre)")
        cbar(im, a, "T_b [mK]")
    for a, mat, title in [(ax[1, 2], view["S_ref"], r"$|S_\mathrm{ref}|$  (dense sky, track-averaged, no DC)"),
                          (ax[1, 3], view["V_ref"], r"$|V_\mathrm{ref}| = |G S_\mathrm{ref} G^\dagger|$")]:
        logpanel(a, np.abs(mat), title, "|value| (log); grey = 0 / unobserved")
        a.set_xlabel("antenna j")
        a.set_ylabel("antenna i")

    fig.suptitle(f"Stage-1 reference calibrator as seen by the instrument "
                 f"({nant} antennas; dirty images averaged over the 2 subsets)", fontsize=13)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    sigma = float(sys.argv[1]) if len(sys.argv) > 1 else SIGMA_REF
    view = reference_view(sigma)
    settings.FIGURES_DIR.mkdir(exist_ok=True)
    out = settings.FIGURES_DIR / "ref_calibrator_view.png"
    plot_reference_view(view, save=out)
    print(f"Saved {out}")
    for k, val in summarize(view).items():
        print(f"  {k:48s} {val:.4g}" if isinstance(val, float) else f"  {k:48s} {val}")
