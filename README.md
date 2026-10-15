# Joint gain calibration and 21-cm power-spectrum inference — SKA-Low AA*

A radio interferometer does not see the sky directly: every baseline (pair of stations i, j)
measures the sky's Fourier transform at one point of the UV plane, multiplied by the unknown
complex gains of its two stations, g_i g_j*. This project asks: **can the 21-cm power spectrum of
the sky be inferred while the antenna gains are still unknown**, i.e. calibrating and doing
science in one step?

It does this with **simulation-based inference** ([falcon](https://pypi.org/project/falcon-sbi/)):
a simulator of the sky and of SKA-Low AA* produces many mock observations; neural networks learn
from them the posterior of the gains (from a calibrator observation) and of the power-spectrum
parameters (from the observed sky, marginalising over the gains). A Fisher forecast gives an
independent check of the power-spectrum errors.

| | |
|---|---|
| Instrument | SKA-Low AA*, 307 stations (46 971 baselines), one 4 h track of 30 s integrations, z = 9 |
| Sky grid | 128 × 128 pixels over 512 cMpc (4 Mpc pixels); UV tracks pre-computed in `src/paddedAAstar_4_30_9_128_512.pth` |
| Science sky | 21-cm brightness temperature T_b = 26.60 (1 + δ) mK (Planck 2018, z = 9, fully neutral, T_S ≫ T_CMB), δ a Gaussian field with a broken power law |
| Cosmology | Planck 2018: h = 0.674, Ω_m = 0.315, Ω_b = 0.0492, Y_p = 0.2454 (`src/settings.py`); D_c(9) = 9401 Mpc |
| Parameters | `pw` = [log10 A, n1, n2, log10 k_break] (power spectrum); `z` = 212 numbers, the gains g = 1 + z of 106 unknown stations (station 0 is the reference, g_0 = 1; 200 stations have known g = 1) |
| Calibrator | a point-like source (0.3 px Gaussian, 2.7 mK) at the beam centre with a known model |
| Observations | `V_ref` (calibrator visibilities, 307 × 307) and `pk_obs` (P(k) of the dirty science image, 2 UV subsets × 50 k-bins) |
| Noise | thermal noise is implemented but **off** (`NOISE_STD = 0`); see [Thermal noise](#thermal-noise) |
| Status | `output/run_04` was trained at the previous truth and priors, before the switch to Planck 2018 and the first-principles T_b; retrain for a run of the current set-up. See [Results so far](#results-so-far-run_04) |

---

## What is in this folder

```
config.yml               falcon graph: which node simulates what, the priors, the two estimators,
                         the embedding networks, buffer and sampling settings
README.md                this file

src/                     the simulator (falcon imports it via config.yml `paths.import`)
  settings.py            THE place to change the set-up: grid, box, z, antennas, noise, calibrator,
                         true parameters and seeds. Every script and notebook reads its numbers here
  functionlistnew.py     simulation library: gains, FFT conventions, sky model, UV gridding,
                         dense (per-baseline) path, P(k) estimator. No falcon dependency
  model.py               one class per falcon graph node, wiring functionlistnew into the graph
  embedding.py           the networks that compress V_ref and pk_obs before the estimators
  paddedAAstar_4_30_9_128_512.pth   UV tracks of all baselines (FFT-ordered pixel, 30 s counts)
  skalow_AAstar_layout.txt          station positions the tracks were made from

data/                    the mock observation falcon conditions on, and its truth (all *.npy),
                         written by scripts/gen_mock_data.py; mock_device.npy records the device
scripts/
  gen_mock_data.py       makes data/ with the same simulator nodes falcon trains on
  plot_pipeline.py       a figure of every simulator stage + 16 value checks -> figures/pipeline_plots/
  plot_reference.py      how the instrument sees the calibrator -> figures/ref_calibrator_view.png

fisher/                  independent Fisher forecast of the pw errors (fixed, known gains)
  fisher_forecast.py     Monte-Carlo covariance + gradients of the band powers -> fisher_results.npz
  fisher_config.py       the forecast knobs (DATA_SPACE, FISHER, WITH_PRIOR), shared
  fisher_assembly.py     the Fisher-matrix maths, shared by the script and the notebook
  fisher_analysis.ipynb  checks of the forecast and the comparison with the SBI posterior of run_04
  fisher_results.npz     the stored forecast

notebooks/               analysis
  plot_posteriors.ipynb  posterior of run_04: corner plots, gain marginals and accuracy,
                         buffer/posterior GIFs
  imageanalysis.ipynb    image reconstruction from the observed UV maps (true / no / self-cal gains)
  reference_calibrator.ipynb   how much of the calibrator each baseline sees, for several widths
  noise.ipynb            radiometer equation -> NOISE_STD for settings.py; signal vs noise in pk_obs
  Antennaedebug.ipynb    exploratory: self-contained copy of the simulator, gain /
                         switched-off-antenna effects
  codetest.ipynb         walkthrough: data/ rebuilt step by step with plain torch (no src/ imports),
                         bit-identical to gen_mock_data.py, with a plot and comment per stage

figures/                 figures of scripts/ and notebooks/ (pipeline_plots/, calibrator, S_ref)
tests/                   unittest suite, see [Tests](#tests)
output/run_04/           the falcon run: config.yml, graph/ (trained networks, logs), samples/
                         (posterior and training-buffer samples), wandb/, the analysis plots
```

---

## How the pipeline works

### 1. The simulator (`config.yml` + `src/model.py`)

```
z ──► gains ─────────────────────────────────────────────────► V_ref            [observed]
            └──► stacked_uv ──┐                                  (G S_ref G† + N)
                              ├──► tracked_uv ──► normed_tracked_uv ──┐
pw ──► df ──► bt ──► image ──► uvimage ──┘                            ├──► x ──► obsx ──► pk_obs [observed]
       └──► pk_df              └──► noiseimage ──────────────────────┘
```

| Node | Class | What it is |
|---|---|---|
| gains | `GainsFromZ` | station gains: g_0 = 1, 1 + z for the 106 unknown, 1 for the 200 known |
| stacked_uv | `UVTracksFromGains` | for each of the 2 UV subsets: Σ over baselines and track of t · g_i g_j* per UV pixel |
| df / bt / image | `DensityFromPw` / `BrightnessFromDensity` / `SkyImageFromBrightness` | δ from the power spectrum, then T_b = 26.60 (1 + δ) mK |
| uvimage | `UVSkyFromImage` | Fourier transform of the sky, phase centre at the beam centre |
| tracked_uv / normed_tracked_uv | `TrackedUVFromSky` / `NormalisedUV` | the sampled sky, divided by the integrations per pixel |
| noiseimage | `ThermalNoiseUV` | thermal noise, NOISE_STD / √n per pixel, Hermitian |
| x / obsx | `VisibilitiesWithNoise` / `DirtyImageFromUV` | observed UV maps; their real dirty images [mK] |
| pk_obs | `PkFromDirtyImage` | P(k) of each subset's dirty image — **Stage-2 data** |
| V_ref | `ReferenceVisFromGains` | g_i S_ref[i,j] g_j* + noise — **Stage-1 data** |
| pk_df | `PkFromDensity` | P(k) of δ (diagnostic only) |

The instrument appears in two equivalent forms:

* **UV-map path (science).** Each baseline's track is gridded with its gain product; a pixel's
  value is the mean over all integrations that land in it. Many baselines share a pixel, so the
  science data mix the gains of many stations.
* **Dense path (calibration).** `SkyMatrixGenerator` gives one value per baseline,
  S[i, j] = mean of the sky over that baseline's track. Baseline (j, i) is tracked on its own,
  along the mirrored track at −u (as the UV maps grid it); for a real sky S comes out Hermitian to
  FFT round-off without being forced to. The noise of (j, i) is the conjugate of (i, j)'s: it is the
  same correlator product. The gains are constant over the 4 h, so V[i, j] = g_i S[i, j] g_j* exactly. `S_ref` is the calibrator through this path; with
  S_ref known, the first row V_ref[0, j] = S_ref[0, j] conj(g_j) fixes every unknown gain.

20 246 of the 46 971 baselines are longer than the box resolves (≳ 2.5 km) and never land on the
grid; 26 725 are observed, covering 76.5 % of the 128² UV cells.

### 2. What falcon learns (`config.yml`)

| Stage | Posterior | Data | Estimator | Embedding |
|---|---|---|---|---|
| 1 — calibration | q(z \| V_ref) | calibrator visibilities | Gaussian | `VRefFirstRowEmbedding`: [Re, Im] of the 230 observed first-row baselines |
| 2 — science | q(pw \| pk_obs, V_ref) | P(k) of the dirty image + V_ref | normalising flow (MAF) | `PkVRefEmbedding`: MLP(log10 pk_obs) + [log\|V\|, cos, sin] of the V_ref first row |

Both are trained together on simulations drawn from the priors in `config.yml` (uniform; pw
ranges in the comments there, z ∈ [−1, 1]) and then sampled at the mock observation of `data/`.
Stage 2 sees V_ref so that it can account for the gains without being given them.

### 3. The mock observation (`scripts/gen_mock_data.py`)

The truth lives in `src/settings.py`: `TRUE_PW = (1.0, −0.5, −2.0, −0.8)` (pixel rms σ_δ = 0.31,
σ_T = 8.2 mK, k_break = 0.16 Mpc⁻¹), the density-field
seed `TRUE_FIELD_SEED = 54`, and gain perturbations z ~ N(0, 0.3²) (seed 99). The script runs the
same graph nodes falcon trains on and writes the observed `pk_obs.npy`, `V_ref_obs.npy`, the
intermediate products (`x_obs`, `obsx_obs`, `image`, `S_ref`, `S_obs`, `V_dense_obs`) and the truth
(`true_*.npy`).

### 4. The independent check (`fisher/`)

`fisher_forecast.py` draws 5000 sky realisations at the true parameters, pushes them through the
same instrument with the **true, fixed gains**, and builds the covariance of the band powers of
both UV subsets (concatenated, the `pk_obs` Stage 2 sees) and their gradients (finite differences
with common random numbers) — the Gaussian Cramér–Rao bound on pw from the same data.
If the gains are recoverable (noise off), the SBI posterior widths should match it. At the current
truth it gives σ = 0.057 (log10 A), 0.143 (n1), 0.034 (n2), 0.031 (log10 k_break), data only.

The pw priors (`config.yml`) are sized from this forecast and from the field's pixel rms
σ_δ(pw) = (Σ_k P(k) / (dx dy N²))^½, which is exact and needs no simulation: the truth is ≥ 5.3 σ from
every edge, each prior ≥ 14 σ wide, and every prior draw has σ_δ < 1 (worst corner 0.96, 2 % of draws
above 0.5), so δ stays physical (δ ≥ −1 for almost every pixel); σ_T spans 0.6–10.7 mK (5–95 %).

---

## Results so far (`run_04`)

> `run_04` was trained at the **previous truth** (−1.8, −2.0, −3.0, −0.3) with the previous priors,
> tracks and T_b; the numbers below describe that set-up, not the current one, and
> `tests/test_scripts.py` (`TestRun04`) fails until a run of the current `config.yml` replaces it.

**Power spectrum (Stage 2)** — `fisher/fisher_analysis.ipynb`, `notebooks/plot_posteriors.ipynb`:

| parameter | truth | SBI median ± 1σ | Fisher σ | Fisher / SBI | (median − truth) / σ |
|---|---|---|---|---|---|
| log10 A | −1.800 | −1.862 ± 0.049 | 0.051 | 1.04 | −1.26 |
| n1 | −2.000 | −2.029 ± 0.040 | 0.042 | 1.05 | −0.72 |
| n2 | −3.000 | −3.108 ± 0.119 | 0.102 | 0.86 | −0.91 |
| log10 k_break | −0.300 | −0.274 ± 0.020 | 0.021 | 1.05 | +1.32 |

The Fisher column is the earlier forecast (subset-averaged band powers, old tracks and T_b); the
current `fisher/fisher_forecast.py` uses both subsets' band powers and must be compared with a
retrained run. The SBI errors match that fixed-gain Fisher forecast within ~15 %, and the medians lie within
1.3σ of the truth (one sky realisation, so ~1σ offsets are expected): marginalising over the
212 gain parameters costs little when the noise is off.

**Gains (Stage 1)** — the posterior medians track the true gains well (correlation 0.98, rms
error 0.075), but the widths are **~2× too wide**: the truth falls inside the 68 % interval for
99 % of the 212 parameters. With noise off the gains are exactly recoverable from V_ref (the
closed form and `imageanalysis.ipynb` recover them to 10⁻⁸), so this is a training limit, not
information: do not quote these widths as calibration precision. (Computed in
`plot_posteriors.ipynb`, "Gain posterior vs truth".)

**Imaging** — `imageanalysis.ipynb`: with the true gains or with self-calibrated gains the
reconstruction error is 0.132 (limited by the 23.5 % unsampled UV cells); ignoring the gains, 0.142.

---

## Quick start

All commands run from the project root. The environment used for `run_04` is the conda env
`newfalcongaincal` (Python 3.12, torch 2.5.1 + CUDA 12.4, falcon 0.4.1, numpy, scipy, matplotlib,
pyyaml, corner, pandas, jupyter / nbclient).

```bash
# check everything (~15 s; GPU if available, CUDA_VISIBLE_DEVICES="" forces the CPU)
python -m unittest discover -s tests -v

# figures of every simulator stage, with value checks
python scripts/plot_pipeline.py
python scripts/plot_reference.py                  # the calibrator as the instrument sees it

# train and sample (needs GPUs: config.yml asks ray for 0.9 GPU per estimator)
falcon launch                                     # new run directory under output/
falcon graph                                      # print the graph only

# Fisher forecast (~1 min) -> fisher/fisher_results.npz, then fisher/fisher_analysis.ipynb
python fisher/fisher_forecast.py
```

Notebooks find the project root themselves (the first parent folder holding `config.yml`), so
they run from their own folder or from the root.

### Typical workflows

* **Change the set-up** (grid, noise, antennas, calibrator, truth): edit `src/settings.py`; keep
  the priors in `config.yml` consistent (the `z` list has 2 × unknown-antenna entries, the truth
  must lie inside the priors — the tests check both); run `python scripts/gen_mock_data.py`; run
  the tests; train a new run. A different array, track, integration time, z or grid needs a new
  tracks file (`settings.PADDED_PATH`).
* **Analyse a run**: set `RUN_NAME` in `notebooks/plot_posteriors.ipynb` (it refuses a run whose
  graph / buffer / sampling settings differ from `config.yml`), then `fisher/fisher_analysis.ipynb`.
* **Do not regenerate `data/` for an existing run**: a run conditions on the `data/` it was trained
  with.

### Full re-run: data → Fisher → training → analysis

1. **Settings** (`src/settings.py`). With noise on (`NOISE_STD`, `NOISE_UNIT`, `N_DAYS`, see
   [Thermal noise](#thermal-noise)), also raise `CALIBRATOR_FLUX`: at 1000 h the noise on one V_ref
   baseline is ~200 mK against a 0.02 mK calibrator signal, so Stage 1 would learn nothing (a 1 Jy
   calibrator is ≈ 8.9 × 10⁶ mK). The 21-cm signal is then below the noise in `pk_obs` except the
   lowest k-bin.
2. **Mock observation, on the CPU** (seeded draws differ between CPU and GPU, so the GPU would give
   another true field), then the tests:
   ```bash
   CUDA_VISIBLE_DEVICES="" python scripts/gen_mock_data.py
   python -m unittest discover -s tests -v
   ```
3. **Fisher forecast** (~1 min; fixed true gains). With noise on, first set
   `FISHER = "both_numeric"` in `fisher/fisher_config.py` (`"both_analytic"` assumes cosmic
   variance only):
   ```bash
   python fisher/fisher_forecast.py              # -> fisher/fisher_results.npz
   ```
4. **Train** on a GPU node. Optionally rename `logging.wandb.group` in `config.yml` first. Training
   on the GPU with CPU-made `data/` is fine (the observation is a fixed file):
   ```bash
   falcon graph                                  # cheap check that every node builds
   falcon launch                                 # -> new output/run_XX
   ```
5. **Analyse**: set `RUN_NAME` to the new run in `notebooks/plot_posteriors.ipynb` and
   `fisher/fisher_analysis.ipynb` and run them; point `RUN` in `tests/test_scripts.py` at the new run;
   run the slow tests (`RUN_SLOW=1 ...`, see [Tests](#tests)); update
   [Results so far](#results-so-far-run_04) and the Status line at the top.

---

## Conventions

* Arrays are `array[..., y, x]` (row = y down, column = x right).
* **Images** have the beam / phase centre at pixel (64, 64), as displayed.
* **UV arrays** are in FFT order: DC at [0, 0], negative frequencies wrapped — the order of the
  tracks file and `fftfreq`. `np.fft.fftshift` is used only for display.
* `func.image_to_uv` = `fft2(ifftshift(·))` and `func.uv_to_image` = `fftshift(ifft2(·))` (both
  orthonormal) are the only image ↔ UV transforms. A source at the beam centre has zero phase on
  every baseline; `uv_to_image(image_to_uv(img)) == img`.
* UV maps are Hermitian (each baseline also gridded as its conjugate at (−u, −v)), so dirty images
  are real; the DC pixel is dropped, so they have zero mean.
* P(k): `PowerSpectrumMeasurer` returns |δ̃_ortho|² dx dy and `DensityField2D` draws modes with
  variance P / (dx dy), so the measured P(k) equals the analytic one.
* Gains are stored interleaved `[Re g_0, Im g_0, Re g_1, …]`; baseline (i, j), i < j, carries
  g_i conj(g_j) in `torch.triu_indices` order (the order of the tracks file and the layout file).

## Thermal noise

Three settings in `src/settings.py`:

| Setting | Meaning |
|---|---|
| `NOISE_STD` | std of the complex noise on ONE visibility: one baseline, one 30 s integration (E\|n\|² = NOISE_STD²; Re and Im each NOISE_STD / √2) |
| `NOISE_UNIT` | `"mK"` (pipeline units, default) or `"Jy"` (physical; converted to mK once in `src/model.py`) |
| `N_DAYS` | days the 4 h track is observed (same UV coverage every sidereal day); `T_OBS_H = 4 h × N_DAYS`, e.g. 250 days = 1000 h |

`src/model.py` turns them into `model.NOISE_STD = NOISE_STD [mK] / √N_DAYS` (one 30 s integration
of the whole observation), and the noise nodes divide further by √n, n = integrations of ONE track:
per baseline for V_ref, per UV pixel for the science maps. The noise is added after the gains
(gain-independent) and is Hermitian like the data: the (j, i) entry of V_ref is the conjugate of (i, j),
the UV pixel at −u the conjugate of +u. The model assumes the same gains on every day.
`ThermalNoiseUV(noise_std=...)` / `ReferenceVisFromGains(noise_std=...)` override the level for one
node (mK; `plot_pipeline.py` uses 0.0).

**Units.** The pipeline works in mK: images are brightness temperatures and visibilities are their
ortho-FFT. `NOISE_UNIT = "mK"` (default) means `NOISE_STD` is already in these units;
`NOISE_UNIT = "Jy"` means it is a physical visibility noise in Jy, which `src/model.py` converts
once to mK before it is added. The conversion is `functionlistnew.jansky_to_mK` / `mK_to_jansky`
(any tensor, array or number; `domain="image"` for Jy per pixel, `"uv"` for Jy per visibility —
they differ by N = 128). It uses the project cosmology (Planck 2018, D_c(9) = 9401 Mpc): 1 Jy per pixel = 8.91 × 10⁶ mK,
1 Jy per visibility = 6.96 × 10⁴ mK.

**Cosmology and the tracks file.** The tracks were made with tools21cm 2.4.3,
`get_full_uv_map_with_antenna_stamp(128, 9, "AA*", 4, 30, 512)`, which grids each baseline at UV cell
`round(u_λ × box / z_to_cdist(z))`, with tools21cm's constants (`const.py`) set to the project's:
Planck 2018, exact c and ν21, so `z_to_cdist(9)` = 9400.70 Mpc = `functionlistnew.comoving_distance(9)`.
Neither that tools21cm copy nor the conversion notebook is kept in this folder; regenerating the tracks
needs tools21cm 2.4.3 with those constants. Recomputing the tracks from `src/skalow_AAstar_layout.txt`
with this recipe reproduced `src/paddedAAstar_4_30_9_128_512.pth` bit for bit (checked 2026-10-14). The
earlier file, made with tools21cm's defaults (h = 0.7, Ω_m = 0.27, c = 3.0 × 10⁵ km/s, ν21 = 1420 MHz,
D_c(9) = 9566.75 Mpc), had every baseline 1.8 % too close to the centre; `output/run_04` was trained with it.
Other properties of the tracks: hour angle −2 h … +2 h about transit (15°/h), declination −30°
(tools21cm's default), baseline (i, j) = (r_i − r_j) / λ in `itertools.combinations` order
(= `torch.triu_indices`), nearest-cell gridding, negative indices wrapped (FFT order).

It is **0** at present. `notebooks/noise.ipynb` derives the physical value from the radiometer
equation (Planck 2018, as everywhere): **6.9 × 10⁴ mK** (0.993 Jy) for one 30 s integration;
with `N_DAYS = 250` (1000 h) the effective value is 4.4 × 10³ mK. At that level the 21-cm signal is far below the noise in `pk_obs` except the lowest k-bin,
and the 2.7 mK calibrator (≈ 0.3 µJy) is far too faint: a 1 Jy calibrator is
`CALIBRATOR_FLUX` ≈ 8.9 × 10⁶ mK. Switching the noise on therefore also needs a brighter calibrator.

## Devices (CPU / GPU)

Everything picks the GPU if there is one and the CPU otherwise — `src/` (falcon), the scripts,
the Fisher forecast and the notebooks. Results agree between devices to float rounding, with one
exception: torch draws **different random numbers on CPU and GPU for the same seed**. So

* the true field of the mock observation depends on where `scripts/gen_mock_data.py` ran; it
  records this in `data/mock_device.npy` (the current `data/` was made on the CPU). Checks that
  regenerate the truth (`scripts/plot_pipeline.py`, `notebooks/plot_posteriors.ipynb`,
  `notebooks/Antennaedebug.ipynb`, `tests/test_devices.py`) name both devices;
  `plot_posteriors` draws truth-seed fields on the device of `data/`;
* Monte-Carlo results (the Fisher forecast, noise realisations) differ between devices at the
  Monte-Carlo level: a CPU re-run of `fisher_forecast.py` gives sigmas within ~3 % of the stored
  bundle, which was not made on this CPU (most likely on a GPU).

`tests/test_devices.py` checks that the simulator reproduces `data/` from the stored truth on
whatever device it runs, and, on a GPU node, that CPU and GPU agree component by component. If the
devices disagree by more than float rounding, it is a bug.

---

## Tests

```bash
python -m unittest discover -s tests -v                                  # default, ~15 s
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v          # force the CPU
RUN_SLOW=1 python -m unittest discover -s tests -p "test_slow.py" -v     # + notebooks, Fisher (~2 min)
```

Nothing in the project is written by the tests (scripts and notebooks run with temporary output).

| File | What it checks |
|---|---|
| `test_pipeline.py` | the simulator: config (targets, parents, priors vs truth), FFT conventions (phase centre, round trip, FFT-ordered tracks), sky model (Hermitian modes, measured P(k) = analytic), UV path (exact sampling with unit gains, Hermitian x, real DC-free dirty images, PSF at the beam centre), noise (Hermitian, std ∝ 1/√n, zero where unobserved), reference path (S_ref flat, V_ref = g S g*, gains recoverable from the first row), embeddings, and that `data/*.npy` come from the current code |
| `test_devices.py` | the simulator reproduces `data/` from the stored truth on the current device; the seeded truth is reproduced on the device that made `data/`; on a GPU node, CPU and GPU agree for every deterministic component |
| `test_scripts.py` | `gen_mock_data.py` reproduces `data/` exactly; `plot_pipeline.py` passes all its checks; `plot_reference.py` sees the calibrator on every observed baseline; `output/run_04` used the current `config.yml` and has complete posterior samples inside the priors |
| `test_fisher.py` | the Fisher maths on a synthetic case with a known answer (Hartlap debiasing over many small surveys, Percival m1 against a hand-computed value, prior, covariance terms, pseudo-inverse fallback); the knobs vs `config.yml`; the stored bundle is complete, records its knobs, has converged linear and log gradients (d ln P / d log10 A = ln 10) and is reproduced exactly by the shared assembly code |
| `test_slow.py` (opt-in) | every notebook runs top to bottom; `fisher_forecast.py` runs and agrees with the stored bundle to Monte-Carlo precision |

---

## Notes and known limitations

* **First-row embedding.** Stage 1 uses only the first row of V_ref (230 observed baselines). With
  `KNOWN_ANT = 200` every unknown station has such a baseline. If you lower `KNOWN_ANT`, check this
  again (with 100 known, stations 201–206 had none and stayed at the prior);
  `VRefUpperTriEmbedding` uses all 26 725 observed baselines.
* **Outer stations.** Stations with index ≥ 200 have a median of 2 % of their baselines observed;
  49 % of the baselines involving a known-gain station are unobserved.
* **Gain errors only partly average out in the UV maps.** Each UV pixel holds the
  integration-weighted mean of g_i g_j* over the baselines that land in it, so zero-mean errors
  partly cancel — well in the dense centre, poorly at the coverage edge, where few baselines
  contribute and the mean of |g|² (1.25 for the true gains) exceeds |mean g|² (1.02). For the mock
  observation this biases pk_obs of subset 0 (baselines of stations 0–89, almost all touching an
  unknown station) by +8–30 % at k ≈ 0.19–0.30 Mpc⁻¹ and up to ×2.3 near k ≈ 0.74, and subset 1 by
  0 to −10 % (`notebooks/codetest.ipynb`, step 11). The bias is scale-dependent and degenerate with the
  P(k) shape; Stage 2 sees `V_ref` so that it can undo it.
* **2D, single channel.** The sky is one 2D slice (one 227 kHz channel at 4 Mpc depth); a real
  power-spectrum measurement averages many channels.
* **Fisher forecast with noise.** `fisher_forecast.py` adds the same thermal noise as the SBI
  pipeline; the `both_analytic` covariance derivative assumes cosmic variance only (and is
  identically zero in log space), so use `FISHER = "both_numeric"` when the noise matters.
