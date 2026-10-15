"""
Checks of scripts/ and of the data/ and output/ they belong to (~30 s). Run from the project root:

    python -m unittest discover -s tests -v

The scripts are run in this process with data/ and figures/ redirected to a temporary directory,
so nothing in the project is overwritten.

  TestMockData      scripts/gen_mock_data.py reproduces data/ exactly (on the device data/ was made on)
  TestPlotScripts   scripts/plot_pipeline.py: every value check passes;
                    scripts/plot_reference.py: the calibrator is seen on every observed baseline
  TestRun04         output/run_04 is a run of the current config.yml, with complete posterior samples
"""

import contextlib
import io
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import matplotlib                   # noqa: E402
matplotlib.use("Agg")
import model                        # noqa: E402
import settings                     # noqa: E402

DATA = settings.DATA_DIR
RUN = ROOT / "output" / "run_04"


@contextlib.contextmanager
def redirected(data_dir=None, figures_dir=None):
    """Point settings.DATA_DIR / FIGURES_DIR elsewhere; restore them (and model.NOISE_STD) afterwards."""
    old = settings.DATA_DIR, settings.FIGURES_DIR, model.NOISE_STD
    settings.DATA_DIR = Path(data_dir) if data_dir else settings.DATA_DIR
    settings.FIGURES_DIR = Path(figures_dir) if figures_dir else settings.FIGURES_DIR
    try:
        yield
    finally:
        settings.DATA_DIR, settings.FIGURES_DIR, model.NOISE_STD = old


def run_script(name, *args):
    """Run scripts/<name> as __main__ in this process; return what it printed."""
    path = ROOT / "scripts" / name
    argv, sys.argv = sys.argv, [str(path), *map(str, args)]
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = argv
    return out.getvalue()


def same(a, b):
    if a.dtype.kind in "US":
        return np.array_equal(a, b)
    return np.allclose(a, b, rtol=1e-6, atol=0, equal_nan=True)


class TestMockData(unittest.TestCase):
    def test_gen_mock_data_reproduces_data(self):
        dev = str(np.load(DATA / "mock_device.npy")) if (DATA / "mock_device.npy").exists() else None
        if dev != torch.device(model.device).type:
            self.skipTest(f"data/ made on {dev}, this run on {model.device}: the seeded truth differs "
                          f"between CPU and GPU (see tests/test_devices.py)")
        with tempfile.TemporaryDirectory() as tmp, redirected(data_dir=tmp):
            run_script("gen_mock_data.py")
            made = sorted(Path(tmp).glob("*.npy"))
            self.assertEqual({f.name for f in made}, {f.name for f in DATA.glob("*.npy")})
            for f in made:
                with self.subTest(file=f.name):
                    self.assertTrue(same(np.load(f), np.load(DATA / f.name)), f"{f.name} differs from data/")

    def test_observed_files_match_config(self):
        graph = yaml.safe_load(settings.CONFIG_PATH.read_text())["graph"]
        shapes = {"pk_obs": (settings.N_SUBSETS, 50), "V_ref": (model.NANT, model.NANT)}
        for node, shape in shapes.items():
            with self.subTest(node=node):
                self.assertEqual(np.load(ROOT / graph[node]["observed"]).shape, shape)


class TestPlotScripts(unittest.TestCase):
    def test_plot_pipeline_checks_pass(self):
        with tempfile.TemporaryDirectory() as tmp, redirected(figures_dir=tmp):
            out = run_script("plot_pipeline.py")
            figs = sorted(p.name for p in Path(tmp, "pipeline_plots").glob("*.png"))
        self.assertNotIn("[FAIL]", out, out)
        self.assertGreaterEqual(out.count("[PASS]"), 15, out)
        self.assertEqual(figs, ["01_science_sky.png", "02_uv_and_gains.png", "03_observed_sky.png", "04_reference.png"])

    def test_plot_reference_summary(self):
        import plot_reference
        s = plot_reference.summarize(plot_reference.reference_view())
        counts = model.sky_matrix.counts.cpu().numpy()
        first_row = model.sky_matrix.i_idx.cpu().numpy() == 0
        self.assertEqual(s["observed baselines"], int((counts > 0).sum()))
        usable = [v for k, v in s.items() if k.startswith("usable baselines")][0]
        self.assertEqual(usable, s["observed baselines"])            # a 0.3 px calibrator: flat |FFT|
        usable_row = [v for k, v in s.items() if k.startswith("usable first-row")][0]
        self.assertEqual(usable_row, int((counts[first_row] > 0).sum()))
        self.assertAlmostEqual(s["sky peak"], settings.CALIBRATOR_FLUX, places=5)


@unittest.skipUnless(RUN.exists(), "output/run_04 not present")
class TestRun04(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = sorted((RUN / "samples" / "posterior").glob("*.npz"))

    def test_run_used_current_config(self):
        run_cfg = yaml.safe_load((RUN / "config.yml").read_text())
        cur_cfg = yaml.safe_load(settings.CONFIG_PATH.read_text())
        for key in ("graph", "buffer", "sample"):
            with self.subTest(section=key):
                self.assertEqual(run_cfg[key], cur_cfg[key], f"output/run_04 '{key}' differs from config.yml")

    def test_posterior_samples_complete(self):
        n = yaml.safe_load(settings.CONFIG_PATH.read_text())["sample"]["posterior"]["n"]
        self.assertEqual(len(self.files), n)
        with np.load(self.files[0]) as d:
            self.assertEqual(d["pw"].shape, (len(settings.PW_NAMES),))
            self.assertEqual(d["z"].shape, (2 * model.N_UNKNOWN,))

    def test_posterior_samples_inside_priors(self):
        graph = yaml.safe_load(settings.CONFIG_PATH.read_text())["graph"]
        for node in ("pw", "z"):
            lo, hi = np.array([p[1:] for p in graph[node]["simulator"]["priors"]], dtype=float).T
            s = np.stack([np.load(f)[node] for f in self.files])
            with self.subTest(node=node):
                self.assertTrue(np.all((s >= lo) & (s <= hi)))


if __name__ == "__main__":
    unittest.main()
