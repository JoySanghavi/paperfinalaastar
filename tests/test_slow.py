"""
Slow end-to-end checks (~2 min on a CPU; opt-in). Run from the project root:

    RUN_SLOW=1 python -m unittest discover -s tests -p "test_slow.py" -v

Nothing in the project is written: notebooks are executed in memory (their saved outputs are not
touched) and the Fisher forecast writes to a temporary file.

  TestNotebooks      every notebook runs top to bottom without an error (plot_posteriors: its set-up
                     cells, including the truth check — the full notebook rewrites the PDFs / GIFs of
                     output/run_04)
  TestFisherForecast fisher/fisher_forecast.py runs and agrees with fisher/fisher_results.npz to
                     Monte-Carlo precision (the stored bundle may come from the other device)
"""

import contextlib
import io
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SLOW = os.environ.get("RUN_SLOW") == "1"
SKIP = "slow: set RUN_SLOW=1"


def execute_notebook(path, keep_cells=None):
    """Execute a notebook in memory, in its own folder (as Jupyter does); raise on a cell error."""
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    if keep_cells is not None:
        nb.cells = [nb.cells[i] for i in keep_cells]
    NotebookClient(nb, timeout=3600, resources={"metadata": {"path": str(path.parent)}}).execute()
    return nb


@unittest.skipUnless(SLOW, SKIP)
class TestNotebooks(unittest.TestCase):
    def run_nb(self, rel, keep_cells=None):
        nb = execute_notebook(ROOT / rel, keep_cells)
        errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
        self.assertEqual(errors, [], rel)

    def test_imageanalysis(self):
        self.run_nb("notebooks/imageanalysis.ipynb")

    def test_reference_calibrator(self):
        self.run_nb("notebooks/reference_calibrator.ipynb")

    def test_plot_posteriors_setup(self):
        # cells 0, 1, 7, 9: run selection, config guard + posterior samples, gain posterior vs
        # truth, set-up + truth check
        self.run_nb("notebooks/plot_posteriors.ipynb", keep_cells=[0, 1, 7, 9])

    def test_fisher_analysis(self):
        self.run_nb("fisher/fisher_analysis.ipynb")

    def test_noise(self):
        self.run_nb("notebooks/noise.ipynb")

    def test_antennaedebug(self):
        self.run_nb("notebooks/Antennaedebug.ipynb")

    def test_codetest(self):
        # asserts that its independent numpy rebuild of data/ (and of the UV tracks from the
        # station layout) agrees with gen_mock_data.py to float32 precision
        self.run_nb("notebooks/codetest.ipynb")


@unittest.skipUnless(SLOW, SKIP)
class TestFisherForecast(unittest.TestCase):
    def test_forecast_reproduces_stored_bundle(self):
        stored_path = ROOT / "fisher" / "fisher_results.npz"
        if not stored_path.exists():
            self.skipTest("no stored fisher_results.npz to compare with")
        script = ROOT / "fisher" / "fisher_forecast.py"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fisher_results.npz"
            argv, sys.argv = sys.argv, [str(script), str(out)]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(script), run_name="__main__")
            finally:
                sys.argv = argv
            with np.load(out) as new, np.load(stored_path) as old:
                np.testing.assert_array_equal(new["valid_mask"], old["valid_mask"])
                np.testing.assert_allclose(new["true_pw"], old["true_pw"])
                # Monte-Carlo estimate: a few per cent between runs on different devices (a CPU
                # re-run vs the stored bundle: sigma within 2.9 %, pk_model_fid within 2.3 %)
                np.testing.assert_allclose(new["fisher_sigma"], old["fisher_sigma"], rtol=0.08)
                np.testing.assert_allclose(new["pk_model_fid"], old["pk_model_fid"], rtol=0.06)
                self.assertTrue(np.all(np.linalg.eigvalsh(new["fisher_matrix"]) > 0))


if __name__ == "__main__":
    unittest.main()
