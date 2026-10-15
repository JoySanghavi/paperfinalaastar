"""
Do the answers match on CPU and GPU? Run from the project root:

    python -m unittest discover -s tests -v      # on a GPU node: also the CPU-vs-GPU class

  TestForwardModelReproducesData   always runs, on model.device (the GPU if there is one): the
      simulator, started from the STORED truth (data/true_delta.npy, data/true_z.npy, no random
      draws), reproduces the mock observation in data/ — to float rounding on another device.
  TestSeededTruth                  the seeded true field is reproduced on the device that made data/
      (data/mock_device.npy). Seeded draws are expected to differ between CPU and GPU.
  TestCPUvsGPU                     skipped without a GPU: the deterministic parts of the simulator
      (FFTs, UV gridding, dense path, P(k), power spectrum) agree between CPU and GPU.

A failure here on one device but not the other points to a CPU/GPU issue.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import functionlistnew as func      # noqa: E402
import model                        # noqa: E402
import settings                     # noqa: E402

DATA = settings.DATA_DIR
RUN_DEVICE = torch.device(model.device).type


def mock_device():
    path = DATA / "mock_device.npy"
    return str(np.load(path)) if path.exists() else None


def assert_close(test, new, old, rel, what):
    """max |new - old| <= rel * max |old| (float32 rounding across devices)."""
    err = np.abs(np.asarray(new) - np.asarray(old)).max()
    test.assertLessEqual(err, rel * np.abs(old).max(), f"{what}: max diff {err:.2e} (device {RUN_DEVICE})")


class TestForwardModelReproducesData(unittest.TestCase):
    """The pipeline nodes of src/model.py, from the stored truth, give data/ back."""

    @classmethod
    def setUpClass(cls):
        cls.delta = np.load(DATA / "true_delta.npy")
        cls.gains = model.GainsFromZ().simulate_batch(1, np.load(DATA / "true_z.npy"))
        bt = model.BrightnessFromDensity().simulate_batch(1, cls.delta)
        cls.image = model.SkyImageFromBrightness().simulate_batch(1, bt)
        cls.uvimage = model.UVSkyFromImage().simulate_batch(1, cls.image)
        stacked = model.UVTracksFromGains().simulate_batch(1, cls.gains)
        cls.x = model.NormalisedUV().simulate_batch(1, model.TrackedUVFromSky().simulate_batch(1, cls.uvimage, stacked))
        cls.obsx = model.DirtyImageFromUV().simulate_batch(1, cls.x)
        cls.pk_obs = model.PkFromDirtyImage().simulate_batch(1, cls.obsx)
        old = model.NOISE_STD
        model.NOISE_STD = 0.0
        try:
            cls.V_ref = model.ReferenceVisFromGains().simulate_batch(1, cls.gains)
        finally:
            model.NOISE_STD = old

    def setUp(self):
        if settings.NOISE_STD != 0:
            self.skipTest("data/ contains a noise realisation (NOISE_STD > 0)")

    def test_sky(self):
        assert_close(self, self.image[0], np.load(DATA / "image.npy"), 1e-6, "image")

    def test_uv_maps(self):
        assert_close(self, self.x[0], np.load(DATA / "x_obs.npy"), 1e-5, "x_obs")

    def test_dirty_images(self):
        assert_close(self, self.obsx[0], np.load(DATA / "obsx_obs.npy"), 1e-5, "obsx_obs")

    def test_pk_obs(self):
        new = func.drop_roundoff_bins(torch.tensor(self.pk_obs[0])).numpy()
        old = func.drop_roundoff_bins(torch.tensor(np.load(DATA / "pk_obs.npy"))).numpy()
        data = np.isfinite(new)                                   # bins with sampled UV modes
        np.testing.assert_array_equal(data, np.isfinite(old), err_msg="bins with data differ")
        self.assertTrue(data.any(), "pk_obs has no bin with data")
        np.testing.assert_allclose(new[data], old[data], rtol=1e-4, err_msg=f"pk_obs on {RUN_DEVICE}")

    def test_reference_visibilities(self):
        assert_close(self, self.V_ref[0], np.load(DATA / "V_ref_obs.npy"), 1e-5, "V_ref_obs")

    def test_dense_science_path(self):
        S = model.SkyMatrixFromUVSky().simulate_batch(1, self.uvimage)
        assert_close(self, S[0], np.load(DATA / "S_obs.npy"), 1e-5, "S_obs")


class TestSeededTruth(unittest.TestCase):
    def test_provenance_recorded(self):
        dev = mock_device()
        self.assertIn(dev, ("cpu", "cuda"), "data/mock_device.npy missing: run python scripts/gen_mock_data.py")

    def test_truth_regenerates_on_its_device(self):
        dev = mock_device()
        if dev is None or (dev == "cuda" and not torch.cuda.is_available()):
            self.skipTest(f"data/ made on {dev}; that device is not available here")
        cfg = func.SimulationConfig(device=dev)
        lA, n1, n2, lkb = settings.TRUE_PW
        A, n1, n2, kb = (torch.tensor([v], device=dev) for v in (10 ** lA, n1, n2, 10 ** lkb))
        delta = func.DensityField2D(cfg).sample_field(A, n1, n2, kb, seed=settings.TRUE_FIELD_SEED)
        np.testing.assert_allclose(delta.cpu().numpy(), np.load(DATA / "true_delta.npy"), rtol=0, atol=1e-6,
                                   err_msg=f"seeded truth on {dev} != data/true_delta.npy")


@unittest.skipUnless(torch.cuda.is_available(), "no GPU: CPU-vs-GPU comparison skipped")
class TestCPUvsGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        padded = torch.load(settings.PADDED_PATH, map_location="cpu", weights_only=True)
        nant = func.nant_from_padded(padded)
        subsets = func.split_indices(padded.shape[0], settings.N_SUBSETS)
        cls.cfg = {dev: func.SimulationConfig(nant=nant, padded=padded.to(dev), subsetindex=subsets, device=dev)
                   for dev in ("cpu", "cuda")}
        cls.gains = func.theta_to_gains(torch.tensor(np.load(DATA / "true_z.npy")), settings.KNOWN_ANT)
        gen = torch.Generator().manual_seed(0)
        cls.images = torch.randn(3, settings.N_PIX, settings.N_PIX, generator=gen)

    def both(self, f):
        """f(device) on CPU and GPU -> (cpu numpy, gpu numpy)."""
        return tuple(f(dev).detach().cpu().numpy() for dev in ("cpu", "cuda"))

    def test_image_uv_transforms(self):
        a, b = self.both(lambda d: func.image_to_uv(self.images.to(d)))
        assert_close(self, b, a, 1e-5, "image_to_uv")
        a, b = self.both(lambda d: func.uv_to_image(func.image_to_uv(self.images.to(d))).real)
        assert_close(self, b, a, 1e-5, "uv_to_image")

    def test_unit_counts_identical(self):
        a, b = self.both(lambda d: func.UnitUVMapGenerator(self.cfg[d])().real)
        np.testing.assert_array_equal(a, b)

    def test_gain_weighted_uv_maps(self):
        a, b = self.both(lambda d: func.UVMapGenerator(self.cfg[d])(self.gains.to(d)))
        assert_close(self, b, a, 1e-5, "UVMapGenerator")

    def test_dense_sky_matrix(self):
        def S(d):
            cfg = self.cfg[d]
            return func.SkyMatrixGenerator(cfg.padded, cfg.shape, cfg.nant)(func.image_to_uv(self.images[:1].to(d)))
        a, b = self.both(S)
        assert_close(self, b, a, 1e-5, "SkyMatrixGenerator")

    def test_power_spectrum_measurer(self):
        def pk(d):
            return func.PowerSpectrumMeasurer((settings.N_PIX,) * 2, settings.BOX_SIZE, device=d)(self.images.to(d))[1]
        a, b = self.both(pk)
        np.testing.assert_allclose(b, a, rtol=1e-4)

    def test_power_spectrum_model(self):
        lA, n1, n2, lkb = settings.TRUE_PW
        def P(d):
            k = func.make_kgrid((settings.N_PIX,) * 2, settings.BOX_SIZE, d, rfft=True)
            A, a1, a2, kb = (torch.tensor([v], device=d) for v in (10 ** lA, n1, n2, 10 ** lkb))
            return func.power_spectrum_2d(k, A, a1, a2, kb)
        a, b = self.both(P)
        np.testing.assert_allclose(b, a, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
