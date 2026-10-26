"""
Embedding networks for the two-stage inference (config.yml `embedding._target_`).

  Stage 1  q(z  | V_ref)
      VRefFirstRowEmbedding   [Re, Im] of the observed first row of V_ref (baselines with g_0),
                              unknown antennas only
      VRefUpperTriEmbedding   [Re, Im] of every observed baseline (strict upper triangle)
  Stage 2  q(pw | pk_obs, V_ref)
      PkVRefEmbedding         MLP(log10 P(k)) + polar features of the V_ref first row
      PkVRefUpperEmbedding    MLP(log10 P(k)) + polar features of every observed baseline
      ImageVRefEmbedding      CNN(obsx)       + polar features of the V_ref first row

Identifiability (AA*, 128 px, 512 Mpc): with S_ref known, V_ref[0, j] = conj(g_j) S_ref[0, j]
gives every gain that shares an observed baseline with antenna 0 (230 antennas). With
KNOWN_ANT = 200 that covers all 106 unknown antennas (1-106). For larger unknown sets this can
fail (KNOWN_ANT = 100: antennas 201-206 have no such baseline); the upper-triangle variants
reach every antenna connected to g_0 or a known antenna through observed baselines.

VRefFirstRowEmbedding drops the baselines to the KNOWN_ANT known antennas (g = 1): they are
S_ref[0, j] in every simulation, carry no information on z, and their float32 values differ by
~1e-9 between CPU and GPU. The Gaussian estimator whitens each feature by its training std, so a
constant feature turned that rounding into a ~50σ input at an observation made on the other
device (output/run_01: z posterior uncorrelated with the truth).

Entries that are always zero (unobserved baselines in V_ref, empty k-bins in pk_obs) are
removed by MaskedFlatten, using a non-zero test on the first batch seen.
"""

import torch
import torch.nn as nn

import functionlistnew as func
import settings

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(x, dtype):
    return torch.as_tensor(x, dtype=dtype, device=device)


# ===========================================================================
# Shared building blocks
# ===========================================================================

class MaskedFlatten(nn.Module):
    """
    Flatten (B, ...) → (B, n_kept), keeping only positions that are non-zero somewhere in the
    first batch seen (structural zeros are dropped).

    The mask is not saved in the state dict (persistent=False): it is rebuilt from the first
    batch after loading, so checkpoints load cleanly into a fresh module.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("indices", None, persistent=False)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.indices is None:
            self.indices = (x != 0).any(dim=0).nonzero().squeeze(1)   # (n_kept,)
        return x[:, self.indices]


class VRefRow(nn.Module):
    """
    Observed off-diagonal first row of V_ref: (B, N, N) complex → (B, M) complex.

    known_ant: drop the last known_ant antennas (the known-gain block of the antenna layout, see
    settings.KNOWN_ANT), keeping j = 1 .. N-1-known_ant.
    """
    def __init__(self, known_ant=0):
        super().__init__()
        self.known_ant = known_ant
        self.flat = MaskedFlatten()

    def forward(self, V_ref):
        V = _to_tensor(V_ref, torch.complex64)
        return self.flat(V[:, 0, 1:V.shape[-1] - self.known_ant])


class VRefUpper(nn.Module):
    """
    Observed strict upper triangle of V_ref: (B, N, N) complex → (B, M) complex.

    V_ref is Hermitian, so i < j holds every baseline once; the diagonal is excluded.
    """
    def __init__(self):
        super().__init__()
        self.flat = MaskedFlatten()

    def forward(self, V_ref):
        V = _to_tensor(V_ref, torch.complex64)
        iu, ju = torch.triu_indices(V.shape[-2], V.shape[-1], offset=1, device=V.device)
        return self.flat(V[:, iu, ju])


def polar_features(row):
    """
    (B, M) complex → (B, 3M) real, all ~O(1): [log|v| / 10, cos(arg v), sin(arg v)].

    The log-magnitude keeps absolute amplitude information (needed to infer A after gain
    marginalisation); BatchNorm was discarding this and causing a flat A posterior.
    """
    angle = torch.angle(row)
    return torch.cat([
        torch.log(torch.abs(row) + 1e-10) / 10.0,
        torch.cos(angle),
        torch.sin(angle),
    ], dim=-1).float()


class _ObsVRefEmbedding(nn.Module):
    """Stage-2 base: encode(observation) concatenated with polar_features(V_ref baselines)."""
    def __init__(self):
        super().__init__()
        self.row = VRefRow()

    def encode(self, obs):
        raise NotImplementedError

    def forward(self, obs, V_ref):
        return torch.cat([self.encode(obs), polar_features(self.row(V_ref))], dim=-1)


# ===========================================================================
# Stage 1  —  q(z | V_ref)
# ===========================================================================

class VRefFirstRowEmbedding(nn.Module):
    """
    Input:  V_ref (B, N, N) complex
    Output: (B, 2M) float64 — [Re, Im] of V_ref[:, 0, j] on the M observed baselines to the
            unknown antennas j = 1 .. N-1-KNOWN_ANT (M = 106 for AA*, KNOWN_ANT = 200).
    """
    def __init__(self):
        super().__init__()
        self.row = VRefRow(known_ant=settings.KNOWN_ANT)

    def forward(self, V_ref):
        row = self.row(V_ref)
        return torch.cat([row.real, row.imag], dim=-1).double()


class VRefUpperTriEmbedding(VRefFirstRowEmbedding):
    """
    Input:  V_ref (B, N, N) complex
    Output: (B, 2M) float64 — [Re, Im] of the strict upper triangle of V_ref on the M
            observed baselines (26725 for AA*).
    """
    def __init__(self):
        super().__init__()
        self.row = VRefUpper()


# ===========================================================================
# Stage 2  —  q(pw | ·, V_ref)
# ===========================================================================

class PkVRefEmbedding(_ObsVRefEmbedding):
    """
    Inputs:  pk_obs (B, S, n_bins) real — P(k) of the dirty image obsx per UV subset
             V_ref  (B, N, N)      complex
    Output:  (B, feat_dim + 3M) real
    """
    def __init__(self, hidden_dim=256, feat_dim=128, roundoff_rtol=1e-11, log_floor=1e-15):
        super().__init__()
        self.roundoff_rtol = roundoff_rtol
        self.log_floor = log_floor
        self.flat = MaskedFlatten()                 # drops empty and no-data k-bins (zeroed below)
        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feat_dim),
            nn.GELU(),
        )

    def encode(self, pk_obs):
        x = torch.nan_to_num(_to_tensor(pk_obs, torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # Empty and no-data (FFT round-off only) bins → 0, so MaskedFlatten drops them
        x = func.drop_roundoff_bins(x, self.roundoff_rtol, fill=0.0)
        x = self.flat(x)                            # (B, n_kept)
        # log keeps amplitude relationships; the floor only guards log(0): in ABSOLUTE terms bins
        # with data measure >= ~2.5e-3 (minimum over 1024 prior draws with random gains),
        # far above log_floor = 1e-15, so clamp leaves every data value unchanged
        return self.mlp(torch.log10(x.clamp(min=self.log_floor)))


class PkVRefUpperEmbedding(PkVRefEmbedding):
    """As PkVRefEmbedding, with V_ref polar features from every observed baseline."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.row = VRefUpper()


class ImageVRefEmbedding(_ObsVRefEmbedding):
    """
    Inputs:  obsx  (B, S, H, W) real [mK] — dirty images (Hermitian UV maps → real)
             V_ref (B, N, N)    complex
    Output:  (B, cnn_features + 3M) real
    """
    def __init__(self, backbone: str = "resnet50d"):
        super().__init__()
        import timm
        self.encoder = timm.create_model(backbone, pretrained=True, in_chans=2, num_classes=0)

    def encode(self, obsx):
        inp = _to_tensor(obsx, torch.float32) / 27.0               # in units of the mean Tb (27 mK)
        return self.encoder(inp).flatten(1)
