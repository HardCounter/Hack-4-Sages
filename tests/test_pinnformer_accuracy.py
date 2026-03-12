"""
Comprehensive accuracy and physics-consistency tests for a trained
PINNFormer 3-D model.

Two usage modes
---------------
1. **Post-training validation** (run against saved weights):

       pytest tests/test_pinnformer_accuracy.py -v

   Loads ``models/pinn3d_weights.pt`` and checks physical plausibility,
   PDE residual quality, day-night contrast, smoothness, equatorial
   symmetry, and comparison against the analytical eyeball map.

2. **Quick CI smoke test** (trains a tiny model on the spot):

       pytest tests/test_pinnformer_accuracy.py -v -k smoke

   Trains 200 epochs on CPU with 1024 collocation points.  Useful for
   regression detection, not for accuracy gating.

Exit codes
----------
All assertions carry descriptive messages so a failed test immediately
tells you *which* physics property was violated and by how much.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.pinnformer3d import (
    PINNPhysicsConfig,
    PINNFormer3D,
    load_pinnformer,
    sample_surface_map,
    train_pinnformer,
    pinn_loss_3d,
    _sample_collocation,
    SIGMA,
    KAPPA_ATM,
    S_MAX,
)
from modules.astro_physics import habitable_surface_fraction
from modules.visualization import generate_eyeball_map

WEIGHTS_PATH = os.path.join(ROOT, "models", "pinn3d_weights.pt")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model():
    """Load the production-trained model if weights exist, else skip."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip(f"No trained weights at {WEIGHTS_PATH}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_pinnformer(WEIGHTS_PATH, device=device)
    model.eval()
    return model, device


@pytest.fixture(scope="module")
def trained_cfg():
    """Load physics config stored alongside weights."""
    if not os.path.exists(WEIGHTS_PATH):
        pytest.skip(f"No trained weights at {WEIGHTS_PATH}")
    payload = torch.load(WEIGHTS_PATH, map_location="cpu")
    if isinstance(payload, dict) and "physics_config" in payload:
        return PINNPhysicsConfig(**payload["physics_config"])
    return PINNPhysicsConfig()


@pytest.fixture(scope="module")
def surface_map(trained_model):
    model, device = trained_model
    return sample_surface_map(model, n_lat=64, n_lon=128, device=device)


@pytest.fixture(scope="module")
def smoke_model():
    """Train a minimal model for CI smoke tests."""
    cfg = PINNPhysicsConfig.from_mode("basic")
    device = "cpu"
    model, history = train_pinnformer(
        cfg=cfg,
        n_colloc=1024,
        epochs=200,
        device=device,
        log_every=100,
        verbose=False,
        warmup_epochs=20,
        resample_every=50,
    )
    return model, history, device


# ── 1. Physical plausibility ─────────────────────────────────────────────────

class TestPhysicalPlausibility:
    """Temperature predictions must be physically sensible."""

    def test_no_nan_or_inf(self, surface_map):
        assert np.all(np.isfinite(surface_map)), (
            "Surface map contains NaN or Inf values"
        )

    def test_temperature_range(self, surface_map):
        t_min, t_max = surface_map.min(), surface_map.max()
        assert t_min >= 30.0, f"T_min={t_min:.1f} K is below 30 K (unphysical)"
        assert t_max <= 3000.0, f"T_max={t_max:.1f} K exceeds 3000 K (unphysical)"

    def test_substellar_hotter_than_nightside(self, surface_map):
        """The dayside (center columns) must be warmer than the nightside."""
        n_lon = surface_map.shape[1]
        q = n_lon // 4
        dayside = surface_map[:, q : 3 * q]
        nightside = np.concatenate(
            [surface_map[:, :q], surface_map[:, 3 * q :]], axis=1,
        )
        day_mean = dayside.mean()
        night_mean = nightside.mean()
        assert day_mean > night_mean, (
            f"Dayside mean ({day_mean:.1f} K) should exceed nightside "
            f"({night_mean:.1f} K) for a tidally locked planet"
        )

    def test_day_night_contrast(self, surface_map):
        """There should be a meaningful temperature contrast (> 20 K)."""
        n_lon = surface_map.shape[1]
        q = n_lon // 4
        day_max = surface_map[:, q : 3 * q].max()
        night_min = surface_map[:, :q].min()
        contrast = day_max - night_min
        assert contrast > 20.0, (
            f"Day-night contrast is only {contrast:.1f} K — expected > 20 K"
        )

    def test_habitable_fraction_range(self, surface_map):
        f_hab = habitable_surface_fraction(surface_map)
        assert 0.0 <= f_hab <= 1.0, (
            f"Habitable fraction {f_hab:.3f} outside [0, 1]"
        )

    def test_map_shape(self, surface_map):
        assert surface_map.shape == (64, 128), (
            f"Expected (64, 128), got {surface_map.shape}"
        )


# ── 2. PDE residual quality ─────────────────────────────────────────────────

class TestPDEResidual:
    """The trained model should approximately satisfy the governing PDE."""

    def _compute_residual_stats(self, model, cfg, device, n_pts=8192):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        model.eval()
        x = _sample_collocation(n_pts, device).requires_grad_(True)
        with sdpa_kernel([SDPBackend.MATH]):
            out = model(x)
        T = out[:, 0]
        grad_T = torch.autograd.grad(T.sum(), x, create_graph=True)[0]
        lap = torch.zeros(n_pts, device=device)
        for i in range(3):
            g2 = torch.autograd.grad(
                grad_T[:, i].sum(), x,
                create_graph=False, retain_graph=(i < 2),
            )[0]
            lap = lap + g2[:, i]

        theta, phi = x[:, 0], x[:, 1]
        S = S_MAX * torch.clamp(
            torch.cos(theta.detach()) * torch.cos(phi.detach()), min=0,
        )
        emission = SIGMA * T ** 4
        if cfg.enable_greenhouse:
            emission = (1.0 - cfg.greenhouse_G) * emission
        residual = KAPPA_ATM * lap + S * (1.0 - 0.30) - emission
        res_np = residual.detach().cpu().numpy()
        return {
            "rmse": float(np.sqrt(np.mean(res_np ** 2))),
            "max": float(np.max(np.abs(res_np))),
            "mean": float(np.mean(np.abs(res_np))),
        }

    def test_pde_residual_rmse(self, trained_model, trained_cfg):
        model, device = trained_model
        stats = self._compute_residual_stats(model, trained_cfg, device)
        assert stats["rmse"] < 500.0, (
            f"PDE residual RMSE = {stats['rmse']:.2e} — should be < 500. "
            "Model may need more training epochs."
        )

    def test_pde_residual_max(self, trained_model, trained_cfg):
        model, device = trained_model
        stats = self._compute_residual_stats(model, trained_cfg, device)
        assert stats["max"] < 5000.0, (
            f"PDE residual max = {stats['max']:.2e} — localised violation. "
            "Check for boundary artifacts."
        )


# ── 3. Smoothness ────────────────────────────────────────────────────────────

class TestSmoothness:
    """Temperature maps should be smooth — no discontinuities or spikes."""

    def test_no_sharp_latitudinal_jumps(self, surface_map):
        dlat = np.pi / (surface_map.shape[0] - 1)
        dT = np.abs(np.diff(surface_map, axis=0)) / dlat
        max_grad = dT.max()
        assert max_grad < 5000.0, (
            f"Max latitudinal gradient = {max_grad:.1f} K/rad — "
            "suggests a discontinuity or spike"
        )

    def test_no_sharp_longitudinal_jumps(self, surface_map):
        dlon = 2 * np.pi / (surface_map.shape[1] - 1)
        dT = np.abs(np.diff(surface_map, axis=1)) / dlon
        max_grad = dT.max()
        assert max_grad < 5000.0, (
            f"Max longitudinal gradient = {max_grad:.1f} K/rad — "
            "suggests a discontinuity or spike"
        )

    def test_temperature_std_reasonable(self, surface_map):
        std = surface_map.std()
        assert 5.0 < std < 500.0, (
            f"T std = {std:.1f} K — "
            "too low means flat map, too high means chaotic"
        )


# ── 4. Equatorial symmetry ──────────────────────────────────────────────────

class TestSymmetry:
    """Tidally locked planets should be approximately symmetric about the
    equator (north-south) when there is no obliquity."""

    def test_north_south_symmetry(self, surface_map):
        n_lat = surface_map.shape[0]
        mid = n_lat // 2
        north = surface_map[:mid, :]
        south = surface_map[mid:, :][::-1, :]
        if north.shape != south.shape:
            min_rows = min(north.shape[0], south.shape[0])
            north = north[:min_rows]
            south = south[:min_rows]

        rmse_ns = float(np.sqrt(np.mean((north - south) ** 2)))
        t_range = surface_map.max() - surface_map.min()
        relative = rmse_ns / max(t_range, 1.0)
        assert relative < 0.25, (
            f"N-S asymmetry RMSE = {rmse_ns:.1f} K "
            f"({relative*100:.1f}% of T range) — "
            "should be < 25% for zero-obliquity planet"
        )


# ── 5. Comparison against analytical eyeball map ─────────────────────────────

class TestAnalyticalComparison:
    """Compare trained PINN output against the parametric analytical map."""

    @pytest.fixture
    def analytic_map(self):
        return generate_eyeball_map(
            T_eq=254.0, tidally_locked=True, n_lat=64, n_lon=128,
        )

    def test_pattern_correlation(self, surface_map, analytic_map):
        r = float(np.corrcoef(
            surface_map.ravel(), analytic_map.ravel(),
        )[0, 1])
        assert r > 0.3, (
            f"Pattern correlation with analytic = {r:.3f} — "
            "expected > 0.3 for even a roughly correct map"
        )

    def test_rmse_vs_analytic(self, surface_map, analytic_map):
        rmse = float(np.sqrt(np.mean((surface_map - analytic_map) ** 2)))
        assert rmse < 200.0, (
            f"RMSE vs analytic = {rmse:.1f} K — "
            "model disagrees substantially with analytical solution"
        )

    def test_mean_temperature_order_of_magnitude(self, surface_map):
        t_mean = surface_map.mean()
        assert 50.0 < t_mean < 600.0, (
            f"Mean T = {t_mean:.1f} K — outside plausible range "
            "[50, 600] K for a Proxima-b-like planet"
        )


# ── 6. Smoke test (CPU, tiny model) ─────────────────────────────────────────

class TestSmoke:
    """Quick CI test: train a tiny model and check basic properties."""

    def test_smoke_training_converges(self, smoke_model):
        _, history, _ = smoke_model
        assert len(history.loss_total) >= 2, "Not enough logged epochs"
        assert history.loss_total[-1] < history.loss_total[0], (
            f"Loss did not decrease: "
            f"{history.loss_total[0]:.4e} -> {history.loss_total[-1]:.4e}"
        )

    def test_smoke_finite_validation(self, smoke_model):
        _, history, _ = smoke_model
        v = history.validation
        assert np.isfinite(v["pde_residual_rmse"]), "PDE RMSE is not finite"
        assert np.isfinite(v["T_mean"]), "T_mean is not finite"

    def test_smoke_temperature_range(self, smoke_model):
        model, _, device = smoke_model
        tmap = sample_surface_map(model, device=device)
        assert tmap.shape == (32, 64)
        assert np.all(np.isfinite(tmap))
        assert tmap.min() >= 30.0
        assert tmap.max() <= 3000.0

    def test_smoke_validation_has_keys(self, smoke_model):
        """Verify validation dict has all expected keys."""
        _, history, _ = smoke_model
        v = history.validation
        for key in ("pde_residual_rmse", "T_mean", "T_min", "T_max", "T_std"):
            assert key in v, f"Missing validation key: {key}"
            assert np.isfinite(v[key]), f"Validation key {key} is not finite"


# ── 7. Multi-resolution consistency ─────────────────────────────────────────

class TestMultiResolution:
    """Maps at different resolutions should be qualitatively consistent."""

    def test_mean_temperature_consistent_across_resolutions(self, trained_model):
        model, device = trained_model
        map_lo = sample_surface_map(model, n_lat=16, n_lon=32, device=device)
        map_hi = sample_surface_map(model, n_lat=64, n_lon=128, device=device)
        diff = abs(map_lo.mean() - map_hi.mean())
        assert diff < 30.0, (
            f"Mean T differs by {diff:.1f} K between 16x32 and 64x128 grids"
        )


# ── 8. Boundary condition satisfaction ───────────────────────────────────────

class TestBoundaryConditions:
    """The model should approximately respect its training BCs."""

    def test_substellar_point_temperature(self, trained_model):
        model, device = trained_model
        model.eval()
        x_sub = torch.zeros(1, 3, device=device)
        with torch.no_grad():
            T_sub = model(x_sub)[0, 0].item()
        assert 200.0 < T_sub < 500.0, (
            f"Substellar T = {T_sub:.1f} K — expected near 320 K BC"
        )

    def test_nightside_point_temperature(self, trained_model):
        model, device = trained_model
        model.eval()
        x_night = torch.tensor([[0.0, np.pi, 0.0]], device=device)
        with torch.no_grad():
            T_night = model(x_night)[0, 0].item()
        assert 30.0 < T_night < 250.0, (
            f"Nightside T = {T_night:.1f} K — expected near 80 K BC"
        )


# ── CLI runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
