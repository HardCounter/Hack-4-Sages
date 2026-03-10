"""Tests for Pydantic physics guardrails."""

import pytest

from modules.validators import PlanetaryParameters, SimulationOutput, StellarParameters


class TestStellarParameters:
    def test_valid_sun(self):
        s = StellarParameters(name="Sun", teff=5778, radius=1.0, mass=1.0)
        assert s.teff == 5778

    def test_rejects_too_cold(self):
        with pytest.raises(Exception):
            StellarParameters(name="cold", teff=500, radius=1.0, mass=1.0)

    def test_rejects_too_small_radius(self):
        with pytest.raises(Exception):
            StellarParameters(name="tiny", teff=3000, radius=0.01, mass=1.0)


class TestPlanetaryParameters:
    def test_valid_earth(self):
        p = PlanetaryParameters(
            name="Earth", radius_earth=1.0, mass_earth=1.0,
            semi_major_axis=1.0, albedo=0.3,
        )
        assert p.radius_earth == 1.0

    def test_rejects_too_large_radius(self):
        with pytest.raises(Exception):
            PlanetaryParameters(
                name="giant", radius_earth=100.0,
                semi_major_axis=1.0, albedo=0.3,
            )

    def test_mass_radius_consistency_rejects_extreme(self):
        with pytest.raises(Exception):
            PlanetaryParameters(
                name="bad", radius_earth=1.0, mass_earth=1000.0,
                semi_major_axis=1.0, albedo=0.3,
            )


class TestSimulationOutput:
    def test_valid_output(self):
        o = SimulationOutput(T_eq_K=255.0, ESI=0.85, flux_earth=1.0)
        assert o.ESI == 0.85

    def test_rejects_negative_esi(self):
        with pytest.raises(Exception):
            SimulationOutput(T_eq_K=255.0, ESI=-0.1, flux_earth=1.0)

    def test_rejects_impossible_temperature(self):
        with pytest.raises(Exception):
            SimulationOutput(T_eq_K=1.0, ESI=0.5, flux_earth=1.0)
