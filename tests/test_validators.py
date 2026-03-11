"""Tests for Pydantic physics guardrails."""

import pytest

from modules.validators import (
    PLANET_MASS_UPPER_MEARTH,
    PlanetaryParameters,
    SimulationOutput,
    StellarParameters,
)


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

    def test_eccentricity_field_accepted(self):
        p = PlanetaryParameters(
            name="Eccentric", radius_earth=1.0, mass_earth=1.0,
            semi_major_axis=1.0, albedo=0.3, eccentricity=0.5,
        )
        assert p.eccentricity == 0.5

    def test_eccentricity_rejects_too_high(self):
        with pytest.raises(Exception):
            PlanetaryParameters(
                name="bad_ecc", radius_earth=1.0, mass_earth=1.0,
                semi_major_axis=1.0, albedo=0.3, eccentricity=0.95,
            )

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

    def test_rejects_brown_dwarf_mass(self):
        with pytest.raises(Exception, match="brown dwarf"):
            PlanetaryParameters(
                name="BD", radius_earth=15.0,
                mass_earth=PLANET_MASS_UPPER_MEARTH + 100,
                semi_major_axis=1.0, albedo=0.3,
            )

    def test_accepts_max_planet_mass(self):
        p = PlanetaryParameters(
            name="Heavy", radius_earth=15.0,
            mass_earth=3000.0,
            semi_major_axis=1.0, albedo=0.3,
        )
        assert p.mass_earth == 3000.0

    def test_surface_type_validation(self):
        with pytest.raises(Exception, match="surface_type"):
            PlanetaryParameters(
                name="bad_surf", radius_earth=1.0,
                semi_major_axis=1.0, albedo=0.3,
                surface_type="lava",
            )

    def test_atmosphere_type_validation(self):
        with pytest.raises(Exception, match="atmosphere_type"):
            PlanetaryParameters(
                name="bad_atm", radius_earth=1.0,
                semi_major_axis=1.0, albedo=0.3,
                atmosphere_type="venus_like",
            )

    def test_albedo_none_allowed(self):
        p = PlanetaryParameters(
            name="no_albedo", radius_earth=1.0,
            semi_major_axis=1.0,
        )
        assert p.albedo is None


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
