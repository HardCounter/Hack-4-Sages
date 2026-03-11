"""Tests for the astrophysics calculation engine."""

import numpy as np
import pytest

from modules.astro_physics import (
    assess_biosignature_false_positives,
    compute_esi,
    compute_sephi,
    equilibrium_temperature,
    estimate_albedo,
    estimate_atmospheric_escape,
    estimate_density,
    estimate_escape_velocity,
    estimate_isa_interaction,
    estimate_outgassing_rate,
    estimate_uv_flux,
    habitable_surface_fraction,
    hz_boundaries,
    orbit_averaged_flux_factor,
    redistribution_factor,
    stellar_flux,
)


class TestEquilibriumTemperature:
    def test_earth_analogue(self):
        T = equilibrium_temperature(5778, 1.0, 1.0, 0.3, False)
        assert 240 < T < 270, f"Earth T_eq should be ~255 K, got {T}"

    def test_tidally_locked_hotter(self):
        T_free = equilibrium_temperature(3000, 0.15, 0.05, 0.3, False)
        T_locked = equilibrium_temperature(3000, 0.15, 0.05, 0.3, True)
        assert T_locked > T_free

    def test_higher_albedo_cooler(self):
        T_low = equilibrium_temperature(5778, 1.0, 1.0, 0.1, False)
        T_high = equilibrium_temperature(5778, 1.0, 1.0, 0.9, False)
        assert T_high < T_low

    def test_closer_is_hotter(self):
        T_close = equilibrium_temperature(5778, 1.0, 0.5, 0.3, False)
        T_far = equilibrium_temperature(5778, 1.0, 2.0, 0.3, False)
        assert T_close > T_far

    def test_eccentricity_increases_temperature(self):
        T_circ = equilibrium_temperature(5778, 1.0, 1.0, 0.3, False, eccentricity=0.0)
        T_ecc = equilibrium_temperature(5778, 1.0, 1.0, 0.3, False, eccentricity=0.5)
        assert T_ecc > T_circ

    def test_eccentricity_zero_unchanged(self):
        T = equilibrium_temperature(5778, 1.0, 1.0, 0.3, False, eccentricity=0.0)
        T2 = equilibrium_temperature(5778, 1.0, 1.0, 0.3, False)
        assert abs(T - T2) < 0.01


class TestRedistributionFactor:
    def test_locked_moderate_returns_sqrt2(self):
        f = redistribution_factor(True, "moderate")
        assert abs(f - np.sqrt(2)) < 0.01

    def test_fast_moderate_returns_2(self):
        f = redistribution_factor(False, "moderate")
        assert abs(f - 2.0) < 0.01

    def test_thick_locked_higher_than_thin(self):
        f_thick = redistribution_factor(True, "thick")
        f_thin = redistribution_factor(True, "thin")
        assert f_thick > f_thin

    def test_all_classes_valid(self):
        for cls in ("thin", "moderate", "thick"):
            for locked in (True, False):
                f = redistribution_factor(locked, cls)
                assert 1.0 < f <= 2.0


class TestOrbitAveragedFlux:
    def test_circular_returns_one(self):
        assert abs(orbit_averaged_flux_factor(0.0) - 1.0) < 1e-10

    def test_eccentric_greater_than_one(self):
        assert orbit_averaged_flux_factor(0.5) > 1.0

    def test_high_eccentricity(self):
        assert orbit_averaged_flux_factor(0.8) > orbit_averaged_flux_factor(0.3)


class TestSemiEmpiricalAlbedo:
    def test_default_returns_dict(self):
        a = estimate_albedo()
        assert "albedo" in a
        assert "albedo_uncertainty" in a
        assert 0 < a["albedo"] < 1

    def test_user_override(self):
        a = estimate_albedo(user_override=0.42)
        assert a["albedo"] == 0.42
        assert a["albedo_uncertainty"] == 0.0

    def test_ice_higher_than_ocean(self):
        a_ice = estimate_albedo("ice", "temperate")
        a_ocean = estimate_albedo("ocean", "temperate")
        assert a_ice["albedo"] > a_ocean["albedo"]


class TestStellarFlux:
    def test_earth_gets_solar_constant(self):
        S_abs, S_norm = stellar_flux(5778, 1.0, 1.0)
        assert 0.9 < S_norm < 1.1

    def test_inverse_square_law(self):
        _, S1 = stellar_flux(5778, 1.0, 1.0)
        _, S2 = stellar_flux(5778, 1.0, 2.0)
        assert abs(S1 / S2 - 4.0) < 0.1

    def test_eccentricity_increases_flux(self):
        _, S_circ = stellar_flux(5778, 1.0, 1.0, eccentricity=0.0)
        _, S_ecc = stellar_flux(5778, 1.0, 1.0, eccentricity=0.5)
        assert S_ecc > S_circ


class TestESI:
    def test_earth_has_esi_one(self):
        esi = compute_esi(1.0, 5.51, 11.19, 288.0)
        assert 0.99 <= esi <= 1.0

    def test_esi_bounded(self):
        esi = compute_esi(2.0, 3.0, 8.0, 350.0)
        assert 0.0 <= esi <= 1.0

    def test_very_different_planet_low_esi(self):
        esi = compute_esi(15.0, 0.5, 30.0, 1500.0)
        assert esi < 0.3


class TestSEPHI:
    def test_earth_like_scores_high(self):
        s = compute_sephi(300, 1.0, 1.0)
        assert s["sephi_score"] >= 0.66

    def test_too_cold_fails_thermal(self):
        s = compute_sephi(100, 1.0, 1.0)
        assert s["thermal_ok"] is False

    def test_too_light_fails_atmosphere(self):
        s = compute_sephi(300, 0.05, 0.5)
        assert s["atmosphere_ok"] == False


class TestDensityAndEscapeVel:
    def test_earth_density(self):
        d = estimate_density(1.0, 1.0)
        assert abs(d - 5.51) < 0.01

    def test_earth_escape_velocity(self):
        v = estimate_escape_velocity(1.0, 1.0)
        assert abs(v - 11.19) < 0.01


class TestHabitableZone:
    def test_sun_hz_contains_earth(self):
        hz = hz_boundaries(5778, 1.0)
        inner = hz["runaway_gh"]
        outer = hz["max_gh"]
        assert inner < 1.0 < outer

    def test_cooler_star_closer_hz(self):
        hz_cool = hz_boundaries(3000, 0.01)
        hz_hot = hz_boundaries(6000, 1.0)
        assert hz_cool["runaway_gh"] < hz_hot["runaway_gh"]


class TestHabitableSurfaceFraction:
    def test_all_habitable(self):
        tmap = np.full((32, 64), 300.0)
        assert habitable_surface_fraction(tmap) > 0.99

    def test_all_frozen(self):
        tmap = np.full((32, 64), 100.0)
        assert habitable_surface_fraction(tmap) < 0.01


class TestISA:
    def test_earth_like_high_isa(self):
        isa = estimate_isa_interaction(1.0, 1.0, 288.0, False)
        assert isa["isa_score"] >= 0.75

    def test_outgassing_earth(self):
        og = estimate_outgassing_rate(1.0, 1.0, 4.5)
        assert 0.8 < og["outgassing_rate_earth"] < 1.2

    def test_plate_tectonics_three_levels(self):
        isa = estimate_isa_interaction(1.0, 1.0, 288.0, False)
        assert isa["plate_tectonics"] in ("plausible", "uncertain", "unlikely")

    def test_massive_planet_unlikely_tectonics(self):
        isa = estimate_isa_interaction(10.0, 3.0, 300.0, False)
        assert isa["plate_tectonics"] == "unlikely"

    def test_young_planet_higher_outgassing(self):
        og_old = estimate_outgassing_rate(1.0, 1.0, 4.5)
        og_young = estimate_outgassing_rate(1.0, 1.0, 0.5)
        assert og_young["outgassing_rate_earth"] > og_old["outgassing_rate_earth"]

    def test_outgassing_has_radiogenic_factor(self):
        og = estimate_outgassing_rate(1.0, 1.0, 0.5)
        assert "radiogenic_factor" in og
        assert og["radiogenic_factor"] > 1.0


class TestFalsePositives:
    def test_earth_low_risk(self):
        fp = assess_biosignature_false_positives(5778, 1.0, 1.0, 255, 1.0, 1.0)
        assert fp["overall_false_positive_risk"] == "low"

    def test_uv_flux_earth(self):
        uv = estimate_uv_flux(5778, 1.0, 1.0)
        assert 0.5 < uv["uv_flux_earth"] < 2.0

    def test_uv_has_fraction_field(self):
        uv = estimate_uv_flux(5778, 1.0, 1.0)
        assert "uv_fraction_used" in uv

    def test_m_dwarf_lower_uv_fraction(self):
        uv_m = estimate_uv_flux(3000, 0.15, 0.05)
        uv_g = estimate_uv_flux(5778, 1.0, 1.0)
        assert uv_m["uv_fraction_used"] < uv_g["uv_fraction_used"]


class TestAtmosphericEscape:
    def test_earth_retained(self):
        esc = estimate_atmospheric_escape(1.0, 1.0, 5778, 1.0, 1.0)
        assert esc["escape_flag"] == "retained"

    def test_small_planet_close_to_hot_star(self):
        esc = estimate_atmospheric_escape(0.05, 0.3, 6500, 0.01, 1.2, age_gyr=0.5)
        assert esc["escape_flag"] in ("escape_dominated", "borderline")

    def test_returns_timescale(self):
        esc = estimate_atmospheric_escape(1.0, 1.0, 5778, 1.0, 1.0)
        assert esc["escape_timescale_gyr"] is not None
        assert esc["escape_timescale_gyr"] > 0
