"""Tests for the astrophysics calculation engine."""

import numpy as np
import pytest

from modules.astro_physics import (
    assess_biosignature_false_positives,
    compute_esi,
    compute_sephi,
    equilibrium_temperature,
    estimate_density,
    estimate_escape_veloócity,
    estimate_isa_interaction,
    estimate_outgassing_rate,
    estimate_uv_flux,
    habitable_surface_fraction,
    hz_boundaries,
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


class TestStellarFlux:
    def test_earth_gets_solar_constant(self):
        S_abs, S_norm = stellar_flux(5778, 1.0, 1.0)
        assert 0.9 < S_norm < 1.1

    def test_inverse_square_law(self):
        _, S1 = stellar_flux(5778, 1.0, 1.0)
        _, S2 = stellar_flux(5778, 1.0, 2.0)
        assert abs(S1 / S2 - 4.0) < 0.1


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


class TestFalsePositives:
    def test_earth_low_risk(self):
        fp = assess_biosignature_false_positives(5778, 1.0, 1.0, 255, 1.0, 1.0)
        assert fp["overall_false_positive_risk"] == "low"

    def test_uv_flux_earth(self):
        uv = estimate_uv_flux(5778, 1.0, 1.0)
        assert 0.5 < uv["uv_flux_earth"] < 2.0
