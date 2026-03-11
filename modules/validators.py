"""
Pydantic validation models — physics guardrails.

These validators act as safety barriers preventing the LLM agent
(or any other upstream component) from producing results that
violate thermodynamic or astrophysical constraints.
"""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

M_EARTH_PER_M_JUP = 317.83
DEUTERIUM_BURNING_LIMIT_MJUP = 13.0
PLANET_MASS_UPPER_MEARTH = DEUTERIUM_BURNING_LIMIT_MJUP * M_EARTH_PER_M_JUP


class StellarParameters(BaseModel):
    """Validated stellar host parameters."""

    name: str = Field(min_length=1, max_length=100)
    teff: float = Field(
        ge=2000, le=50000, description="Effective temperature [K]"
    )
    radius: float = Field(
        ge=0.08, le=100.0, description="Stellar radius [R☉]"
    )
    mass: float = Field(
        ge=0.08, le=150.0, description="Stellar mass [M☉]"
    )
    luminosity: Optional[float] = Field(
        default=None, ge=-5.0, le=7.0, description="log(L/L☉)"
    )

    @field_validator("teff")
    @classmethod
    def validate_teff_physical(cls, v: float) -> float:
        if v < 2000:
            raise ValueError(
                f"T_eff={v}K below hydrogen-burning limit (~2000K)"
            )
        if v > 50000:
            raise ValueError(
                f"T_eff={v}K exceeds O-type main-sequence upper bound"
            )
        return v


class PlanetaryParameters(BaseModel):
    """Validated planetary parameters with physics constraints.

    The mass upper bound is set at the deuterium-burning limit
    (~13 M_Jup ≈ 4132 M_Earth). Objects above this threshold
    are brown dwarfs and fall outside the scope of the exoplanet
    simulation pipeline.
    """

    name: str = Field(min_length=1, max_length=100)
    radius_earth: float = Field(
        ge=0.3, le=25.0, description="Planet radius [R⊕]"
    )
    mass_earth: Optional[float] = Field(
        default=None, ge=0.01,
        description="Planet mass [M⊕]",
    )
    semi_major_axis: float = Field(
        ge=0.001, le=1000.0, description="Semi-major axis [AU]"
    )
    eccentricity: float = Field(
        default=0.0, ge=0.0, le=0.9,
        description="Orbital eccentricity (0 = circular)",
    )
    albedo: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Bond albedo (None → estimated from surface/atmosphere class)",
    )
    surface_type: str = Field(
        default="mixed_rocky",
        description="Surface class for albedo estimation",
    )
    atmosphere_type: str = Field(
        default="temperate",
        description="Atmosphere class for albedo estimation",
    )
    tidally_locked: bool = False
    orbital_period: Optional[float] = Field(
        default=None, ge=0.01, description="Orbital period [days]"
    )
    insol: Optional[float] = Field(
        default=None, ge=0.0, le=10000.0, description="Instellation [S_Earth]"
    )

    @field_validator("radius_earth")
    @classmethod
    def validate_not_star(cls, v: float) -> float:
        if v > 25.0:
            raise ValueError(f"Radius {v} R_Earth exceeds planet definition")
        return v

    @field_validator("mass_earth")
    @classmethod
    def validate_not_brown_dwarf(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v > PLANET_MASS_UPPER_MEARTH:
            raise ValueError(
                f"Mass {v:.0f} M_Earth exceeds deuterium-burning limit "
                f"(~{PLANET_MASS_UPPER_MEARTH:.0f} M_Earth ≈ 13 M_Jup). "
                f"This object is a brown dwarf, not a planet."
            )
        return v

    @field_validator("surface_type")
    @classmethod
    def validate_surface_type(cls, v: str) -> str:
        allowed = {"ocean", "desert", "ice", "mixed_rocky"}
        if v not in allowed:
            raise ValueError(f"surface_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("atmosphere_type")
    @classmethod
    def validate_atmosphere_type(cls, v: str) -> str:
        allowed = {"thin", "temperate", "thick_cloudy"}
        if v not in allowed:
            raise ValueError(
                f"atmosphere_type must be one of {{'thin', 'temperate', 'thick_cloudy'}}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def validate_mass_radius_consistency(self):
        """Chen & Kipping (2017) empirical mass-radius relation."""
        if self.mass_earth is not None and self.radius_earth is not None:
            if self.radius_earth < 4.0:
                expected_r = self.mass_earth ** 0.279
                if expected_r > 0:
                    ratio = self.radius_earth / expected_r
                    if ratio < 0.2 or ratio > 5.0:
                        raise ValueError(
                            f"Mass-radius relation violated "
                            f"(M={self.mass_earth} M\u2295, R={self.radius_earth} R\u2295, "
                            f"expected R\u2295~{expected_r:.2f})"
                        )
        return self


class SimulationOutput(BaseModel):
    """Validated output from the physics engine."""

    T_eq_K: float = Field(ge=10.0, le=5000.0)
    ESI: float = Field(ge=0.0, le=1.0)
    flux_earth: float = Field(ge=0.0)
    temperature_map: Optional[List] = None

    @field_validator("T_eq_K")
    @classmethod
    def validate_temperature_thermodynamics(cls, v: float) -> float:
        if v < 2.7:
            raise ValueError(
                f"T={v}K below cosmic microwave background (2.7K)"
            )
        return v

    @field_validator("temperature_map")
    @classmethod
    def validate_temperature_map(cls, v: Optional[List]) -> Optional[List]:
        if v is not None:
            arr = np.asarray(v, dtype=np.float64)
            if np.any(np.isnan(arr)):
                raise ValueError("Temperature map contains NaN")
            if np.any(arr < 0):
                raise ValueError("Temperature map contains negative values")
            if np.any(arr > 5000):
                raise ValueError(
                    "Temperature map contains unphysically high values"
                )
        return v
