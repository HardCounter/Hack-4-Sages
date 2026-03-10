"""
Pydantic validation models — physics guardrails.

These validators act as safety barriers preventing the LLM agent
(or any other upstream component) from producing results that
violate thermodynamic or astrophysical constraints.
"""

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class StellarParameters(BaseModel):
    """Validated stellar host parameters."""

    name: str = Field(min_length=1, max_length=100)
    teff: float = Field(
        ge=2300, le=10000, description="Effective temperature [K]"
    )
    radius: float = Field(
        ge=0.08, le=100.0, description="Stellar radius [R_sun]"
    )
    mass: float = Field(
        ge=0.08, le=150.0, description="Stellar mass [M_sun]"
    )
    luminosity: Optional[float] = Field(
        default=None, ge=-5.0, le=7.0, description="log(L/L_sun)"
    )

    @field_validator("teff")
    @classmethod
    def validate_teff_physical(cls, v: float) -> float:
        if v < 2300:
            raise ValueError(
                f"T_eff={v}K below brown-dwarf limit (~2300K)"
            )
        return v


class PlanetaryParameters(BaseModel):
    """Validated planetary parameters with physics constraints."""

    name: str = Field(min_length=1, max_length=100)
    radius_earth: float = Field(
        ge=0.3, le=25.0, description="Planet radius [R_Earth]"
    )
    mass_earth: Optional[float] = Field(
        default=None, ge=0.01, le=5000.0, description="Planet mass [M_Earth]"
    )
    semi_major_axis: float = Field(
        ge=0.001, le=1000.0, description="Semi-major axis [AU]"
    )
    albedo: float = Field(ge=0.0, le=1.0, description="Bond albedo")
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

    @model_validator(mode="after")
    def validate_mass_radius_consistency(self):
        """Chen & Kipping (2017) empirical mass-radius relation."""
        if self.mass_earth is not None and self.radius_earth is not None:
            if self.radius_earth < 4.0:
                expected_r = self.mass_earth ** 0.27
                if expected_r > 0:
                    ratio = self.radius_earth / expected_r
                    if ratio < 0.2 or ratio > 5.0:
                        raise ValueError(
                            f"Mass-radius relation violated "
                            f"(M={self.mass_earth}, R={self.radius_earth}, "
                            f"expected R~{expected_r:.2f})"
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
            arr = np.array(v)
            if np.any(arr < 0):
                raise ValueError("Temperature map contains negative values")
            if np.any(arr > 5000):
                raise ValueError(
                    "Temperature map contains unphysically high values"
                )
            if np.any(np.isnan(arr)):
                raise ValueError("Temperature map contains NaN")
        return v
