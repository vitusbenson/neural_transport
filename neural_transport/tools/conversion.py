import numpy as np
import torch


R_EARTH = 6.371e3  # km

M_CO2 = 44.009e-3
M_air = 28.9652e-3
M_C = 12.011e-3
M_CH4 = 16.043e-3
M_CO = 28.0101e-3

def massmix_to_density(massmix, airdensity, ppm = False, eps = 1e-12):
    if ppm:
        return massmix * 1e-6 * (airdensity + eps)
    else:
        return massmix * (airdensity + eps)

def density_to_massmix(density, airdensity, ppm = False, eps = 1e-12):
    if ppm:
        return density / (airdensity + eps) * 1e6
    else:
        return density / (airdensity + eps)

def molemix_to_massmix(molemix, M = M_CO2):
    return molemix * M / M_air

def massmix_to_molemix(massmix, M = M_CO2):
    return massmix * M_air / M

def massmix_to_mass(massmix, airdensity, V, ppm = False, eps = 1e-12):
    if ppm:
        return massmix * 1e-6 * (airdensity * V + eps)
    else:
        return massmix * (airdensity * V + eps)

def mass_to_massmix(mass, airdensity, V, ppm = False, eps = 1e-12):
    if ppm:
        return mass / (airdensity * V + eps) * 1e6
    else:
        return mass / (airdensity * V + eps)

def density_to_mass(density, V, eps = 1e-12):
    return density * (V + eps)

def mass_to_density(mass, V, eps = 1e-12):
    return mass / (V + eps)

def zonal_wavenumber_to_wavelength(k, lat=0.0):
    """
    Convert zonal wavenumber to wavelength in km for a given latitude.

    Parameters
    ----------
    k : array-like or float
        Zonal wavenumber (1 = one wave around a full circle)
    lat : float
        Latitude in degrees. Determines the effective circumference.

    Returns
    -------
    wavelength : array-like or float
        Corresponding wavelength in km.
    """
    lat_rad = np.deg2rad(lat)
    circumference = 2 * np.pi * R_EARTH * np.cos(lat_rad)
    k = np.maximum(np.array(k, dtype=float), 1e-6)  # avoid division by zero
    return circumference / k

def wavelength_to_zonal_wavenumber(wavelength_km, lat=0.0):
    """
    Convert wavelength (in km) to zonal wavenumber for a given latitude.

    Parameters
    ----------
    wavelength_km : array-like or float
        Wavelength in km.
    lat : float
        Latitude in degrees. Determines the effective circumference.

    Returns
    -------
    k : array-like or float
        Corresponding zonal wavenumber.
    """
    lat_rad = np.deg2rad(lat)
    circumference = 2 * np.pi * R_EARTH * np.cos(lat_rad)
    wavelength_km = np.maximum(np.array(wavelength_km, dtype=float), 1e-6)
    return circumference / wavelength_km

def km_per_gridcell(batch):
    lat = batch["lat"].values
    lon = batch["lon"].values

    lat_mean = float(np.mean(lat))
    circ_at_lat = 2 * np.pi * R_EARTH * np.cos(np.deg2rad(lat_mean))
    dx = circ_at_lat / len(lon)
    return dx, circ_at_lat

def compute_xco2_via_ak(
    co2_profile,
    ak,
    xco2_prior,
    co2_profile_prior,
):
    """
    Compute XCO₂ via the averaging kernel equation from the CO₂ profile.
    
    Parameters:
    - co2_profile: CO₂ profile to be converted, shape [N, C]
    - ak: Averaging kernel, shape [N, C]
    - xco2_prior: Prior XCO₂, shape [N]
    - co2_profile_prior: Prior CO₂ profile, shape [N, C]

    Returns:
    - xco2: Computed XCO₂, shape [N]
    """
    is_torch = isinstance(co2_profile, torch.Tensor)

    if is_torch:
        xco2 = xco2_prior + (ak * (co2_profile - co2_profile_prior)).sum(dim=-1)
    else:
        xco2 = xco2_prior + np.sum(ak * (co2_profile - co2_profile_prior), axis=-1)
    
    print("\nDEBUG compute_xco2_via_ak")
    print(f"  co2_profile range: {np.nanmin(co2_profile)}, {np.nanmax(co2_profile)}")
    print(f"  ak range: {np.nanmin(ak)}, {np.nanmax(ak)}")
    print(f"  co2_profile_prior range: {np.nanmin(co2_profile_prior)}, {np.nanmax(co2_profile_prior)}")
    print(f"  xco2_prior range: {np.nanmin(xco2_prior)}, {np.nanmax(xco2_prior)}")
    print(f"  xco2 range: {np.nanmin(xco2)}, {np.nanmax(xco2)}")
    print(f"  co2_profile has NaN: {np.isnan(co2_profile).any()}")
    print(f"  ak has NaN: {np.isnan(ak).any()}")
    print(f"  co2_profile_prior has NaN: {np.isnan(co2_profile_prior).any()}")
    print(f"  xco2_prior has NaN: {np.isnan(xco2_prior).any()}")
    print(f"  xco2 has NaN: {np.isnan(xco2).any()}")
    return xco2
