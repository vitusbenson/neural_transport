




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
