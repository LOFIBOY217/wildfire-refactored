"""
Canadian Forest Fire Weather Index (FWI) System — Van Wagner (1987).

Vectorized numpy implementation for gridded data. Computes all 6 FWI
components (FFMC, DMC, DC, ISI, BUI, FWI) from daily noon weather:
  - temperature (°C)
  - relative humidity (%)
  - wind speed (km/h)
  - 24h precipitation (mm)

Reference:
  Van Wagner, C.E. 1987. Development and structure of the Canadian
  Forest Fire Weather Index System. Forestry Technical Report 35.

Usage:
  calc = FWICalculator()
  for day in range(n_days):
      result = calc.update(temp[day], rh[day], wind[day], rain[day], month[day])
      fwi_values = result["FWI"]
"""

import numpy as np


# Day-length adjustment factors by month (index 0 = January)
# For DMC (Le) — effective day-length, latitude >= 30N (Canada)
DMC_DAY_LENGTH = np.array(
    [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0],
    dtype=np.float64,
)

# For DC (Lf) — day-length factor, latitude >= 20N
DC_DAY_LENGTH = np.array(
    [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6],
    dtype=np.float64,
)

# Standard starting values (spring startup)
DEFAULT_FFMC = 85.0
DEFAULT_DMC = 6.0
DEFAULT_DC = 15.0


def _ffmc_from_moisture(m):
    """Convert moisture content (%) to FFMC code."""
    return 59.5 * (250.0 - m) / (147.2 + m)


def _moisture_from_ffmc(f):
    """Convert FFMC code to moisture content (%)."""
    return 147.2 * (101.0 - f) / (59.5 + f)


def compute_ffmc(temp, rh, wind, rain, ffmc_prev):
    """
    Compute Fine Fuel Moisture Code (FFMC).

    All inputs can be scalars or numpy arrays of the same shape.
    Returns FFMC (same shape as inputs).
    """
    temp = np.asarray(temp, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    wind = np.asarray(wind, dtype=np.float64)
    rain = np.asarray(rain, dtype=np.float64)
    ffmc_prev = np.asarray(ffmc_prev, dtype=np.float64)

    mo = _moisture_from_ffmc(ffmc_prev)

    # --- Rain correction (threshold 0.5 mm) ---
    has_rain = rain > 0.5
    if np.any(has_rain):
        rf = np.where(has_rain, rain - 0.5, 0.0)

        # Base wetting
        delta_mr = (
            42.5 * rf * np.exp(-100.0 / (251.0 - mo))
            * (1.0 - np.exp(-6.93 / np.maximum(rf, 1e-10)))
        )
        # Extra wetting for high moisture
        high_m = mo > 150.0
        extra = np.where(
            high_m,
            0.0015 * (mo - 150.0) ** 2 * np.sqrt(np.maximum(rf, 0.0)),
            0.0,
        )
        delta_mr = delta_mr + extra
        mo = np.where(has_rain, np.minimum(mo + delta_mr, 250.0), mo)

    # --- Equilibrium moisture content ---
    # Drying EMC (Ed)
    ed = (
        0.942 * np.power(rh, 0.679)
        + 11.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    )
    # Wetting EMC (Ew)
    ew = (
        0.618 * np.power(rh, 0.753)
        + 10.0 * np.exp((rh - 100.0) / 10.0)
        + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
    )

    # --- Drying (mo > Ed) ---
    ko = (
        0.424 * (1.0 - np.power(rh / 100.0, 1.7))
        + 0.0694 * np.sqrt(wind) * (1.0 - np.power(rh / 100.0, 8.0))
    )
    kd = ko * 0.0579 * np.exp(0.0365 * temp)
    m_dry = ed + (mo - ed) * np.power(10.0, -kd)

    # --- Wetting (mo < Ew) ---
    k1 = (
        0.424 * (1.0 - np.power((100.0 - rh) / 100.0, 1.7))
        + 0.0694 * np.sqrt(wind) * (1.0 - np.power((100.0 - rh) / 100.0, 8.0))
    )
    kw = k1 * 0.0579 * np.exp(0.0365 * temp)
    m_wet = ew - (ew - mo) * np.power(10.0, -kw)

    # Select based on moisture vs EMC
    m = np.where(mo > ed, m_dry, np.where(mo < ew, m_wet, mo))
    m = np.clip(m, 0.0, 250.0)

    ffmc = _ffmc_from_moisture(m)
    return np.clip(ffmc, 0.0, 101.0)


def compute_dmc(temp, rh, rain, month, dmc_prev):
    """
    Compute Duff Moisture Code (DMC).

    month: 1-12 integer (scalar or array).
    """
    temp = np.asarray(temp, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    rain = np.asarray(rain, dtype=np.float64)
    dmc_prev = np.asarray(dmc_prev, dtype=np.float64)
    month = np.asarray(month, dtype=np.int32)

    # Temperature floor
    t_eff = np.maximum(temp, -1.1)

    po = dmc_prev.copy()

    # --- Rain correction (threshold 1.5 mm) ---
    has_rain = rain > 1.5
    if np.any(has_rain):
        re = 0.92 * rain - 1.27

        # Previous moisture content
        mo = 20.0 + 280.0 / np.exp(0.023 * po)

        # Slope coefficient b
        b = np.where(
            po <= 33.0,
            100.0 / (0.5 + 0.3 * po),
            np.where(
                po <= 65.0,
                14.0 - 1.3 * np.log(np.maximum(po, 1e-10)),
                6.2 * np.log(np.maximum(po, 1e-10)) - 17.2,
            ),
        )

        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 244.72 - 43.43 * np.log(np.maximum(mr - 20.0, 1e-10))
        pr = np.maximum(pr, 0.0)

        po = np.where(has_rain, pr, po)

    # --- Drying increment ---
    le = DMC_DAY_LENGTH[month - 1]  # day-length factor
    k = 1.894 * (t_eff + 1.1) * (100.0 - rh) * le * 1e-4
    k = np.maximum(k, 0.0)

    dmc = po + k
    return np.maximum(dmc, 0.0)


def compute_dc(temp, rain, month, dc_prev):
    """
    Compute Drought Code (DC).

    month: 1-12 integer (scalar or array).
    """
    temp = np.asarray(temp, dtype=np.float64)
    rain = np.asarray(rain, dtype=np.float64)
    dc_prev = np.asarray(dc_prev, dtype=np.float64)
    month = np.asarray(month, dtype=np.int32)

    # Temperature floor
    t_eff = np.maximum(temp, -2.8)

    do = dc_prev.copy()

    # --- Rain correction (threshold 2.8 mm) ---
    has_rain = rain > 2.8
    if np.any(has_rain):
        rd = 0.83 * rain - 1.27
        qo = 800.0 * np.exp(-do / 400.0)
        qr = qo + 3.937 * rd
        dr = 400.0 * np.log(np.maximum(800.0 / qr, 1e-10))
        dr = np.maximum(dr, 0.0)
        do = np.where(has_rain, dr, do)

    # --- Potential evapotranspiration ---
    lf = DC_DAY_LENGTH[month - 1]
    v = 0.36 * (t_eff + 2.8) + lf
    v = np.maximum(v, 0.0)

    dc = do + 0.5 * v
    return np.maximum(dc, 0.0)


def compute_isi(wind, ffmc):
    """Compute Initial Spread Index (ISI) from wind and FFMC."""
    wind = np.asarray(wind, dtype=np.float64)
    ffmc = np.asarray(ffmc, dtype=np.float64)

    m = _moisture_from_ffmc(ffmc)
    fw = np.exp(0.05039 * wind)
    ff = 91.9 * np.exp(-0.1386 * m) * (1.0 + np.power(m, 5.31) / 4.93e7)
    return 0.208 * fw * ff


def compute_bui(dmc, dc):
    """Compute Buildup Index (BUI) from DMC and DC."""
    dmc = np.asarray(dmc, dtype=np.float64)
    dc = np.asarray(dc, dtype=np.float64)

    # Handle zero case
    both_zero = (dmc <= 0) & (dc <= 0)

    # Standard formula: two branches
    ratio_ok = dmc <= 0.4 * dc
    u_low = 0.8 * dmc * dc / np.maximum(dmc + 0.4 * dc, 1e-10)
    u_high = dmc - (
        1.0 - 0.8 * dc / np.maximum(dmc + 0.4 * dc, 1e-10)
    ) * (0.92 + np.power(0.0114 * dmc, 1.7))

    bui = np.where(ratio_ok, u_low, u_high)
    bui = np.where(both_zero, 0.0, bui)
    return np.maximum(bui, 0.0)


def compute_fwi(isi, bui):
    """Compute Fire Weather Index (FWI) from ISI and BUI."""
    isi = np.asarray(isi, dtype=np.float64)
    bui = np.asarray(bui, dtype=np.float64)

    # Duff moisture function
    fd = np.where(
        bui <= 80.0,
        0.626 * np.power(bui, 0.809) + 2.0,
        1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)),
    )

    # Intermediate intensity
    b = 0.1 * isi * fd

    # Log transform (guard against negative base in fractional exponent)
    log_b = np.where(b > 1.0, 0.434 * np.log(np.maximum(b, 1e-10)), 0.0)
    fwi = np.where(
        b > 1.0,
        np.exp(2.72 * np.power(np.maximum(log_b, 0.0), 0.647)),
        b,
    )
    return np.maximum(fwi, 0.0)


def dewpoint_to_rh(temp_c, dewpoint_c):
    """
    Compute relative humidity (%) from temperature and dewpoint (both °C).
    Uses Magnus formula: es(T) = 6.112 * exp(17.67 * T / (T + 243.5))
    """
    temp_c = np.asarray(temp_c, dtype=np.float64)
    dewpoint_c = np.asarray(dewpoint_c, dtype=np.float64)
    es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
    ea = 6.112 * np.exp(17.67 * dewpoint_c / (dewpoint_c + 243.5))
    rh = 100.0 * ea / np.maximum(es, 1e-10)
    return np.clip(rh, 0.0, 100.0)


def wind_components_to_speed(u10, v10):
    """Compute wind speed (m/s) from u and v 10m components."""
    return np.sqrt(np.asarray(u10) ** 2 + np.asarray(v10) ** 2)


class FWICalculator:
    """
    Stateful FWI calculator that tracks FFMC/DMC/DC across days.

    Supports gridded (array) or scalar inputs. All arrays must broadcast.

    Example:
        calc = FWICalculator(shape=(100, 200))
        for day_idx in range(n_days):
            result = calc.update(temp[day_idx], rh[day_idx],
                                 wind[day_idx], rain[day_idx], month=6)
            fwi_map = result["FWI"]
    """

    def __init__(self, shape=None, ffmc0=DEFAULT_FFMC, dmc0=DEFAULT_DMC,
                 dc0=DEFAULT_DC):
        """
        shape: tuple for gridded data, or None for scalar.
        ffmc0/dmc0/dc0: initial values (scalar or array matching shape).
        """
        if shape is not None:
            self.ffmc = np.full(shape, ffmc0, dtype=np.float64)
            self.dmc = np.full(shape, dmc0, dtype=np.float64)
            self.dc = np.full(shape, dc0, dtype=np.float64)
        else:
            self.ffmc = np.float64(ffmc0)
            self.dmc = np.float64(dmc0)
            self.dc = np.float64(dc0)

    def set_state(self, ffmc, dmc, dc):
        """Override current moisture code state (e.g., from observations)."""
        self.ffmc = np.asarray(ffmc, dtype=np.float64)
        self.dmc = np.asarray(dmc, dtype=np.float64)
        self.dc = np.asarray(dc, dtype=np.float64)

    def update(self, temp, rh, wind, rain, month):
        """
        Advance one day.

        Args:
            temp:  temperature (°C)
            rh:    relative humidity (%)
            wind:  wind speed (km/h)
            rain:  24h precipitation (mm)
            month: calendar month (1-12)

        Returns dict with keys: FFMC, DMC, DC, ISI, BUI, FWI
        """
        self.ffmc = compute_ffmc(temp, rh, wind, rain, self.ffmc)
        self.dmc = compute_dmc(temp, rh, rain, month, self.dmc)
        self.dc = compute_dc(temp, rain, month, self.dc)
        isi = compute_isi(wind, self.ffmc)
        bui = compute_bui(self.dmc, self.dc)
        fwi = compute_fwi(isi, bui)
        return {
            "FFMC": self.ffmc.copy(),
            "DMC": self.dmc.copy(),
            "DC": self.dc.copy(),
            "ISI": isi,
            "BUI": bui,
            "FWI": fwi,
        }
