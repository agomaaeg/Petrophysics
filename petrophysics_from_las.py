import numpy as np
import pandas as pd
import lasio
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ============================================================================
#  UTILITY
# ============================================================================

def clip(a, lo=None, hi=None):
    """Clip array values to [lo, hi]."""
    a = a.copy()
    if lo is not None: a[a < lo] = lo
    if hi is not None: a[a > hi] = hi
    return a

# ============================================================================
#  1.  SHALE VOLUME  (Vsh)
# ============================================================================

def igr_linear(gr, gr_clean, gr_shale):
    """Linear Gamma Ray Index (IGR)."""
    igr = (gr - gr_clean) / (gr_shale - gr_clean)
    return clip(igr, 0, 1)

def vsh_linear(gr, gr_clean, gr_shale):
    """Vsh – Linear GR method."""
    return igr_linear(gr, gr_clean, gr_shale)

def vsh_larionov_tertiary(gr, gr_clean, gr_shale):
    """Vsh – Larionov (1969) for Tertiary / unconsolidated rocks."""
    igr = igr_linear(gr, gr_clean, gr_shale)
    vsh = 0.083 * (2**(3.7 * igr) - 1)
    return clip(vsh, 0, 1)

def vsh_larionov_older(gr, gr_clean, gr_shale):
    """Vsh – Larionov (1969) for older / consolidated rocks."""
    igr = igr_linear(gr, gr_clean, gr_shale)
    vsh = 0.33 * (2**(2.0 * igr) - 1)
    return clip(vsh, 0, 1)

def vsh_steiber(gr, gr_clean, gr_shale):
    """Vsh – Steiber (1970)."""
    igr = igr_linear(gr, gr_clean, gr_shale)
    vsh = igr / (3.0 - 2.0 * igr)
    return clip(vsh, 0, 1)

def vsh_clavier(gr, gr_clean, gr_shale):
    """Vsh – Clavier (1971)."""
    igr = igr_linear(gr, gr_clean, gr_shale)
    vsh = 1.7 - np.sqrt(3.38 - (igr + 0.7)**2)
    return clip(vsh, 0, 1)

def vsh_from_sp(sp, sp_clean, sp_shale):
    """Vsh from Spontaneous Potential (SP) log.
    sp_clean  = SP reading in clean formation
    sp_shale  = SP reading in shale
    """
    vsh = (sp - sp_clean) / (sp_shale - sp_clean)
    return clip(vsh, 0, 1)

def vsh_from_neutron_density(phin, phid, phin_sh, phid_sh):
    """Vsh from Neutron-Density crossplot separation.
    phin_sh, phid_sh = neutron/density porosity in shale.
    """
    vsh = (phin - phid) / (phin_sh - phid_sh)
    return clip(vsh, 0, 1)

# ============================================================================
#  2.  POROSITY
# ============================================================================

def phi_density(rhob, rhob_ma=2.65, rhob_fl=1.0):
    """Density porosity."""
    phid = (rhob_ma - rhob) / (rhob_ma - rhob_fl)
    return clip(phid, 0, 0.6)

def phi_neutron_from_nphi(nphi, unit="v/v"):
    """Neutron porosity (pass-through with optional unit conversion)."""
    x = nphi.copy()
    if unit.lower() in ["pu", "p.u.", "%", "percent"]:
        x = x / 100.0
    return clip(x, 0, 0.6)

def phi_sonic_wyllie(dt, dt_ma=55.5, dt_fl=189.0, cp=1.0):
    """Sonic (Wyllie Time Average) porosity.
    dt_ma : matrix transit time (µs/ft)  – Sandstone ~55.5, Limestone ~47.5, Dolomite ~43.5
    dt_fl : fluid transit time (µs/ft)   – ~189 (fresh), ~185 (salt)
    cp    : compaction correction factor  – typically 1.0 for consolidated
    """
    phis = (dt - dt_ma) / (dt_fl - dt_ma) * (1.0 / cp)
    return clip(phis, 0, 0.6)

def phi_sonic_rhg(dt, dt_ma=55.5, alpha=0.625):
    """Sonic porosity – Raymer-Hunt-Gardner (1980).
    alpha : empirical constant, 0.625 (default) or 0.67.
    """
    phis = alpha * (dt - dt_ma) / dt
    return clip(phis, 0, 0.6)

def phi_total(phid, phin, method="avg"):
    """Total porosity (PHIT) from Density and Neutron porosities."""
    if method == "avg":
        return clip(0.5 * (phid + phin), 0, 0.6)
    elif method == "rms":
        return clip(np.sqrt(0.5 * (phid**2 + phin**2)), 0, 0.6)
    elif method == "density":
        return phid
    elif method == "neutron":
        return phin
    else:
        raise ValueError(f"Unknown method: {method}")

def phi_effective(phi_t, vsh, phi_sh=0.10):
    """Effective porosity: PHIE = PHIT – Vsh × φ_shale."""
    phie = phi_t - vsh * phi_sh
    return clip(phie, 0, 0.6)

def phi_secondary(phin, phis):
    """Secondary (vuggy/fracture) porosity index = φN – φS."""
    return clip(phin - phis, 0, 0.6)

def phi_neutron_density_gas_corrected(phin, phid):
    """Gas-corrected porosity from Neutron-Density crossplot.
    Uses the equation:  φ = sqrt( (φN² + φD²) / 2 )
    """
    phig = np.sqrt((phin**2 + phid**2) / 2.0)
    return clip(phig, 0, 0.6)

# ============================================================================
#  3.  WATER SATURATION  (Sw)
# ============================================================================

def sw_archie(rt, rw, phi, a=1.0, m=2.0, n=2.0):
    """Archie (1942) water saturation.
    Sw = (a · Rw / (Rt · φ^m))^(1/n)
    """
    sw = (a * rw / (rt * np.power(phi, m))) ** (1.0 / n)
    return clip(sw, 0, 1)

def sw_simandoux(rt, rw, phi, vsh, rsh, a=1.0, m=2.0, n=2.0):
    """Simandoux (1963) – shaly sand Sw.
    rsh : resistivity of pure shale
    """
    C = (1.0 - vsh) * a * rw / np.power(phi, m)
    D = C * vsh / (2.0 * rsh)
    E = C / rt
    # Quadratic: Sw^n = -D + sqrt(D² + E)
    sw = (-D + np.sqrt(D**2 + E))
    sw = np.power(sw, 1.0 / n) if n != 1 else sw
    return clip(sw, 0, 1)

def sw_modified_simandoux(rt, rw, phi, vsh, rsh, a=1.0, m=2.0, n=2.0):
    """Modified Simandoux (Bardon & Pied 1969).
    Better correction for higher Vsh.
    """
    F = a * rw / (np.power(phi, m) * (1 - vsh))
    alpha = F * vsh / (2.0 * rsh)
    beta = F / rt
    sw = (-alpha + np.sqrt(alpha**2 + beta))
    sw = np.power(np.abs(sw), 1.0 / n)
    return clip(sw, 0, 1)

def sw_indonesia(rt, rw, phi, vsh, rsh, a=1.0, m=2.0, n=2.0):
    """Indonesia / Poupon-Leveaux (1971) equation – shaly sands.
    Widely used for SE Asian and Middle Eastern reservoirs.
    """
    A_term = np.sqrt(1.0 / rt)
    B_term = np.power(vsh, (1.0 - 0.5 * vsh)) / np.sqrt(rsh)
    C_term = np.sqrt(np.power(phi, m) / (a * rw))
    sw_inv = B_term + C_term
    sw = np.power(A_term / sw_inv, 2.0 / n)
    return clip(sw, 0, 1)

def sw_dual_water(rt, rw, rwb, phi_t, phi_e, a=1.0, m=2.0, n=2.0):
    """Dual Water model (Clavier et al. 1984).
    rwb : bound water resistivity (~0.3× Rw at reservoir temperature)
    """
    swb = (phi_t - phi_e) / phi_t  # Bound water saturation
    swb = clip(swb, 0, 1)
    # Total water saturation from dual water
    # 1/Rt = (φt^m / a) * [ Swt^n / Rw + Swb*(1/Rwb - 1/Rw) ]
    # Simplified iterative: initial guess Swt from Archie on PHIT
    swt_init = sw_archie(rt, rw, phi_t, a=a, m=m, n=n)
    # One-pass correction
    rw_eff = 1.0 / (1.0/rw + swb * (1.0/rwb - 1.0/rw))
    swt = sw_archie(rt, rw_eff, phi_t, a=a, m=m, n=n)
    # Effective Sw
    swe = (swt * phi_t - swb * phi_t) / phi_e
    swe = np.where(phi_e > 0.01, swe, np.nan)
    return clip(swe, 0, 1)

def sw_waxman_smits(rt, rw, phi, qv, B, a=1.0, m_star=2.0, n_star=2.0):
    """Waxman-Smits (1968) model.
    qv : cation exchange capacity per pore volume (meq/mL)
    B  : equivalent conductance of clay counter-ions (mho·cm²/meq)
         ~ 0.046 × (1 – 0.6 × exp(–0.77 / Rw))  at reservoir temp
    """
    # F* = a / φ^m*
    F_star = a / np.power(phi, m_star)
    # 1/Rt = (Sw^n* / F* · Rw) × (1 + Rw · B · Qv / Sw)
    # Iterative: start with Archie
    sw = sw_archie(rt, rw, phi, a=a, m=m_star, n=n_star)
    for _ in range(5):
        rw_eff = rw / (1.0 + rw * B * qv / sw)
        sw = sw_archie(rt, rw_eff, phi, a=a, m=m_star, n=n_star)
    return clip(sw, 0, 1)

# ============================================================================
#  4.  PERMEABILITY  (mD)
# ============================================================================

def perm_timur(phi, swir):
    """Timur (1968) permeability (mD).
    K = 0.136 × φ^4.4 / Swir²      (φ in fraction)
    """
    k = 0.136 * np.power(phi, 4.4) / np.power(swir, 2.0)
    return clip(k, 0, None)

def perm_coates(phi, swir):
    """Coates / Coates-Dumanoir permeability (mD).
    K = (100 × φ² × ((1-Swir)/Swir))²
    """
    ratio = (1.0 - swir) / swir
    k = (100.0 * phi**2 * ratio) ** 2
    return clip(k, 0, None)

def perm_morris_biggs_oil(phi, swir):
    """Morris-Biggs (1967) – Oil zone.
    K = 62.5 × (φ³ / Swir)²
    """
    k = 62.5 * (phi**3 / swir) ** 2
    return clip(k, 0, None)

def perm_morris_biggs_gas(phi, swir):
    """Morris-Biggs (1967) – Gas zone.
    K = 2.5 × (φ³ / Swir)²
    """
    k = 2.5 * (phi**3 / swir) ** 2
    return clip(k, 0, None)

def perm_tixier(phi, swir):
    """Tixier (1949) permeability (mD).
    K = (250 × φ³ / Swir)²
    """
    k = (250.0 * phi**3 / swir) ** 2
    return clip(k, 0, None)

def perm_kozeny_carman(phi, svgr=None, tau=2.5):
    """Kozeny-Carman equation.
    svgr : specific surface area per grain volume (1/µm). Default None → uses generic.
    tau  : tortuosity (default 2.5)
    If svgr is not given, a generic approximation is used.
    """
    if svgr is None:
        svgr = 1.0  # placeholder – user must provide
    k = (phi ** 3) / (tau * svgr**2 * (1 - phi)**2) * 1e6  # approx mD
    return clip(k, 0, None)

# ============================================================================
#  5.  ROCK TYPING
# ============================================================================

def winland_r35(phi, k):
    """Winland R35 pore throat radius (µm).
    log(R35) = 0.732 + 0.588·log(K) – 0.864·log(φ)
    φ in %, K in mD.
    """
    phi_pct = phi * 100.0
    phi_pct = np.where(phi_pct > 0, phi_pct, np.nan)
    k_safe = np.where(k > 0, k, np.nan)
    log_r35 = 0.732 + 0.588 * np.log10(k_safe) - 0.864 * np.log10(phi_pct)
    return 10 ** log_r35

def flow_zone_indicator(phi, k):
    """Flow Zone Indicator (FZI) – Amaefule et al. (1993).
    FZI = RQI / φz     where RQI = 0.0314·√(K/φ),  φz = φ/(1-φ)
    φ in fraction, K in mD.
    """
    rqi = 0.0314 * np.sqrt(k / phi)
    phi_z = phi / (1.0 - phi)
    fzi = rqi / phi_z
    return fzi

def lucia_rfn(phi, k):
    """Lucia Rock Fabric Number (RFN) – carbonate classification.
    RFN ≈ K / (φ^(6.5)) × 7.7e-6  — simplified empirical.
    Use mainly for classifying carbonates into Class 1/2/3.
    """
    rfn = (k / np.power(phi, 6.5)) * 7.7e-6
    return rfn

# ============================================================================
#  6.  FLUID PROPERTIES & TEMPERATURE
# ============================================================================

def formation_temperature(depth, ts=70.0, grad=1.2, depth_unit="ft", temp_unit="F"):
    """Formation temperature from geothermal gradient.
    ts   : surface temperature (°F)
    grad : geothermal gradient (°F/100ft)
    """
    if depth_unit == "m":
        depth_ft = depth * 3.28084
    else:
        depth_ft = depth
    tf = ts + grad * depth_ft / 100.0
    if temp_unit == "C":
        tf = (tf - 32) * 5.0 / 9.0
    return tf

def rw_temperature_correction(rw_meas, t_meas, t_form):
    """Arps (1953) formula for Rw temperature correction.
    All temperatures in °F.
    Rw2 = Rw1 × (T1 + 6.77) / (T2 + 6.77)
    """
    return rw_meas * (t_meas + 6.77) / (t_form + 6.77)

def rw_from_sp(sp, ssp, rmf, t_form):
    """Estimate Rw from SP log.
    sp   : SP reading (mV)
    ssp  : Static SP (mV) – max SP deflection
    rmf  : mud filtrate resistivity at formation temperature
    t_form : formation temperature (°F)
    Returns Rw (ohm·m).
    """
    # Rmfe from Rmf (if Rmf > 0.1)
    rmfe = 0.85 * rmf if rmf > 0.1 else (146 * rmf - 5) / (377 * rmf + 77)
    # SP = -K × log(Rmfe/Rwe)   where K = 61 + 0.133×Tf
    K = 61 + 0.133 * t_form
    rwe = rmfe / (10 ** (ssp / K))
    # Rw from Rwe (if Rwe > 0.12)
    rw = rwe if rwe > 0.12 else (77 * rwe + 5) / (-146 * rwe + 377)
    return rw

def rw_nacl(salinity_ppm, temp_f):
    """Approximate Rw from NaCl salinity (ppm) and temperature (°F).
    Based on Schlumberger Gen-9 chart approximation.
    """
    # Simplified: Rw ≈ (400000 / (salinity × (temp_F + 6.77))) at 77°F ref
    rw = (400000.0 / salinity_ppm) * (77.0 + 6.77) / (temp_f + 6.77)
    return rw

# ============================================================================
#  7.  MECHANICAL ROCK PROPERTIES  (from Sonic / Density)
# ============================================================================

def dynamic_youngs_modulus(rhob, dts, dtc):
    """Dynamic Young's Modulus (GPa) from logs.
    rhob : bulk density (g/cc)
    dts  : shear slowness (µs/ft)
    dtc  : compressional slowness (µs/ft)
    """
    vs = 304800.0 / dts  # m/s (1 ft = 304800 µm)
    vp = 304800.0 / dtc
    pr = poisson_ratio(dts, dtc)
    E = rhob * 1000 * vs**2 * (3 * vp**2 - 4 * vs**2) / (vp**2 - vs**2)
    E_gpa = E / 1e9
    return E_gpa

def poisson_ratio(dts, dtc):
    """Dynamic Poisson's Ratio from sonic logs.
    Returns dimensionless ν.
    """
    ratio2 = (dts / dtc) ** 2
    nu = (ratio2 - 2) / (2 * (ratio2 - 1))
    return clip(nu, 0, 0.5)

def bulk_modulus(rhob, dtc, dts):
    """Bulk Modulus K (GPa) from logs."""
    vp = 304800.0 / dtc
    vs = 304800.0 / dts
    K = rhob * 1000 * (vp**2 - (4.0/3.0) * vs**2)
    return K / 1e9

def shear_modulus(rhob, dts):
    """Shear Modulus G (GPa) from logs."""
    vs = 304800.0 / dts
    G = rhob * 1000 * vs**2
    return G / 1e9

def ucs_from_E(E_gpa, lithology="sandstone"):
    """Unconfined Compressive Strength (MPa) estimated from Young's Modulus.
    Empirical correlations (approximate).
    """
    if lithology.lower() == "sandstone":
        return 2.28 + 4.1089 * E_gpa        # McNally (1987)
    elif lithology.lower() == "shale":
        return 0.77 * E_gpa ** 0.93          # Horsrud (2001)
    elif lithology.lower() in ["limestone", "carbonate"]:
        return 13.7 * E_gpa ** 0.51          # Militzer & Stoll (1973)
    else:
        return 2.28 + 4.1089 * E_gpa

# ============================================================================
#  8.  HYDROCARBON VOLUMES & PAY FLAGS
# ============================================================================

def bulk_volume_water(sw, phi):
    """Bulk Volume Water: BVW = Sw × φ."""
    return sw * phi

def bulk_volume_hydrocarbon(sw, phi):
    """Bulk Volume Hydrocarbon: BVH = (1 – Sw) × φ."""
    return (1 - sw) * phi

def hydrocarbon_pore_volume(phi_e, sw, h_net, area_acres=None):
    """Hydrocarbon Pore Volume.
    HCPV_fraction = PHIE × (1 – Sw)    (per unit thickness)
    If area_acres provided → returns volume in acre-ft.
    h_net : net pay thickness array (ft)
    """
    hcpv = phi_e * (1 - sw)
    if area_acres is not None:
        # Summation over net pay interval
        return np.nansum(hcpv * h_net) * area_acres  # acre-ft
    return hcpv

def pay_flag(vsh, phie, sw, vsh_cut=0.35, phie_cut=0.08, sw_cut=0.60):
    """Simple pay flag: 1 = pay, 0 = non-pay."""
    flag = ((vsh <= vsh_cut) & (phie >= phie_cut) & (sw <= sw_cut)).astype(int)
    return flag

def net_to_gross(pay_flag_arr):
    """Net-to-Gross ratio from pay flag array."""
    total = np.sum(~np.isnan(pay_flag_arr.astype(float)))
    net = np.nansum(pay_flag_arr)
    return net / total if total > 0 else 0.0

def net_pay_summary(df, depth_col="DEPT", pay_col="PAY_FLAG",
                    phi_col="PHIE", sw_col="SW", vsh_col="VSH"):
    """Calculate net pay summary statistics from a DataFrame."""
    pay = df[df[pay_col] == 1].copy()
    if pay.empty:
        return {"Net_Pay_ft": 0, "Avg_PHIE": 0, "Avg_Sw": 0, "Avg_Vsh": 0, "NTG": 0}

    step = np.abs(np.median(np.diff(df.index.values)))
    summary = {
        "Net_Pay_ft": len(pay) * step,
        "Avg_PHIE": pay[phi_col].mean(),
        "Avg_Sw": pay[sw_col].mean(),
        "Avg_Vsh": pay[vsh_col].mean(),
        "NTG": len(pay) / len(df),
        "Avg_BVW": (pay[sw_col] * pay[phi_col]).mean(),
        "Avg_BVH": ((1 - pay[sw_col]) * pay[phi_col]).mean(),
    }
    return summary

# ============================================================================
#  9.  LITHOLOGY IDENTIFICATION (Crossplot helpers)
# ============================================================================

def lithology_from_rhob_nphi(rhob, nphi):
    """Simple lithology guess from RHOB-NPHI crossplot position.
    Returns a string label per sample.
    """
    labels = []
    for r, n in zip(rhob, nphi):
        if np.isnan(r) or np.isnan(n):
            labels.append("Unknown")
        elif r >= 2.60 and n < 0.05:
            labels.append("Tight SS / Anhydrite")
        elif r >= 2.50 and n < 0.12:
            labels.append("Sandstone")
        elif 2.40 <= r < 2.60 and 0.05 <= n < 0.20:
            labels.append("Limestone")
        elif 2.70 <= r and 0.02 <= n < 0.10:
            labels.append("Dolomite")
        elif r < 2.30:
            labels.append("Coal / Low Density")
        elif n > 0.30:
            labels.append("Shale")
        else:
            labels.append("Mixed / Uncertain")
    return labels

def m_n_litho(phin, phid, dt, dt_fl=189, rhob_fl=1.0, dt_ma=55.5, rhob_ma=2.65):
    """M-N crossplot values for mineral identification.
    M = (dt_fl – dt) / (ρb – ρfl) × 0.01
    N = (φNfl – φN) / (ρb – ρfl)
    """
    M = 0.01 * (dt_fl - dt) / (rhob_ma - rhob_fl)
    N = (1.0 - phin) / (rhob_ma - rhob_fl)  # simplified
    return M, N

# ============================================================================
#  10.  CROSSPLOT SUPPORT (Pickett, Hingle)
# ============================================================================

def pickett_plot_data(phi, rt, rw, a=1.0, m=2.0):
    """Prepare data for a Pickett Plot (log Rt vs log φ).
    Returns phi, rt, and Sw=100% line coordinates.
    """
    phi_range = np.linspace(0.01, 0.50, 100)
    rt_sw100 = a * rw / np.power(phi_range, m)
    return phi_range, rt_sw100

def hingle_plot_data(phi, rt, rw, a=1.0, m=2.0):
    """Prepare data for a Hingle Plot (1/√Rt vs φ).
    Returns 1/sqrt(Rt), phi, and Sw=100% line.
    """
    inv_sqrt_rt = 1.0 / np.sqrt(rt)
    phi_range = np.linspace(0.01, 0.50, 100)
    # At Sw=100%: 1/√Rt = √(φ^m / (a·Rw))
    inv_sqrt_rt_100 = np.sqrt(np.power(phi_range, m) / (a * rw))
    return inv_sqrt_rt, phi_range, inv_sqrt_rt_100

# ============================================================================
#  MAIN WORKFLOW
# ============================================================================

def main(
    las_path,
    out_csv="petro_output.csv",
    out_las="petro_output.las",
    # Curve mnemonics (adjust for each LAS file)
    gr_mn="GR",
    rt_mn="AHT90",      # Deep resistivity
    rxo_mn="AHT10",     # Shallow / Rxo
    rhob_mn="RHOZ",     # Bulk density
    nphi_mn="TNPH",     # Neutron porosity
    dt_mn="DT",         # Compressional sonic
    dts_mn="DTS",       # Shear sonic
    pe_mn="PEFZ",       # Photoelectric Factor
    cal_mn="HCAL",      # Caliper
    bs_mn="BS",         # Bit Size
    sp_mn="SP",         # Spontaneous Potential
    # ---- Parameters (ADJUST FOR YOUR WELL) ----
    gr_clean=30.0,
    gr_shale=120.0,
    vsh_method="linear",    # "linear", "larionov_tertiary", "larionov_older", "steiber", "clavier"
    rhob_ma=2.65,           # Sandstone matrix density
    rhob_fl=1.00,           # Fluid density
    dt_ma=55.5,             # Sandstone matrix DT
    dt_fl=189.0,            # Fluid DT
    nphi_unit="v/v",
    phi_sh=0.10,            # Shale porosity
    rw=0.03,                # Formation water resistivity (at formation temp)
    rsh=3.0,                # Shale resistivity
    a=1.0, m=2.0, n=2.0,   # Archie parameters
    sw_model="archie",      # "archie", "simandoux", "mod_simandoux", "indonesia", "dual_water"
    phi_method="avg",       # "avg", "rms", "density", "neutron"
    perm_model="timur",     # "timur", "coates", "morris_oil", "morris_gas", "tixier"
    # Pay cutoffs
    vsh_cut=0.35,
    phie_cut=0.08,
    sw_cut=0.60,
    # Temperature
    surface_temp=70.0,
    temp_gradient=1.2,      # °F / 100 ft
):
    print(f"Reading LAS file: {las_path}...")
    las = lasio.read(las_path)

    # Convert to DataFrame with depth index
    df = las.df()
    df.index.name = "DEPT"

    # Normalize common nulls
    df = df.replace([-999.25, -999.0, -9999.0], np.nan)

    # Check required curves
    missing = []
    for mn in [gr_mn, rt_mn, rhob_mn, nphi_mn]:
        if mn not in df.columns:
            missing.append(mn)
    if missing:
        raise KeyError(f"Missing required curves in LAS: {missing}. Available: {list(df.columns)}")

    # ----- Inputs -----
    gr   = df[gr_mn].astype(float)
    rt   = df[rt_mn].astype(float)
    rhob = df[rhob_mn].astype(float)
    nphi = df[nphi_mn].astype(float)

    # Optional curves
    has_rxo = rxo_mn in df.columns
    has_dt  = dt_mn in df.columns
    has_dts = dts_mn in df.columns
    has_pe  = pe_mn in df.columns
    has_cal = cal_mn in df.columns
    has_bs  = bs_mn in df.columns
    has_sp  = sp_mn in df.columns

    rxo = df[rxo_mn].astype(float) if has_rxo else None
    dt  = df[dt_mn].astype(float) if has_dt else None
    dts = df[dts_mn].astype(float) if has_dts else None
    pe  = df[pe_mn].astype(float) if has_pe else None
    cal = df[cal_mn].astype(float) if has_cal else None
    bs  = df[bs_mn].astype(float) if has_bs else None
    sp  = df[sp_mn].astype(float) if has_sp else None

    depth_vals = df.index.values

    # ===================================================================
    # Formation Temperature
    # ===================================================================
    df["TEMP_F"] = formation_temperature(depth_vals, ts=surface_temp, grad=temp_gradient)

    # ===================================================================
    # 1. Vsh
    # ===================================================================
    vsh_methods = {
        "linear": vsh_linear,
        "larionov_tertiary": vsh_larionov_tertiary,
        "larionov_older": vsh_larionov_older,
        "steiber": vsh_steiber,
        "clavier": vsh_clavier,
    }
    vsh_func = vsh_methods.get(vsh_method, vsh_linear)
    df["VSH"] = vsh_func(gr, gr_clean, gr_shale)
    print(f"  Vsh method: {vsh_method}")

    # ===================================================================
    # 2. Porosity
    # ===================================================================
    df["PHID"] = phi_density(rhob, rhob_ma=rhob_ma, rhob_fl=rhob_fl)
    df["PHIN"] = phi_neutron_from_nphi(nphi, unit=nphi_unit)
    if has_dt:
        df["PHIS_WYLLIE"] = phi_sonic_wyllie(dt, dt_ma=dt_ma, dt_fl=dt_fl)
        df["PHIS_RHG"] = phi_sonic_rhg(dt, dt_ma=dt_ma)
    df["PHIT"] = phi_total(df["PHID"], df["PHIN"], method=phi_method)
    df["PHIE"] = phi_effective(df["PHIT"], df["VSH"], phi_sh=phi_sh)
    df["PHIG"] = phi_neutron_density_gas_corrected(df["PHIN"], df["PHID"])  # gas-corrected
    if has_dt:
        df["SPI"] = phi_secondary(df["PHIN"], df["PHIS_WYLLIE"])  # secondary porosity index
    print(f"  Porosity method: {phi_method}")

    # ===================================================================
    # 3. Water Saturation
    # ===================================================================
    phi_for_sw = df["PHIE"].where(df["PHIE"] > 0.02)
    rt_for_sw  = rt.where(rt > 0.01)

    if sw_model == "archie":
        df["SW"] = sw_archie(rt_for_sw, rw=rw, phi=phi_for_sw, a=a, m=m, n=n)
    elif sw_model == "simandoux":
        df["SW"] = sw_simandoux(rt_for_sw, rw=rw, phi=phi_for_sw,
                                vsh=df["VSH"], rsh=rsh, a=a, m=m, n=n)
    elif sw_model == "mod_simandoux":
        df["SW"] = sw_modified_simandoux(rt_for_sw, rw=rw, phi=phi_for_sw,
                                         vsh=df["VSH"], rsh=rsh, a=a, m=m, n=n)
    elif sw_model == "indonesia":
        df["SW"] = sw_indonesia(rt_for_sw, rw=rw, phi=phi_for_sw,
                                vsh=df["VSH"], rsh=rsh, a=a, m=m, n=n)
    elif sw_model == "dual_water":
        rwb = rw * 0.3  # approximate
        df["SW"] = sw_dual_water(rt_for_sw, rw=rw, rwb=rwb,
                                 phi_t=df["PHIT"], phi_e=df["PHIE"], a=a, m=m, n=n)
    else:
        df["SW"] = sw_archie(rt_for_sw, rw=rw, phi=phi_for_sw, a=a, m=m, n=n)
    print(f"  Sw model: {sw_model}")

    # ===================================================================
    # 4. BVW, BVH
    # ===================================================================
    df["BVW"] = bulk_volume_water(df["SW"], df["PHIE"])
    df["BVH"] = bulk_volume_hydrocarbon(df["SW"], df["PHIE"])

    # ===================================================================
    # 5. Permeability
    # ===================================================================
    swir = df["SW"].copy()  # Approximation: use Sw as Swir where available
    swir = swir.where(swir > 0.05, 0.05)  # avoid division by zero
    phi_for_perm = df["PHIE"].where(df["PHIE"] > 0.01, np.nan)

    perm_funcs = {
        "timur": perm_timur,
        "coates": perm_coates,
        "morris_oil": perm_morris_biggs_oil,
        "morris_gas": perm_morris_biggs_gas,
        "tixier": perm_tixier,
    }
    perm_func = perm_funcs.get(perm_model, perm_timur)
    df["PERM"] = perm_func(phi_for_perm, swir)
    print(f"  Permeability model: {perm_model}")

    # ===================================================================
    # 6. Rock Typing (where perm is valid)
    # ===================================================================
    mask_rt = (df["PHIE"] > 0.01) & (df["PERM"] > 0.001)
    df["R35"] = np.nan
    df.loc[mask_rt, "R35"] = winland_r35(df.loc[mask_rt, "PHIE"], df.loc[mask_rt, "PERM"])
    df["FZI"] = np.nan
    df.loc[mask_rt, "FZI"] = flow_zone_indicator(df.loc[mask_rt, "PHIE"], df.loc[mask_rt, "PERM"])

    # ===================================================================
    # 7. Mechanical Properties (if DT & DTS available)
    # ===================================================================
    if has_dt and has_dts:
        print("  Computing mechanical properties from sonic logs...")
        df["POISSON"] = poisson_ratio(dts, dt)
        df["YOUNG_GPa"] = dynamic_youngs_modulus(rhob, dts, dt)
        df["BULK_MOD_GPa"] = bulk_modulus(rhob, dt, dts)
        df["SHEAR_MOD_GPa"] = shear_modulus(rhob, dts)

    # ===================================================================
    # 8. Pay Flag & Summary
    # ===================================================================
    df["PAY_FLAG"] = pay_flag(df["VSH"], df["PHIE"], df["SW"],
                              vsh_cut=vsh_cut, phie_cut=phie_cut, sw_cut=sw_cut)

    summary = net_pay_summary(df, pay_col="PAY_FLAG")
    print("\n" + "="*50)
    print("  NET PAY SUMMARY")
    print("="*50)
    for k, v in summary.items():
        print(f"    {k:15s}: {v:.4f}" if isinstance(v, float) else f"    {k:15s}: {v}")
    print("="*50 + "\n")

    # ===================================================================
    # EXPORT
    # ===================================================================
    # CSV
    print(f"Exporting CSV to {out_csv}...")
    df.to_csv(out_csv, float_format="%.6f")

    # LAS
    print(f"Exporting LAS to {out_las}...")
    out = las
    computed_cols = [c for c in df.columns if c not in [mn for mn in las.keys()]]
    for col in computed_cols:
        data = df[col].values
        found = False
        for curve in out.curves:
            if curve.mnemonic == col:
                curve.data = data
                found = True
                break
        if not found:
            out.append_curve(col, data, unit="", descr=f"Computed {col}")
    out.write(out_las, version=2.0)

    # ===================================================================
    # COMPREHENSIVE QC PLOT  (7 Tracks)
    # ===================================================================
    print("Generating QC plot...")
    n_tracks = 7
    fig, axes = plt.subplots(1, n_tracks, figsize=(22, 14), sharey=True)
    depth = df.index.values

    # ---- Track 1: GR & Caliper ----
    ax = axes[0]
    ax.plot(gr, depth, color='green', lw=0.8, label='GR')
    ax.set_xlabel("GR (API)")
    ax.set_xlim(0, 150)
    ax.axvline(x=gr_clean, color='green', ls=':', alpha=0.5)
    ax.axvline(x=gr_shale, color='brown', ls=':', alpha=0.5)
    ax.fill_betweenx(depth, gr_clean, gr.clip(gr_clean, gr_shale),
                     where=(gr > gr_clean), color='khaki', alpha=0.4)
    ax.legend(loc='upper left', fontsize='x-small')
    if has_cal or has_bs:
        ax_tw = ax.twiny()
        if has_cal: ax_tw.plot(cal, depth, color='black', lw=0.6, ls='--', label='CAL')
        if has_bs:  ax_tw.plot(bs, depth, color='gray', lw=0.6, ls=':', label='BS')
        ax_tw.set_xlabel("Caliper (in)")
        ax_tw.set_xlim(6, 16)
        ax_tw.legend(loc='upper right', fontsize='x-small')

    # ---- Track 2: Resistivity ----
    ax = axes[1]
    ax.semilogx(rt, depth, color='red', lw=0.8, label=rt_mn)
    if has_rxo:
        ax.semilogx(rxo, depth, color='blue', lw=0.6, ls=':', label=rxo_mn)
    ax.set_xlabel("Resistivity (Ω·m)")
    ax.set_xlim(0.2, 2000)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.legend(loc='upper right', fontsize='x-small')

    # ---- Track 3: Density-Neutron ----
    ax = axes[2]
    ax.plot(rhob, depth, color='red', lw=0.8, label='RHOB')
    ax.set_xlabel("RHOB (g/cc)")
    ax.set_xlim(1.95, 2.95)
    ax_tw = ax.twiny()
    ax_tw.plot(nphi, depth, color='blue', lw=0.8, ls='--', label='NPHI')
    ax_tw.set_xlabel("NPHI (v/v)")
    ax_tw.set_xlim(0.45, -0.15)
    ax.legend(loc='upper left', fontsize='x-small')
    ax_tw.legend(loc='upper right', fontsize='x-small')

    # ---- Track 4: Computed Porosity ----
    ax = axes[3]
    ax.plot(df["PHIE"], depth, color='black', lw=1.0, label='PHIE')
    ax.plot(df["PHIT"], depth, color='gray', lw=0.6, ls='--', label='PHIT')
    if has_dt and "PHIS_WYLLIE" in df.columns:
        ax.plot(df["PHIS_WYLLIE"], depth, color='orange', lw=0.6, ls=':', label='PHIS')
    ax.set_xlabel("Porosity (v/v)")
    ax.set_xlim(0, 0.40)
    ax.legend(loc='upper left', fontsize='x-small')

    # ---- Track 5: Sw & BVW ----
    ax = axes[4]
    ax.plot(df["SW"], depth, color='blue', lw=0.8, label='Sw')
    ax.fill_betweenx(depth, 1, df["SW"], where=(df["SW"] < 1),
                     color='green', alpha=0.25, label='HC')
    ax.fill_betweenx(depth, 0, df["SW"], where=(df["SW"] > 0),
                     color='cyan', alpha=0.15, label='Water')
    ax.set_xlabel("Sw (v/v)")
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right', fontsize='x-small')

    # ---- Track 6: Vsh ----
    ax = axes[5]
    ax.plot(df["VSH"], depth, color='brown', lw=0.8, label='Vsh')
    ax.fill_betweenx(depth, 0, df["VSH"], color='olive', alpha=0.3)
    ax.axvline(x=vsh_cut, color='red', ls='--', lw=0.6, label=f'Cut={vsh_cut}')
    ax.set_xlabel("Vsh (v/v)")
    ax.set_xlim(0, 1)
    ax.legend(loc='upper right', fontsize='x-small')

    # ---- Track 7: Permeability ----
    ax = axes[6]
    ax.semilogx(df["PERM"], depth, color='purple', lw=0.8, label='K')
    ax.set_xlabel("Perm (mD)")
    ax.set_xlim(0.01, 10000)
    ax.grid(True, which='both', ls='-', alpha=0.2)
    ax.legend(loc='upper right', fontsize='x-small')

    # ---- Common settings ----
    for a in axes:
        a.invert_yaxis()
        a.grid(True, alpha=0.3)
        a.tick_params(axis='x', rotation=45, labelsize=7)
    axes[0].set_ylabel("DEPTH (ft)")

    # Pay flag shading
    for a in axes:
        a.fill_betweenx(depth, a.get_xlim()[0], a.get_xlim()[1],
                         where=(df["PAY_FLAG"] == 1),
                         color='gold', alpha=0.08)

    fig.suptitle(f"Petrophysical Evaluation — {las_path}", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("petro_composite_plot.png", dpi=200, bbox_inches='tight')
    plt.show()

    # ===================================================================
    # CROSSPLOTS
    # ===================================================================
    print("Generating crossplots...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    # Pickett Plot
    ax = axes2[0]
    valid = (df["PHIE"] > 0.01) & (rt > 0.01) & (~np.isnan(df["PHIE"])) & (~np.isnan(rt))
    sc = ax.scatter(df.loc[valid, "PHIE"], rt[valid], c=df.loc[valid, "SW"],
                    cmap='coolwarm_r', s=4, alpha=0.6, vmin=0, vmax=1)
    phi_line, rt_line = pickett_plot_data(df["PHIE"], rt, rw, a=a, m=m)
    ax.plot(phi_line, rt_line, 'k--', lw=1, label='Sw=100%')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel("PHIE (v/v)"); ax.set_ylabel("Rt (Ω·m)")
    ax.set_title("Pickett Plot")
    ax.legend(fontsize='small')
    plt.colorbar(sc, ax=ax, label='Sw')

    # RHOB-NPHI Crossplot
    ax = axes2[1]
    valid2 = (~np.isnan(rhob)) & (~np.isnan(nphi))
    sc2 = ax.scatter(nphi[valid2], rhob[valid2], c=df.loc[valid2, "VSH"],
                     cmap='YlOrBr', s=4, alpha=0.6, vmin=0, vmax=1)
    ax.set_xlabel("NPHI (v/v)"); ax.set_ylabel("RHOB (g/cc)")
    ax.set_xlim(-0.05, 0.50); ax.set_ylim(3.0, 1.8)
    ax.set_title("Neutron-Density Crossplot")
    # Reference lines (approx mineral endpoints)
    ax.plot([0, 0.45], [2.65, 1.0], 'b--', lw=0.8, alpha=0.5, label='SS line')
    ax.plot([0, 0.45], [2.71, 1.0], 'g--', lw=0.8, alpha=0.5, label='LS line')
    ax.legend(fontsize='small')
    plt.colorbar(sc2, ax=ax, label='Vsh')

    # PHI vs K crossplot (if perm exists)
    ax = axes2[2]
    valid3 = (df["PHIE"] > 0.01) & (df["PERM"] > 0.001)
    if valid3.any():
        sc3 = ax.scatter(df.loc[valid3, "PHIE"], df.loc[valid3, "PERM"],
                         c=df.loc[valid3, "R35"], cmap='jet', s=6, alpha=0.6,
                         norm=plt.matplotlib.colors.LogNorm(vmin=0.1, vmax=100))
        ax.set_yscale('log')
        ax.set_xlabel("PHIE (v/v)"); ax.set_ylabel("Perm (mD)")
        ax.set_title("Porosity-Permeability (color=R35)")
        plt.colorbar(sc3, ax=ax, label='R35 (µm)')
    else:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Porosity-Permeability")

    fig2.suptitle("Petrophysical Crossplots", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("petro_crossplots.png", dpi=200, bbox_inches='tight')
    plt.show()

    print("Done! ✓")
    return df

# ============================================================================
#  ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Example usage:
    # df = main("well.las", vsh_method="larionov_tertiary", sw_model="indonesia", perm_model="coates")
    print("Script ready. Please call main('your_file.las') with appropriate parameters.")
    print("\nAvailable Vsh methods:    linear, larionov_tertiary, larionov_older, steiber, clavier")
    print("Available Sw models:      archie, simandoux, mod_simandoux, indonesia, dual_water")
    print("Available Perm models:    timur, coates, morris_oil, morris_gas, tixier")
    print("Available Porosity modes: avg, rms, density, neutron")
