import streamlit as st
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import io
import tempfile
import os

# =============================================================================
# PAGE CONFIG & PREMIUM STYLING
# =============================================================================
st.set_page_config(
    page_title="Petrophysics Pro — Evaluation Suite",
    layout="wide",
    page_icon="🛢️",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e0e0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #38bdf8 !important;
    }

    /* Headers */
    h1 { color: #38bdf8 !important; font-weight: 700 !important; }
    h2 { color: #818cf8 !important; font-weight: 600 !important; }
    h3 { color: #a78bfa !important; font-weight: 500 !important; }

    /* Cards / Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(56,189,248,0.08) 0%, rgba(129,140,248,0.08) 100%);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h4 {
        color: #94a3b8 !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .metric-card .value {
        color: #38bdf8;
        font-size: 1.6rem;
        font-weight: 700;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56,189,248,0.15) !important;
        color: #38bdf8 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        color: #a78bfa !important;
        font-weight: 500;
    }

    /* Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(56,189,248,0.3);
    }

    /* Success / Info alerts */
    .stSuccess { border-left: 4px solid #22c55e !important; }
    .stInfo    { border-left: 4px solid #38bdf8 !important; }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* Hide hamburger */
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PETROPHYSICS ENGINE  (all equations embedded)
# =============================================================================

def clip(a, lo=None, hi=None):
    a = np.array(a, dtype=float)
    if lo is not None: a[a < lo] = lo
    if hi is not None: a[a > hi] = hi
    return a

# ---- Vsh ----
def igr_linear(gr, gr_clean, gr_shale):
    return clip((gr - gr_clean) / (gr_shale - gr_clean), 0, 1)

def vsh_linear(gr, grc, grsh):          return igr_linear(gr, grc, grsh)
def vsh_larionov_tertiary(gr, grc, grsh):
    igr = igr_linear(gr, grc, grsh)
    return clip(0.083 * (2**(3.7*igr) - 1), 0, 1)
def vsh_larionov_older(gr, grc, grsh):
    igr = igr_linear(gr, grc, grsh)
    return clip(0.33 * (2**(2.0*igr) - 1), 0, 1)
def vsh_steiber(gr, grc, grsh):
    igr = igr_linear(gr, grc, grsh)
    return clip(igr / (3.0 - 2.0*igr), 0, 1)
def vsh_clavier(gr, grc, grsh):
    igr = igr_linear(gr, grc, grsh)
    return clip(1.7 - np.sqrt(3.38 - (igr + 0.7)**2), 0, 1)

VSH_METHODS = {
    "Linear": vsh_linear,
    "Larionov (Tertiary)": vsh_larionov_tertiary,
    "Larionov (Older Rocks)": vsh_larionov_older,
    "Steiber": vsh_steiber,
    "Clavier": vsh_clavier,
}

# ---- Porosity ----
def phi_density(rhob, rhob_ma=2.65, rhob_fl=1.0):
    return clip((rhob_ma - rhob) / (rhob_ma - rhob_fl), 0, 0.6)

def phi_neutron(nphi, unit="v/v"):
    x = np.array(nphi, dtype=float)
    if "%" in unit or "p.u" in unit.lower():
        x = x / 100.0
    return clip(x, 0, 0.6)

def phi_sonic_wyllie(dt, dt_ma=55.5, dt_fl=189.0, cp=1.0):
    return clip((dt - dt_ma) / (dt_fl - dt_ma) / cp, 0, 0.6)

def phi_sonic_rhg(dt, dt_ma=55.5, alpha=0.625):
    return clip(alpha * (dt - dt_ma) / dt, 0, 0.6)

def phi_total(phid, phin, method="Average"):
    if method == "Average":       return clip(0.5*(phid+phin), 0, 0.6)
    elif method == "RMS":         return clip(np.sqrt(0.5*(phid**2+phin**2)), 0, 0.6)
    elif method == "Density":     return phid
    elif method == "Neutron":     return phin
    return phid

def phi_effective(phi_t, vsh, phi_sh=0.10):
    return clip(phi_t - vsh * phi_sh, 0, 0.6)

def phi_nd_gas_corrected(phin, phid):
    return clip(np.sqrt((phin**2 + phid**2) / 2.0), 0, 0.6)

# ---- Water Saturation ----
def sw_archie(rt, rw, phi, a=1, m=2, n=2):
    phi = np.where(phi > 0.01, phi, 0.01)
    rt  = np.where(rt  > 0.01, rt,  0.01)
    sw = (a * rw / (rt * phi**m)) ** (1.0/n)
    return clip(sw, 0, 1)

def sw_simandoux(rt, rw, phi, vsh, rsh, a=1, m=2, n=2):
    phi = np.where(phi > 0.01, phi, 0.01)
    rt  = np.where(rt  > 0.01, rt,  0.01)
    C = (1-vsh) * a * rw / phi**m
    D = C * vsh / (2*rsh)
    E = C / rt
    sw = np.power(np.abs(-D + np.sqrt(np.abs(D**2 + E))), 1.0/n)
    return clip(sw, 0, 1)

def sw_mod_simandoux(rt, rw, phi, vsh, rsh, a=1, m=2, n=2):
    phi = np.where(phi > 0.01, phi, 0.01)
    rt  = np.where(rt  > 0.01, rt,  0.01)
    F = a * rw / (phi**m * np.where(1-vsh > 0.01, 1-vsh, 0.01))
    alpha = F * vsh / (2*rsh)
    beta = F / rt
    sw = np.power(np.abs(-alpha + np.sqrt(np.abs(alpha**2 + beta))), 1.0/n)
    return clip(sw, 0, 1)

def sw_indonesia(rt, rw, phi, vsh, rsh, a=1, m=2, n=2):
    phi = np.where(phi > 0.01, phi, 0.01)
    rt  = np.where(rt  > 0.01, rt,  0.01)
    rsh = np.where(rsh > 0.01, rsh, 0.01)
    A = np.sqrt(1.0 / rt)
    B = np.power(vsh, 1.0 - 0.5*vsh) / np.sqrt(rsh)
    C = np.sqrt(phi**m / (a * rw))
    denom = B + C
    denom = np.where(np.abs(denom) > 1e-10, denom, 1e-10)
    sw = np.power(A / denom, 2.0/n)
    return clip(sw, 0, 1)

def sw_dual_water(rt, rw, phi_t, phi_e, a=1, m=2, n=2):
    rwb = rw * 0.3
    swb = np.where(phi_t > 0.01, (phi_t - phi_e) / phi_t, 0)
    swb = clip(swb, 0, 1)
    rw_eff = 1.0 / (1.0/rw + swb * (1.0/rwb - 1.0/rw))
    swt = sw_archie(rt, rw_eff, phi_t, a, m, n)
    swe = np.where(phi_e > 0.01, (swt * phi_t - swb * phi_t) / phi_e, np.nan)
    return clip(swe, 0, 1)

SW_MODELS = {
    "Archie": "archie",
    "Simandoux": "simandoux",
    "Mod. Simandoux": "mod_simandoux",
    "Indonesia (Poupon)": "indonesia",
    "Dual Water": "dual_water",
}

# ---- Permeability ----
def perm_timur(phi, swir):
    return clip(0.136 * phi**4.4 / swir**2, 0, None)
def perm_coates(phi, swir):
    r = (1-swir) / swir
    return clip((100 * phi**2 * r)**2, 0, None)
def perm_morris_oil(phi, swir):
    return clip(62.5 * (phi**3 / swir)**2, 0, None)
def perm_morris_gas(phi, swir):
    return clip(2.5 * (phi**3 / swir)**2, 0, None)
def perm_tixier(phi, swir):
    return clip((250 * phi**3 / swir)**2, 0, None)

PERM_MODELS = {
    "Timur": perm_timur,
    "Coates": perm_coates,
    "Morris-Biggs (Oil)": perm_morris_oil,
    "Morris-Biggs (Gas)": perm_morris_gas,
    "Tixier": perm_tixier,
}

# ---- Rock Typing ----
def winland_r35(phi, k):
    phi_pct = np.where(phi > 0, phi * 100, np.nan)
    k_safe  = np.where(k > 0, k, np.nan)
    log_r35 = 0.732 + 0.588*np.log10(k_safe) - 0.864*np.log10(phi_pct)
    return 10**log_r35

def flow_zone_indicator(phi, k):
    phi = np.where(phi > 0.001, phi, np.nan)
    k   = np.where(k > 0, k, np.nan)
    rqi = 0.0314 * np.sqrt(k / phi)
    phi_z = phi / (1 - phi)
    return rqi / phi_z

# ---- Mechanical Properties ----
def poisson_ratio(dts, dtc):
    r2 = (dts / dtc)**2
    return clip((r2 - 2) / (2*(r2 - 1)), 0, 0.5)

def dynamic_youngs_modulus(rhob, dts, dtc):
    vs = 304800.0 / dts
    vp = 304800.0 / dtc
    E = rhob * 1000 * vs**2 * (3*vp**2 - 4*vs**2) / (vp**2 - vs**2)
    return E / 1e9

def bulk_modulus_calc(rhob, dtc, dts):
    vp = 304800.0 / dtc
    vs = 304800.0 / dts
    return rhob * 1000 * (vp**2 - 4/3 * vs**2) / 1e9

def shear_modulus_calc(rhob, dts):
    vs = 304800.0 / dts
    return rhob * 1000 * vs**2 / 1e9

# ---- Fluid / Temperature ----
def formation_temperature(depth, ts=70, grad=1.2):
    return ts + grad * depth / 100.0

def rw_temp_correction(rw1, t1, t2):
    return rw1 * (t1 + 6.77) / (t2 + 6.77)


# =============================================================================
# HELPER: smart curve guesser
# =============================================================================
def guess_curve(options, candidates):
    for c in candidates:
        for opt in options:
            if c.lower() == opt.lower():
                return opt
            if c.lower() in opt.lower():
                return opt
    return options[0]


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # ---- Header ----
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
        <h1 style="font-size:2.4rem; margin-bottom:0.2rem;">🛢️ Petrophysics Pro</h1>
        <p style="color:#94a3b8; font-size:1rem; letter-spacing:2px;">
            COMPREHENSIVE LOG ANALYSIS &amp; EVALUATION SUITE
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("## 📂 Data Import")
        uploaded_file = st.file_uploader("Upload LAS File", type=["las"])

        st.divider()
        st.markdown("## ⚙️ Evaluation Parameters")

        # -- Vsh --
        with st.expander("🪨 Shale Volume (Vsh)", expanded=True):
            vsh_method = st.selectbox("Vsh Method", list(VSH_METHODS.keys()))
            gr_clean = st.number_input("GR Clean", value=30.0, step=5.0)
            gr_shale = st.number_input("GR Shale", value=120.0, step=5.0)

        # -- Porosity --
        with st.expander("🕳️ Porosity", expanded=False):
            rhob_ma = st.number_input("Matrix Density (g/cc)", value=2.65, step=0.01, format="%.2f")
            rhob_fl = st.number_input("Fluid Density (g/cc)", value=1.00, step=0.01, format="%.2f")
            dt_ma = st.number_input("Matrix DT (µs/ft)", value=55.5, step=0.5, format="%.1f")
            dt_fl = st.number_input("Fluid DT (µs/ft)", value=189.0, step=1.0, format="%.1f")
            nphi_unit = st.selectbox("NPHI Unit", ["v/v (decimal)", "p.u. (%)"])
            phi_method = st.selectbox("PHIT Method", ["Average", "RMS", "Density", "Neutron"])
            phi_sh = st.number_input("Shale Porosity (φ_sh)", value=0.10, step=0.01, format="%.2f")

        # -- Sw --
        with st.expander("💧 Water Saturation (Sw)", expanded=False):
            sw_model_name = st.selectbox("Sw Model", list(SW_MODELS.keys()))
            rw = st.number_input("Rw (Ω·m)", value=0.03, step=0.005, format="%.4f")
            a_param = st.number_input("a (Tortuosity)", value=1.0, format="%.2f")
            m_param = st.number_input("m (Cementation)", value=2.0, format="%.2f")
            n_param = st.number_input("n (Saturation Exp)", value=2.0, format="%.2f")
            if sw_model_name in ["Simandoux", "Mod. Simandoux", "Indonesia (Poupon)"]:
                rsh = st.number_input("Rsh – Shale Resistivity (Ω·m)", value=3.0, step=0.5, format="%.1f")
            else:
                rsh = 3.0

        # -- Permeability --
        with st.expander("🌊 Permeability", expanded=False):
            perm_model_name = st.selectbox("Permeability Model", list(PERM_MODELS.keys()))

        # -- Pay cutoffs --
        with st.expander("🎯 Pay Cutoffs", expanded=False):
            vsh_cut  = st.slider("Vsh cutoff", 0.0, 1.0, 0.35, 0.05)
            phie_cut = st.slider("PHIE cutoff", 0.0, 0.30, 0.08, 0.01)
            sw_cut   = st.slider("Sw cutoff", 0.0, 1.0, 0.60, 0.05)

        # -- Temperature --
        with st.expander("🌡️ Temperature", expanded=False):
            surface_temp = st.number_input("Surface Temp (°F)", value=70.0)
            temp_grad = st.number_input("Gradient (°F/100ft)", value=1.2, step=0.1, format="%.2f")

    # ========== MAIN CONTENT ==========
    if uploaded_file is None:
        # ---- Landing Page ----
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        features = [
            ("🪨", "6 Vsh Models", "Linear · Larionov · Steiber · Clavier"),
            ("💧", "5 Sw Models", "Archie · Simandoux · Indonesia · Dual Water"),
            ("🌊", "5 Perm Models", "Timur · Coates · Morris-Biggs · Tixier"),
            ("⚙️", "Rock Mechanics", "Young's · Poisson · Bulk & Shear Modulus"),
        ]
        for col, (icon, title, desc) in zip([c1,c2,c3,c4], features):
            col.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;">{icon}</div>
                <h4 style="margin-top:0.5rem;">{title}</h4>
                <p style="color:#64748b; font-size:0.8rem;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        st.info("📂 **Upload a LAS file** from the sidebar to begin your evaluation.")
        return

    # ---- Process LAS ----
    try:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".las")
        tfile.write(uploaded_file.read())
        tfile.close()

        las = lasio.read(tfile.name)
        df = las.df()
        df.index.name = "DEPT"
        df.replace([-999.25, -999.0, -9999.0], np.nan, inplace=True)
    except Exception as e:
        st.error(f"❌ Error reading LAS file: {e}")
        return

    cols = list(df.columns)

    # ---- Curve Mapping ----
    st.markdown("### 1️⃣ Curve Mapping")
    st.caption("Map your log curves to the evaluation engine. The app will auto-detect common mnemonics.")

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        gr_curve  = st.selectbox("🟢 Gamma Ray",    cols, index=cols.index(guess_curve(cols, ["GR","CGR","GAM"])))
    with mc2:
        res_curve = st.selectbox("🔴 Deep Resistivity", cols, index=cols.index(guess_curve(cols, ["AHT90","RT","LLD","RDEP","ILD","AT90"])))
    with mc3:
        den_curve = st.selectbox("🔴 Density (RHOB)", cols, index=cols.index(guess_curve(cols, ["RHOZ","RHOB","DEN","ZDEN"])))
    with mc4:
        neu_curve = st.selectbox("🔵 Neutron (NPHI)", cols, index=cols.index(guess_curve(cols, ["TNPH","NPHI","NEU","NPOR"])))

    mc5, mc6, mc7, mc8 = st.columns(4)
    with mc5:
        rxo_options = ["— None —"] + cols
        rxo_curve = st.selectbox("🟠 Shallow Res (Rxo)", rxo_options,
                                 index=rxo_options.index(guess_curve(rxo_options, ["AHT10","RSFL","RXO","MSFL","— None —"])))
    with mc6:
        dt_options = ["— None —"] + cols
        dt_curve = st.selectbox("🟡 Sonic (DT)", dt_options,
                                index=dt_options.index(guess_curve(dt_options, ["DT","DTC","DTCO","AC","— None —"])))
    with mc7:
        dts_options = ["— None —"] + cols
        dts_curve = st.selectbox("🟡 Shear Sonic (DTS)", dts_options,
                                 index=dts_options.index(guess_curve(dts_options, ["DTS","DTSM","— None —"])))
    with mc8:
        cal_options = ["— None —"] + cols
        cal_curve = st.selectbox("⚫ Caliper", cal_options,
                                 index=cal_options.index(guess_curve(cal_options, ["HCAL","CALI","CAL","— None —"])))

    # ---- Run Calculations ----
    try:
        gr   = df[gr_curve].astype(float).values
        rt   = df[res_curve].astype(float).values
        rhob = df[den_curve].astype(float).values
        nphi_raw = df[neu_curve].astype(float).values
        depth = df.index.values

        has_rxo = rxo_curve != "— None —"
        has_dt  = dt_curve  != "— None —"
        has_dts = dts_curve != "— None —"
        has_cal = cal_curve != "— None —"

        rxo = df[rxo_curve].astype(float).values if has_rxo else None
        dt_arr  = df[dt_curve].astype(float).values if has_dt  else None
        dts_arr = df[dts_curve].astype(float).values if has_dts else None
        cal = df[cal_curve].astype(float).values if has_cal else None

        # Temperature
        df["TEMP_F"] = formation_temperature(depth, ts=surface_temp, grad=temp_grad)

        # Vsh
        vsh_func = VSH_METHODS[vsh_method]
        df["VSH"] = vsh_func(gr, gr_clean, gr_shale)

        # Porosity
        nphi_vals = phi_neutron(nphi_raw, unit=nphi_unit)
        df["PHID"] = phi_density(rhob, rhob_ma, rhob_fl)
        df["PHIN"] = nphi_vals
        if has_dt:
            df["PHIS_W"] = phi_sonic_wyllie(dt_arr, dt_ma, dt_fl)
            df["PHIS_R"] = phi_sonic_rhg(dt_arr, dt_ma)
        df["PHIT"] = phi_total(df["PHID"].values, df["PHIN"].values, method=phi_method)
        df["PHIE"] = phi_effective(df["PHIT"].values, df["VSH"].values, phi_sh=phi_sh)
        df["PHIG"] = phi_nd_gas_corrected(df["PHIN"].values, df["PHID"].values)
        if has_dt:
            df["SPI"] = clip(df["PHIN"].values - df["PHIS_W"].values, 0, 0.6)

        # Water Saturation
        phi_sw = np.where(df["PHIE"].values > 0.02, df["PHIE"].values, np.nan)
        rt_sw  = np.where(rt > 0.01, rt, np.nan)
        sw_key = SW_MODELS[sw_model_name]

        if sw_key == "archie":
            df["SW"] = sw_archie(rt_sw, rw, phi_sw, a_param, m_param, n_param)
        elif sw_key == "simandoux":
            df["SW"] = sw_simandoux(rt_sw, rw, phi_sw, df["VSH"].values, rsh, a_param, m_param, n_param)
        elif sw_key == "mod_simandoux":
            df["SW"] = sw_mod_simandoux(rt_sw, rw, phi_sw, df["VSH"].values, rsh, a_param, m_param, n_param)
        elif sw_key == "indonesia":
            df["SW"] = sw_indonesia(rt_sw, rw, phi_sw, df["VSH"].values, rsh, a_param, m_param, n_param)
        elif sw_key == "dual_water":
            df["SW"] = sw_dual_water(rt_sw, rw, df["PHIT"].values, df["PHIE"].values, a_param, m_param, n_param)

        # BVW / BVH
        df["BVW"] = df["SW"].values * df["PHIE"].values
        df["BVH"] = (1 - df["SW"].values) * df["PHIE"].values

        # Permeability
        swir = np.where(df["SW"].values > 0.05, df["SW"].values, 0.05)
        phi_perm = np.where(df["PHIE"].values > 0.01, df["PHIE"].values, np.nan)
        perm_func = PERM_MODELS[perm_model_name]
        df["PERM"] = perm_func(phi_perm, swir)

        # Rock Typing
        mask_rt = (df["PHIE"].values > 0.01) & (df["PERM"].values > 0.001) & (~np.isnan(df["PERM"].values))
        df["R35"] = np.nan
        df["FZI"] = np.nan
        if mask_rt.any():
            df.loc[mask_rt, "R35"] = winland_r35(df.loc[mask_rt, "PHIE"].values, df.loc[mask_rt, "PERM"].values)
            df.loc[mask_rt, "FZI"] = flow_zone_indicator(df.loc[mask_rt, "PHIE"].values, df.loc[mask_rt, "PERM"].values)

        # Mechanical Properties
        if has_dt and has_dts:
            df["POISSON"] = poisson_ratio(dts_arr, dt_arr)
            df["YOUNG_GPa"] = dynamic_youngs_modulus(rhob, dts_arr, dt_arr)
            df["BULK_MOD"]  = bulk_modulus_calc(rhob, dt_arr, dts_arr)
            df["SHEAR_MOD"] = shear_modulus_calc(rhob, dts_arr)

        # Pay Flag
        df["PAY_FLAG"] = ((df["VSH"] <= vsh_cut) & (df["PHIE"] >= phie_cut) & (df["SW"] <= sw_cut)).astype(int)

        st.success(f"✅ **Analysis Complete** — Vsh: {vsh_method} | Sw: {sw_model_name} | Perm: {perm_model_name}")

    except Exception as e:
        st.error(f"❌ Calculation Error: {e}")
        import traceback; st.code(traceback.format_exc())
        return

    # ========== NET PAY SUMMARY ==========
    st.markdown("### 📊 Net Pay Summary")
    pay_df = df[df["PAY_FLAG"] == 1]
    step = np.abs(np.median(np.diff(depth))) if len(depth) > 1 else 0.5
    net_pay_ft = len(pay_df) * step
    gross_ft   = len(df) * step
    ntg = net_pay_ft / gross_ft if gross_ft > 0 else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    metrics = [
        ("Net Pay", f"{net_pay_ft:.1f} ft"),
        ("NTG", f"{ntg:.2%}"),
        ("Avg PHIE", f"{pay_df['PHIE'].mean():.3f}" if not pay_df.empty else "—"),
        ("Avg Sw", f"{pay_df['SW'].mean():.3f}" if not pay_df.empty else "—"),
        ("Avg Vsh", f"{pay_df['VSH'].mean():.3f}" if not pay_df.empty else "—"),
        ("Avg K (mD)", f"{pay_df['PERM'].mean():.2f}" if not pay_df.empty else "—"),
    ]
    for col, (label, val) in zip([m1,m2,m3,m4,m5,m6], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <h4>{label}</h4>
            <div class="value">{val}</div>
        </div>
        """, unsafe_allow_html=True)

    # ========== TABS ==========
    st.markdown("### 2️⃣ Visualization & QC")
    tab_comp, tab_cross, tab_mech, tab_data = st.tabs(
        ["📈 Composite Log", "🔬 Crossplots", "🏗️ Rock Mechanics", "📋 Data Table"]
    )

    # ---- TAB 1: Composite Log ----
    with tab_comp:
        n_tracks = 7
        fig, axes = plt.subplots(1, n_tracks, figsize=(22, 14), sharey=True)
        fig.patch.set_facecolor("#0f0c29")

        for ax in axes:
            ax.set_facecolor("#1a1a2e")
            ax.tick_params(colors='#94a3b8', labelsize=7)
            ax.xaxis.label.set_color('#94a3b8')
            ax.yaxis.label.set_color('#94a3b8')
            for spine in ax.spines.values():
                spine.set_color('#334155')

        # Track 1: GR
        ax = axes[0]
        ax.plot(gr, depth, color='#22c55e', lw=0.8, label='GR')
        ax.set_xlabel("GR (API)", fontsize=8)
        ax.set_xlim(0, 150)
        ax.axvline(x=gr_clean, color='#22c55e', ls=':', alpha=0.4)
        ax.axvline(x=gr_shale, color='#a16207', ls=':', alpha=0.4)
        ax.fill_betweenx(depth, 0, np.clip(gr, 0, 150), color='#22c55e', alpha=0.12)
        ax.legend(loc='upper left', fontsize=6, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')
        if has_cal:
            ax_tw = ax.twiny()
            ax_tw.plot(cal, depth, color='#94a3b8', lw=0.5, ls='--', label='CAL')
            ax_tw.set_xlim(6, 16)
            ax_tw.tick_params(colors='#64748b', labelsize=6)
            ax_tw.xaxis.label.set_color('#64748b')

        # Track 2: Resistivity
        ax = axes[1]
        ax.semilogx(rt, depth, color='#ef4444', lw=0.8, label=res_curve)
        if has_rxo:
            ax.semilogx(rxo, depth, color='#3b82f6', lw=0.6, ls=':', label=rxo_curve)
        ax.set_xlabel("Res (Ω·m)", fontsize=8)
        ax.set_xlim(0.2, 2000)
        ax.grid(True, which='both', ls='-', alpha=0.1, color='#334155')
        ax.legend(loc='upper right', fontsize=6, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')

        # Track 3: Density-Neutron
        ax = axes[2]
        ax.plot(rhob, depth, color='#ef4444', lw=0.8, label='RHOB')
        ax.set_xlabel("RHOB (g/cc)", fontsize=8)
        ax.set_xlim(1.95, 2.95)
        ax_tw = ax.twiny()
        ax_tw.plot(df["PHIN"].values, depth, color='#3b82f6', lw=0.8, ls='--', label='NPHI')
        ax_tw.set_xlabel("NPHI (v/v)", fontsize=8, color='#3b82f6')
        ax_tw.set_xlim(0.45, -0.15)
        ax_tw.tick_params(colors='#3b82f6', labelsize=6)
        ax.legend(loc='upper left', fontsize=6, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')
        ax_tw.legend(loc='upper right', fontsize=6, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')

        # Track 4: Porosity
        ax = axes[3]
        ax.plot(df["PHIE"], depth, color='#f8fafc', lw=1.0, label='PHIE')
        ax.plot(df["PHIT"], depth, color='#64748b', lw=0.6, ls='--', label='PHIT')
        if has_dt:
            ax.plot(df["PHIS_W"], depth, color='#f97316', lw=0.5, ls=':', label='PHIS')
        ax.set_xlabel("Porosity (v/v)", fontsize=8)
        ax.set_xlim(0, 0.40)
        ax.legend(loc='upper left', fontsize=6, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')

        # Track 5: Sw
        ax = axes[4]
        ax.plot(df["SW"], depth, color='#3b82f6', lw=0.8)
        ax.fill_betweenx(depth, 1, df["SW"].values,
                         where=(df["SW"].values < 1), color='#22c55e', alpha=0.25, label='HC')
        ax.fill_betweenx(depth, 0, df["SW"].values,
                         where=(df["SW"].values > 0), color='#38bdf8', alpha=0.12, label='Water')
        ax.set_xlabel("Sw (v/v)", fontsize=8)
        ax.set_xlim(0, 1)
        ax.legend(loc='upper right', fontsize=6, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')

        # Track 6: Vsh
        ax = axes[5]
        ax.plot(df["VSH"], depth, color='#a16207', lw=0.8)
        ax.fill_betweenx(depth, 0, df["VSH"].values, color='#854d0e', alpha=0.25)
        ax.axvline(x=vsh_cut, color='#dc2626', ls='--', lw=0.6)
        ax.set_xlabel("Vsh (v/v)", fontsize=8)
        ax.set_xlim(0, 1)

        # Track 7: Permeability
        ax = axes[6]
        ax.semilogx(df["PERM"], depth, color='#a855f7', lw=0.8, label='K')
        ax.set_xlabel("K (mD)", fontsize=8)
        ax.set_xlim(0.01, 10000)
        ax.grid(True, which='both', ls='-', alpha=0.1, color='#334155')
        ax.legend(loc='upper right', fontsize=6, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')

        for ax in axes:
            ax.invert_yaxis()
            ax.grid(True, alpha=0.15, color='#334155')
            ax.tick_params(axis='x', rotation=45)
        axes[0].set_ylabel("DEPTH (ft)", fontsize=9, color='#94a3b8')

        # Pay shading
        for ax in axes:
            xlims = ax.get_xlim()
            ax.fill_betweenx(depth, xlims[0], xlims[1],
                             where=(df["PAY_FLAG"].values == 1),
                             color='#fbbf24', alpha=0.06)

        fig.suptitle(f"Petrophysical Evaluation — {uploaded_file.name}",
                     fontsize=13, fontweight='bold', color='#38bdf8')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        st.pyplot(fig)
        plt.close(fig)

    # ---- TAB 2: Crossplots ----
    with tab_cross:
        cp1, cp2 = st.columns(2)

        with cp1:
            st.markdown("#### Pickett Plot")
            fig_p, ax_p = plt.subplots(figsize=(7, 5))
            fig_p.patch.set_facecolor("#0f0c29")
            ax_p.set_facecolor("#1a1a2e")
            valid = (df["PHIE"] > 0.01) & (rt > 0.01) & (~np.isnan(df["SW"]))
            if valid.any():
                sc = ax_p.scatter(df.loc[valid, "PHIE"], rt[valid], c=df.loc[valid, "SW"],
                                  cmap='coolwarm_r', s=6, alpha=0.6, vmin=0, vmax=1, edgecolors='none')
                phi_line = np.linspace(0.01, 0.50, 100)
                rt_100 = a_param * rw / phi_line**m_param
                rt_50 = a_param * rw / (0.5**n_param * phi_line**m_param)
                ax_p.plot(phi_line, rt_100, 'w-', lw=0.8, label='Sw=100%')
                ax_p.plot(phi_line, rt_50, 'w--', lw=0.6, label='Sw=50%')
                ax_p.set_xscale('log'); ax_p.set_yscale('log')
                ax_p.set_xlabel("PHIE (v/v)", color='#94a3b8')
                ax_p.set_ylabel("Rt (Ω·m)", color='#94a3b8')
                ax_p.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')
                ax_p.tick_params(colors='#94a3b8')
                ax_p.grid(True, which='both', alpha=0.1, color='#334155')
                cb = plt.colorbar(sc, ax=ax_p)
                cb.set_label('Sw', color='#94a3b8')
                cb.ax.yaxis.set_tick_params(color='#94a3b8')
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#94a3b8')
            st.pyplot(fig_p)
            plt.close(fig_p)

        with cp2:
            st.markdown("#### Neutron-Density Crossplot")
            fig_nd, ax_nd = plt.subplots(figsize=(7, 5))
            fig_nd.patch.set_facecolor("#0f0c29")
            ax_nd.set_facecolor("#1a1a2e")
            valid2 = (~np.isnan(rhob)) & (~np.isnan(df["PHIN"].values))
            if valid2.any():
                sc2 = ax_nd.scatter(df.loc[valid2, "PHIN"], rhob[valid2], c=df.loc[valid2, "VSH"],
                                     cmap='YlOrBr', s=6, alpha=0.6, vmin=0, vmax=1, edgecolors='none')
                ax_nd.plot([0, 0.45], [2.65, 1.0], '--', color='#3b82f6', lw=0.8, alpha=0.6, label='SS line')
                ax_nd.plot([0, 0.45], [2.71, 1.0], '--', color='#22c55e', lw=0.8, alpha=0.6, label='LS line')
                ax_nd.set_xlabel("NPHI (v/v)", color='#94a3b8')
                ax_nd.set_ylabel("RHOB (g/cc)", color='#94a3b8')
                ax_nd.set_xlim(-0.05, 0.50); ax_nd.set_ylim(3.0, 1.8)
                ax_nd.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')
                ax_nd.tick_params(colors='#94a3b8')
                ax_nd.grid(True, alpha=0.1, color='#334155')
                cb2 = plt.colorbar(sc2, ax=ax_nd)
                cb2.set_label('Vsh', color='#94a3b8')
                cb2.ax.yaxis.set_tick_params(color='#94a3b8')
                plt.setp(plt.getp(cb2.ax.axes, 'yticklabels'), color='#94a3b8')
            st.pyplot(fig_nd)
            plt.close(fig_nd)

        cp3, cp4 = st.columns(2)
        with cp3:
            st.markdown("#### Porosity vs Permeability")
            fig_pk, ax_pk = plt.subplots(figsize=(7, 5))
            fig_pk.patch.set_facecolor("#0f0c29")
            ax_pk.set_facecolor("#1a1a2e")
            valid3 = mask_rt & (~np.isnan(df["R35"].values))
            if valid3.any():
                r35_vals = df.loc[valid3, "R35"].values
                r35_safe = np.where(r35_vals > 0, r35_vals, 0.01)
                sc3 = ax_pk.scatter(df.loc[valid3, "PHIE"], df.loc[valid3, "PERM"],
                                     c=r35_safe, cmap='jet', s=8, alpha=0.6,
                                     norm=mcolors.LogNorm(vmin=max(0.01, np.nanmin(r35_safe)),
                                                           vmax=max(1, np.nanmax(r35_safe))),
                                     edgecolors='none')
                ax_pk.set_yscale('log')
                ax_pk.set_xlabel("PHIE (v/v)", color='#94a3b8')
                ax_pk.set_ylabel("K (mD)", color='#94a3b8')
                ax_pk.tick_params(colors='#94a3b8')
                ax_pk.grid(True, which='both', alpha=0.1, color='#334155')
                cb3 = plt.colorbar(sc3, ax=ax_pk)
                cb3.set_label('R35 (µm)', color='#94a3b8')
                cb3.ax.yaxis.set_tick_params(color='#94a3b8')
                plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color='#94a3b8')
            else:
                ax_pk.text(0.5, 0.5, "No valid data", ha='center', va='center',
                           transform=ax_pk.transAxes, color='#64748b')
            st.pyplot(fig_pk)
            plt.close(fig_pk)

        with cp4:
            st.markdown("#### BVW vs Depth")
            fig_bv, ax_bv = plt.subplots(figsize=(7, 5))
            fig_bv.patch.set_facecolor("#0f0c29")
            ax_bv.set_facecolor("#1a1a2e")
            ax_bv.plot(df["BVW"], depth, color='#38bdf8', lw=0.7, label='BVW')
            ax_bv.plot(df["BVH"], depth, color='#22c55e', lw=0.7, label='BVH')
            ax_bv.set_xlabel("Bulk Volume (v/v)", color='#94a3b8')
            ax_bv.set_ylabel("Depth (ft)", color='#94a3b8')
            ax_bv.set_xlim(0, 0.3)
            ax_bv.invert_yaxis()
            ax_bv.tick_params(colors='#94a3b8')
            ax_bv.grid(True, alpha=0.1, color='#334155')
            ax_bv.legend(fontsize=7, facecolor='#1a1a2e', edgecolor='#334155', labelcolor='#94a3b8')
            st.pyplot(fig_bv)
            plt.close(fig_bv)

    # ---- TAB 3: Rock Mechanics ----
    with tab_mech:
        if has_dt and has_dts:
            fig_m, axes_m = plt.subplots(1, 4, figsize=(18, 10), sharey=True)
            fig_m.patch.set_facecolor("#0f0c29")
            titles_m = ["Poisson's Ratio ν", "Young's Modulus (GPa)",
                        "Bulk Modulus (GPa)", "Shear Modulus (GPa)"]
            cols_m = ["POISSON", "YOUNG_GPa", "BULK_MOD", "SHEAR_MOD"]
            colors_m = ["#f97316", "#a855f7", "#06b6d4", "#ec4899"]
            for i, ax in enumerate(axes_m):
                ax.set_facecolor("#1a1a2e")
                ax.plot(df[cols_m[i]], depth, color=colors_m[i], lw=0.8)
                ax.set_xlabel(titles_m[i], fontsize=9, color='#94a3b8')
                ax.tick_params(colors='#94a3b8', labelsize=7)
                ax.grid(True, alpha=0.1, color='#334155')
                ax.invert_yaxis()
                for spine in ax.spines.values():
                    spine.set_color('#334155')
            axes_m[0].set_ylabel("Depth (ft)", color='#94a3b8')
            fig_m.suptitle("Dynamic Mechanical Properties", fontsize=12,
                          fontweight='bold', color='#a855f7')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            st.pyplot(fig_m)
            plt.close(fig_m)
        else:
            st.warning("⚠️ Sonic (DT) and Shear Sonic (DTS) curves are required for mechanical properties. Map them in Curve Mapping above.")

    # ---- TAB 4: Data Table ----
    with tab_data:
        display_cols = [c for c in ["VSH","PHID","PHIN","PHIT","PHIE","PHIG","SW","BVW","BVH","PERM","R35","FZI","PAY_FLAG","TEMP_F"]
                        if c in df.columns]
        if has_dt:
            display_cols = [c for c in ["PHIS_W","PHIS_R","SPI"] if c in df.columns] + display_cols
        if has_dt and has_dts:
            display_cols += [c for c in ["POISSON","YOUNG_GPa","BULK_MOD","SHEAR_MOD"] if c in df.columns]
        st.dataframe(df[display_cols].style.format("{:.4f}"), use_container_width=True, height=600)

    # ========== EXPORT ==========
    st.markdown("### 3️⃣ Export Results")
    ex1, ex2 = st.columns(2)

    csv_data = df.to_csv().encode('utf-8')
    ex1.download_button("📥 Download CSV", data=csv_data, file_name="petro_output.csv", mime="text/csv")

    try:
        out_las = io.StringIO()
        export_cols = [c for c in df.columns if c not in cols]
        for cname in export_cols:
            try:
                las.append_curve(cname, df[cname].fillna(-999.25).values, unit="", descr=f"Calculated {cname}")
            except Exception:
                pass
        las.write(out_las, version=2.0)
        ex2.download_button("📥 Download LAS", data=out_las.getvalue().encode(),
                           file_name="petro_output.las", mime="text/plain")
    except Exception as e:
        ex2.warning(f"LAS export unavailable: {e}")

    os.unlink(tfile.name)


if __name__ == "__main__":
    main()
