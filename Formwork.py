"""
FormOptimus - Formwork Procurement Optimizer (Enhanced)
========================================================
Enhanced with:
  - Historical Kitting Data Upload & Analysis
  - Repetition Analytics (kit reuse tracking, clustering)
  - BoQ Planned vs Actual Comparison
  - On-site Inventory Tracker
  - Data-fitted sigma values from historical data

Author: Procurement Engineering Cell, L&T B&F
Last touched: Feb 2025 → Enhanced Mar 2025

NOTE TO FUTURE DEV: The ESG section is a WIP. Current multipliers are rough
estimates based on a 2022 CIDC report and one internal study from the Pune
site. Do NOT present these as certified figures to clients until we get
sign-off from the sustainability team.

TODO:
  - Replace hardcoded sigma values with project-type lookup (infra vs residential)
  - Add regional weather data pull instead of manual slider (maybe OpenWeather API?)
  - The PDF footer disclaimer needs legal review before going to clients
  - Figure out why sometimes the CVaR calc blows up on very low sim counts
  - Historical sigma fitting needs minimum ~10 projects to be statistically meaningful
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from io import BytesIO
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter


# ── App config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FormOptimus",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background-color: #0f1117;
        background-image:
            radial-gradient(ellipse at 20% 10%, rgba(30, 64, 175, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 90%, rgba(5, 150, 105, 0.06) 0%, transparent 50%);
    }

    section[data-testid="stSidebar"] {
        background-color: #13161f;
        border-right: 1px solid #1e2130;
    }

    .kpi-block {
        background: #13161f;
        border: 1px solid #1e2130;
        border-radius: 8px;
        padding: 18px 16px;
        margin-bottom: 16px;
        will-change: transform;
        transition: transform 0.18s ease, border-color 0.18s ease;
    }
    .kpi-block:hover {
        transform: translateY(-3px);
        border-color: #3b82f6;
    }
    .kpi-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 600;
        color: #60a5fa;
        letter-spacing: -0.5px;
    }
    .kpi-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 4px;
    }
    .kpi-sub {
        font-size: 11px;
        color: #4b5563;
        margin-top: 3px;
    }

    .sb-section {
        font-size: 10px;
        font-weight: 700;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        padding: 14px 0 6px 0;
        border-top: 1px solid #1e2130;
        margin-top: 4px;
    }
    .sb-section:first-of-type {
        border-top: none;
        padding-top: 4px;
    }

    .section-title {
        font-size: 13px;
        font-weight: 600;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-bottom: 1px solid #1e2130;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    .risk-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 600;
    }
    .risk-low  { background: #052e16; color: #4ade80; }
    .risk-mid  { background: #422006; color: #fb923c; }
    .risk-high { background: #3b0764; color: #e879f9; }

    .note-box {
        background: #1a1d27;
        border-left: 3px solid #f59e0b;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        font-size: 12px;
        color: #9ca3af;
        margin: 4px 0 8px 0;
        line-height: 1.5;
    }

    .info-box {
        background: #0c1a2e;
        border-left: 3px solid #3b82f6;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        font-size: 12px;
        color: #9ca3af;
        margin: 4px 0 8px 0;
        line-height: 1.5;
    }

    .success-box {
        background: #052e16;
        border-left: 3px solid #10b981;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        font-size: 12px;
        color: #9ca3af;
        margin: 4px 0 8px 0;
        line-height: 1.5;
    }

    .tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px;
    }
    .tag-blue  { background: #1e3a5f; color: #60a5fa; }
    .tag-green { background: #052e16; color: #4ade80; }
    .tag-amber { background: #422006; color: #fb923c; }
    .tag-red   { background: #450a0a; color: #f87171; }

    .cluster-card {
        background: #13161f;
        border: 1px solid #1e2130;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }

    .stDataFrame { border: 1px solid #1e2130; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Constants ────────────────────────────────────────────────────────────────
HOLDING_COST_RATE        = 0.045
WORKERS_PER_FLOOR        = 12
SHIFT_HOURS              = 8
TRAD_OVERORDER_MULT      = 1.35
KG_CO2_PER_KIT_TRANSPORT = 38
WEATHER_SIGMA = 3.0
REWORK_SIGMA  = 2.0

STRATEGY_FACTORS = {
    "Balanced":       1.00,
    "Accelerated":    1.20,
    "Cost-Minimized": 0.85,
}

# ── Sample data generators for demo purposes ─────────────────────────────────

def generate_sample_kitting_history() -> pd.DataFrame:
    """
    Generates realistic synthetic historical kitting data for demo.
    Mirrors what a site team would export from their Excel tracker.
    Columns: project, floor, element_type, kit_id, kits_ordered, kits_used,
             cycle_days, weather_delay_days, rework_flag, cost_actual, cost_planned
    """
    np.random.seed(42)
    projects = ["Lodha_Palava_T4", "Godrej_27_Mumbai", "Prestige_Elm_Park",
                "DLF_Sector52", "Brigade_Utopia", "Sobha_Hartland"]
    element_types = ["Standard Slab", "Column Grid A", "Column Grid B",
                     "Shear Wall", "Beam Band", "Cantilever Slab", "Core Wall"]

    rows = []
    for proj in projects:
        n_floors = np.random.randint(18, 45)
        for floor in range(1, n_floors + 1):
            for elem in np.random.choice(element_types, size=np.random.randint(2, 5), replace=False):
                kits_planned = np.random.randint(3, 12)
                wastage = np.random.uniform(0.05, 0.40)  # 5-40% over-order
                kits_ordered = int(kits_planned * (1 + wastage))
                kits_used = kits_planned + np.random.randint(0, 2)
                cycle = np.random.choice([5, 6, 7, 7, 8, 9, 10])
                weather_delay = np.random.choice([0, 0, 0, 1, 2, 3], p=[0.5, 0.2, 0.1, 0.1, 0.07, 0.03])
                rework = np.random.choice([0, 1], p=[0.85, 0.15])
                cost_planned = kits_planned * np.random.randint(12000, 20000)
                cost_actual  = kits_ordered * np.random.randint(12000, 20000)
                rows.append({
                    "project": proj,
                    "floor": floor,
                    "element_type": elem,
                    "kit_id": f"{proj[:3].upper()}-F{floor:02d}-{elem[:3].upper()}",
                    "kits_planned": kits_planned,
                    "kits_ordered": kits_ordered,
                    "kits_used": kits_used,
                    "cycle_days": cycle,
                    "weather_delay_days": weather_delay,
                    "rework_flag": rework,
                    "cost_planned": cost_planned,
                    "cost_actual": cost_actual,
                })
    return pd.DataFrame(rows)


def generate_sample_inventory() -> pd.DataFrame:
    """Current on-site inventory snapshot."""
    np.random.seed(99)
    components = [
        "Plywood Sheathing 18mm", "H20 Timber Beams 2.4m", "H20 Timber Beams 3.6m",
        "Steel Walers 3m", "Steel Props Adjustable", "Wedge Clamps",
        "Panel Frames 1.2x2.4m", "Panel Frames 0.6x2.4m", "Tie Rods 15mm",
        "Scaffold Couplers", "Base Plates 200mm", "Stripping Hooks"
    ]
    rows = []
    for comp in components:
        total = np.random.randint(20, 200)
        in_use = np.random.randint(int(total * 0.3), int(total * 0.9))
        available = total - in_use
        damaged = np.random.randint(0, max(1, int(total * 0.08)))
        unit_cost = np.random.randint(500, 8000)
        rows.append({
            "component": comp,
            "total_units": total,
            "in_use": in_use,
            "available": available,
            "damaged": damaged,
            "utilization_pct": round((in_use / total) * 100, 1),
            "unit_cost_inr": unit_cost,
            "inventory_value_inr": total * unit_cost,
            "reorder_point": max(5, int(total * 0.2)),
            "last_updated": (datetime.now() - timedelta(hours=np.random.randint(1, 48))).strftime("%Y-%m-%d %H:%M"),
        })
    return pd.DataFrame(rows)


# ── Core simulation ──────────────────────────────────────────────────────────

def run_simulation(
    floors: int,
    work_zones: int,
    target_cycle: int,
    weather_risk: float,
    rework_risk: float,
    safety_buffer_pct: float,
    strategy: str,
    kit_cost: float,
    labor_per_floor: float,
    weather_sigma: float = None,
    rework_sigma: float = None,
    n_sims: int = 3000,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Monte Carlo cost simulation for formwork procurement.
    Now accepts optional data-fitted sigma values from historical analysis.
    """
    w_sigma = weather_sigma if weather_sigma is not None else WEATHER_SIGMA
    r_sigma = rework_sigma  if rework_sigma  is not None else REWORK_SIGMA

    base_kits = work_zones + (1 if target_cycle < 6 else 0)
    risk_factor  = 1 + (weather_risk + rework_risk) / 100.0
    strat_factor = STRATEGY_FACTORS.get(strategy, 1.0)
    buffer_mult  = 1 + safety_buffer_pct / 100.0

    optimized_kits = max(1, int(base_kits * risk_factor * strat_factor * buffer_mult))

    inventory_cost = optimized_kits * kit_cost
    labor_total    = floors * labor_per_floor
    total_cost_ai  = inventory_cost + labor_total

    traditional_kits = int(base_kits * TRAD_OVERORDER_MULT)
    manual_baseline  = (traditional_kits * kit_cost) + labor_total

    cost_savings = max(0.0, manual_baseline - total_cost_ai)
    saving_pct   = round((cost_savings / manual_baseline) * 100, 2) if manual_baseline > 0 else 0.0

    rng       = np.random.default_rng()
    w_samples = np.clip(rng.normal(weather_risk, w_sigma, n_sims), 0, None)
    r_samples = np.clip(rng.normal(rework_risk,  r_sigma, n_sims), 0, None)

    sim_risk    = 1 + (w_samples + r_samples) / 100.0
    sim_kits    = base_kits * sim_risk * strat_factor * buffer_mult
    simulations = (sim_kits * kit_cost) + labor_total

    mean_cost = int(np.mean(simulations))
    std_cost  = int(np.std(simulations))
    var_90    = int(np.percentile(simulations, 90))
    var_95    = int(np.percentile(simulations, 95))
    var_99    = int(np.percentile(simulations, 99))
    best_case = int(np.percentile(simulations, 5))
    p10       = int(np.percentile(simulations, 10))

    tail    = simulations[simulations > var_95]
    cvar_95 = int(np.mean(tail)) if len(tail) > 0 else var_95

    overrun_prob = round(float(np.mean(simulations > manual_baseline)) * 100, 2)

    est_days   = max(1, int((floors * target_cycle) / work_zones))
    trad_days  = int(est_days * 1.25)
    days_saved = trad_days - est_days

    man_hours    = floors * target_cycle * WORKERS_PER_FLOOR * SHIFT_HOURS
    kit_util     = round((base_kits / optimized_kits) * 100, 1)
    safety_stock = optimized_kits - base_kits

    holding_cost   = int(inventory_cost * HOLDING_COST_RATE)
    cost_per_floor = int(mean_cost / floors)
    max_variance   = int(np.max(simulations) - np.min(simulations))
    cap_efficiency = round((labor_total / total_cost_ai) * 100, 1)
    roi_pct        = round((cost_savings / max(inventory_cost, 1)) * 100, 1)

    carbon_saved_kg     = int(cost_savings * 0.021)
    trees_equiv         = int(carbon_saved_kg / 21)
    transport_co2_saved = int(safety_stock * KG_CO2_PER_KIT_TRANSPORT)
    waste_reduced_kg    = int(safety_stock * 150)
    water_saved_l       = int(cost_savings * 0.05)

    metrics = {
        "optimized_kits": optimized_kits, "base_kits": base_kits,
        "saving_pct": saving_pct, "cost_savings": cost_savings,
        "total_cost_ai": total_cost_ai, "manual_baseline": manual_baseline,
        "traditional_kits": traditional_kits, "mean_cost": mean_cost,
        "std_cost": std_cost, "best_case": best_case, "p10": p10,
        "var_90": var_90, "var_95": var_95, "var_99": var_99,
        "cvar_95": cvar_95, "max_variance": max_variance,
        "overrun_prob": overrun_prob, "est_days": est_days,
        "trad_days": trad_days, "days_saved": days_saved,
        "man_hours": man_hours, "kit_util": kit_util,
        "safety_stock": safety_stock, "inventory_cost": inventory_cost,
        "labor_total": labor_total, "holding_cost": holding_cost,
        "cost_per_floor": cost_per_floor, "roi_pct": roi_pct,
        "cap_efficiency": cap_efficiency,
        "carbon_saved_kg": carbon_saved_kg, "trees_equiv": trees_equiv,
        "transport_co2_saved": transport_co2_saved,
        "waste_reduced_kg": waste_reduced_kg, "water_saved_l": water_saved_l,
        "fitted_weather_sigma": w_sigma,
        "fitted_rework_sigma":  r_sigma,
    }

    return metrics, simulations


# ── Historical data analysis ─────────────────────────────────────────────────

def analyze_historical_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Derives key analytics from historical kitting data:
    - Overorder rates per project/element
    - Repetition scores per element type
    - Fitted sigma values for Monte Carlo
    - Wastage trend
    """
    df = df.copy()
    df["overorder_pct"]  = ((df["kits_ordered"] - df["kits_used"]) / df["kits_used"].clip(lower=1)) * 100
    df["wastage_pct"]    = ((df["kits_ordered"] - df["kits_used"]) / df["kits_ordered"].clip(lower=1)) * 100
    df["cost_overrun"]   = df["cost_actual"] - df["cost_planned"]
    df["cost_overrun_pct"] = (df["cost_overrun"] / df["cost_planned"].clip(lower=1)) * 100

    # Fitted sigmas from real weather/rework distribution
    if "weather_delay_days" in df.columns and len(df) > 10:
        weather_as_pct = df["weather_delay_days"] / df["cycle_days"].clip(lower=1) * 100
        fitted_w_sigma = float(weather_as_pct.std()) if weather_as_pct.std() > 0.1 else WEATHER_SIGMA
    else:
        fitted_w_sigma = WEATHER_SIGMA

    if "rework_flag" in df.columns and len(df) > 10:
        rework_rate = df["rework_flag"].mean() * 100
        fitted_r_sigma = float(np.std(df["rework_flag"] * rework_rate)) if len(df) > 10 else REWORK_SIGMA
    else:
        fitted_r_sigma = REWORK_SIGMA

    # Repetition analysis: how many times each element type appears across floors
    if "element_type" in df.columns:
        repetition = df.groupby("element_type").agg(
            total_occurrences=("floor", "count"),
            avg_kits_ordered=("kits_ordered", "mean"),
            avg_kits_used=("kits_used", "mean"),
            avg_wastage_pct=("wastage_pct", "mean"),
            avg_cycle_days=("cycle_days", "mean"),
            projects_used_in=("project", "nunique"),
        ).reset_index()
        repetition["reuse_score"] = (
            repetition["total_occurrences"] / repetition["total_occurrences"].max() * 100
        ).round(1)
        repetition["standardization_potential"] = repetition["reuse_score"].apply(
            lambda x: "High" if x > 70 else ("Medium" if x > 40 else "Low")
        )
    else:
        repetition = pd.DataFrame()

    # Per-project summary
    proj_summary = df.groupby("project").agg(
        floors_covered=("floor", "nunique"),
        total_kits_ordered=("kits_ordered", "sum"),
        total_kits_used=("kits_used", "sum"),
        avg_overorder_pct=("overorder_pct", "mean"),
        total_cost_planned=("cost_planned", "sum"),
        total_cost_actual=("cost_actual", "sum"),
        avg_cycle_days=("cycle_days", "mean"),
        weather_delay_days=("weather_delay_days", "mean"),
        rework_rate=("rework_flag", "mean"),
    ).reset_index()
    proj_summary["cost_overrun_pct"] = (
        (proj_summary["total_cost_actual"] - proj_summary["total_cost_planned"])
        / proj_summary["total_cost_planned"].clip(lower=1) * 100
    ).round(1)

    # Wastage by floor band (early floors vs mid vs top)
    if "floor" in df.columns:
        df["floor_band"] = pd.cut(
            df["floor"],
            bins=[0, 5, 10, 20, 100],
            labels=["Podium (1-5)", "Low-Rise (6-10)", "Mid-Rise (11-20)", "High-Rise (21+)"]
        )
        wastage_by_band = df.groupby("floor_band", observed=True)["wastage_pct"].mean().reset_index()
        wastage_by_band.columns = ["floor_band", "avg_wastage_pct"]
    else:
        wastage_by_band = pd.DataFrame()

    return {
        "df_enriched":      df,
        "repetition":       repetition,
        "proj_summary":     proj_summary,
        "wastage_by_band":  wastage_by_band,
        "fitted_w_sigma":   fitted_w_sigma,
        "fitted_r_sigma":   fitted_r_sigma,
        "avg_overorder":    float(df["overorder_pct"].mean()),
        "avg_wastage":      float(df["wastage_pct"].mean()),
        "total_projects":   df["project"].nunique() if "project" in df.columns else 0,
        "total_records":    len(df),
    }


def run_element_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """
    K-Means clustering on element features to find natural groupings
    that could share the same formwork kit design.
    """
    required = ["kits_used", "cycle_days"]
    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    features = df[required].dropna()
    if len(features) < 10:
        return pd.DataFrame()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Elbow heuristic — cap at 6 clusters
    n_clusters = min(6, max(2, len(features) // 20))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    df_out = df.loc[features.index].copy()
    df_out["cluster"] = labels
    df_out["cluster_name"] = df_out["cluster"].apply(lambda x: f"Kit Group {x+1}")
    return df_out


# ── BoQ Analysis ─────────────────────────────────────────────────────────────

def analyze_boq(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Derives BoQ planned vs actual from historical data.
    Returns element-wise and project-wise comparisons.
    """
    df = df.copy()
    if "element_type" not in df.columns:
        return {}

    elem_boq = df.groupby("element_type").agg(
        planned_kits=("kits_planned", "sum"),
        actual_kits=("kits_ordered", "sum"),
        used_kits=("kits_used", "sum"),
        planned_cost=("cost_planned", "sum"),
        actual_cost=("cost_actual", "sum"),
    ).reset_index()

    elem_boq["overorder_kits"] = elem_boq["actual_kits"] - elem_boq["used_kits"]
    elem_boq["overorder_pct"]  = (elem_boq["overorder_kits"] / elem_boq["used_kits"].clip(lower=1) * 100).round(1)
    elem_boq["cost_variance"]  = elem_boq["actual_cost"] - elem_boq["planned_cost"]
    elem_boq["cost_var_pct"]   = (elem_boq["cost_variance"] / elem_boq["planned_cost"].clip(lower=1) * 100).round(1)
    elem_boq["savings_potential"] = (elem_boq["overorder_kits"] * (elem_boq["actual_cost"] / elem_boq["actual_kits"].clip(lower=1))).astype(int)

    total_planned = int(elem_boq["planned_cost"].sum())
    total_actual  = int(elem_boq["actual_cost"].sum())
    total_waste   = int(elem_boq["savings_potential"].sum())

    return {
        "elem_boq":      elem_boq,
        "total_planned": total_planned,
        "total_actual":  total_actual,
        "total_waste":   total_waste,
        "waste_pct":     round(total_waste / max(total_actual, 1) * 100, 1),
    }


# ── PDF generation ────────────────────────────────────────────────────────────

def build_pdf(metrics: Dict[str, Any], inputs: Dict[str, Any]) -> BytesIO:
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        rightMargin=52, leftMargin=52,
        topMargin=48, bottomMargin=48
    )

    styles = getSampleStyleSheet()
    S = styles

    title_st = ParagraphStyle("ReportTitle", parent=S["Normal"],
        fontName="Helvetica-Bold", fontSize=18,
        textColor=colors.HexColor("#0f172a"), spaceAfter=4)
    sub_st = ParagraphStyle("ReportSub", parent=S["Normal"],
        fontName="Helvetica", fontSize=11,
        textColor=colors.HexColor("#64748b"), spaceAfter=16)
    body_st = ParagraphStyle("Body", parent=S["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=15,
        textColor=colors.HexColor("#334155"), spaceAfter=14)
    warn_st = ParagraphStyle("Warning", parent=S["Normal"],
        fontName="Helvetica-Oblique", fontSize=8,
        textColor=colors.HexColor("#92400e"), leading=12, spaceAfter=12)

    elems = []
    elems.append(Paragraph("FormOptimus Procurement Report", title_st))
    elems.append(Paragraph(
        f"Strategy: <b>{inputs['strategy']}</b> &nbsp;|&nbsp; "
        f"Project: <b>{inputs['floors']} Floors</b> &nbsp;|&nbsp; "
        f"Zones: <b>{inputs['work_zones']}</b> &nbsp;|&nbsp; "
        f"Simulations: <b>3,000 iterations</b>",
        sub_st
    ))
    elems.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")))
    elems.append(Spacer(1, 0.15 * inch))

    sigma_note = ""
    if inputs.get("data_fitted"):
        sigma_note = (f" Risk sigmas fitted from {inputs.get('historical_records', 'N/A')} "
                      f"historical records (weather σ={metrics['fitted_weather_sigma']:.1f}, "
                      f"rework σ={metrics['fitted_rework_sigma']:.1f}).")

    elems.append(Paragraph(
        f"The AI engine recommends procuring <b>{metrics['optimized_kits']} formwork kits</b> "
        f"(vs. {metrics['traditional_kits']} kits under a 35% over-order assumption). "
        f"Projected saving: <b>₹ {int(metrics['cost_savings']):,}</b> ({metrics['saving_pct']}%). "
        f"Schedule impact: <b>{metrics['days_saved']} fewer days</b> vs. traditional timelines.{sigma_note}",
        body_st
    ))
    elems.append(Paragraph(
        "⚠ ESG figures below are indicative estimates based on internal benchmarks only. "
        "Not independently audited. Do not cite in client sustainability reports.",
        warn_st
    ))

    def c(v): return f"₹ {int(v):,}"
    def n(v): return f"{int(v):,}"

    rows = [
        ["Metric", "Value"],
        ["FINANCIAL", ""],
        ["Traditional Baseline (35% over-order)", c(metrics["manual_baseline"])],
        ["AI Optimized Expected Cost (mean)", c(metrics["mean_cost"])],
        ["Projected Cost Saving", c(metrics["cost_savings"])],
        ["Saving %", f"{metrics['saving_pct']}%"],
        ["ROI vs. Inventory Spend", f"{metrics['roi_pct']}%"],
        ["Cost Per Floor (mean)", c(metrics["cost_per_floor"])],
        ["Inventory CapEx", c(metrics["inventory_cost"])],
        ["Labor OpEx", c(metrics["labor_total"])],
        ["Estimated Holding Cost (4.5% p.a.)", c(metrics["holding_cost"])],
        ["Capital Efficiency (Labor / Total)", f"{metrics['cap_efficiency']}%"],
        ["RISK ANALYTICS", ""],
        [f"Risk Sigma Source", "Data-fitted" if inputs.get("data_fitted") else "Engineering estimate"],
        ["Budget Overrun Probability vs. Baseline", f"{metrics['overrun_prob']}%"],
        ["Cost Volatility (1σ)", c(metrics["std_cost"])],
        ["Best Case (5th percentile)", c(metrics["best_case"])],
        ["VaR @ 90%", c(metrics["var_90"])],
        ["VaR @ 95%", c(metrics["var_95"])],
        ["VaR @ 99%", c(metrics["var_99"])],
        ["Conditional VaR / Expected Shortfall (>95%)", c(metrics["cvar_95"])],
        ["OPERATIONAL", ""],
        ["Recommended Kits", str(metrics["optimized_kits"])],
        ["Safety Stock (buffer kits)", str(metrics["safety_stock"])],
        ["Kit Utilization Rate", f"{metrics['kit_util']}%"],
        ["Estimated Project Duration", f"{metrics['est_days']} days"],
        ["Schedule Saving vs. Traditional", f"{metrics['days_saved']} days"],
        ["Estimated Total Man-Hours", n(metrics["man_hours"])],
        ["ESG (INDICATIVE — NOT AUDITED)", ""],
        ["Carbon Saving Estimate", f"{n(metrics['carbon_saved_kg'])} kg CO₂e"],
        ["Ecological Offset (approx.)", f"{n(metrics['trees_equiv'])} trees/year"],
        ["Transport Emissions Avoided", f"{n(metrics['transport_co2_saved'])} kg CO₂e"],
        ["Construction Waste Avoided", f"{n(metrics['waste_reduced_kg'])} kg"],
    ]

    tbl = Table(rows, colWidths=[4.7 * inch, 2.1 * inch])
    cmd = [
        ("BACKGROUND", (0,0), (1,0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR",  (0,0), (1,0), colors.white),
        ("FONTNAME",   (0,0), (1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (1,0), 10),
        ("ALIGN",      (1,0), (1,0), "RIGHT"),
        ("BOTTOMPADDING", (0,0), (1,0), 9),
        ("TOPPADDING",    (0,0), (1,0), 9),
        ("FONTNAME", (0,1), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,1), (1,-1), 8.5),
        ("ALIGN",    (1,1), (1,-1), "RIGHT"),
        ("GRID",     (0,0), (-1,-1), 0.4, colors.HexColor("#e2e8f0")),
        ("VALIGN",   (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,1), (-1,-1), 4),
        ("BOTTOMPADDING", (0,1), (-1,-1), 4),
    ]
    for i, row in enumerate(rows):
        if row[1] == "":
            cmd += [
                ("SPAN",       (0,i), (1,i)),
                ("BACKGROUND", (0,i), (1,i), colors.HexColor("#f1f5f9")),
                ("TEXTCOLOR",  (0,i), (1,i), colors.HexColor("#475569")),
                ("FONTNAME",   (0,i), (1,i), "Helvetica-Bold"),
                ("FONTSIZE",   (0,i), (1,i), 8),
                ("TOPPADDING",    (0,i), (1,i), 7),
                ("BOTTOMPADDING", (0,i), (1,i), 7),
            ]
            if "ESG" in row[0]:
                cmd.append(("BACKGROUND", (0,i), (1,i), colors.HexColor("#fef3c7")))
                cmd.append(("TEXTCOLOR",  (0,i), (1,i), colors.HexColor("#92400e")))
        elif i % 2 == 0 and i > 0:
            cmd.append(("BACKGROUND", (0,i), (1,i), colors.HexColor("#f8fafc")))
        else:
            cmd.append(("BACKGROUND", (0,i), (1,i), colors.white))

    tbl.setStyle(TableStyle(cmd))
    elems.append(tbl)
    elems.append(Spacer(1, 0.25 * inch))
    elems.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e2e8f0")))
    footer_st = ParagraphStyle("Footer", parent=S["Normal"],
        fontSize=7, textColor=colors.HexColor("#94a3b8"), leading=11)
    elems.append(Paragraph(
        "INTERNAL USE. Results are probabilistic estimates, not guaranteed outcomes. "
        "Not a substitute for site-specific BOQ or vendor quotes. "
        "ESG figures must not be used for regulatory reporting without sustainability cell review. "
        "Baseline assumes 35% over-order rate (internal study, 2023).",
        footer_st
    ))
    doc.build(elems)
    buf.seek(0)
    return buf


# ── UI helpers ────────────────────────────────────────────────────────────────

def kpi(label: str, value: str, sub: str = ""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="kpi-block">
        <div class="kpi-num">{value}</div>
        <div class="kpi-label">{label}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def risk_badge(prob: float) -> str:
    if prob < 15:
        return f'<span class="risk-badge risk-low">LOW {prob}%</span>'
    elif prob < 35:
        return f'<span class="risk-badge risk-mid">MODERATE {prob}%</span>'
    else:
        return f'<span class="risk-badge risk-high">HIGH {prob}%</span>'


def sb_header(text: str):
    st.markdown(f'<div class="sb-section">{text}</div>', unsafe_allow_html=True)


def dark_chart_layout(title: str = "", height: int = 320) -> dict:
    return dict(
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#9ca3af", family="IBM Plex Sans"),
        margin=dict(l=10, r=10, t=35 if title else 20, b=10),
        height=height,
        title=dict(text=title, font=dict(size=12, color="#6b7280")) if title else None,
    )


# ── Layout ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:8px;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;color:#f1f5f9;">
        FormOptimus
    </span>
    <span style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:#4b5563;margin-left:12px;">
        Formwork Procurement Optimizer · Monte Carlo Risk Engine · Historical Analytics
    </span>
</div>
""", unsafe_allow_html=True)


# ── Session state for historical data ────────────────────────────────────────
if "hist_df" not in st.session_state:
    st.session_state.hist_df = None
if "hist_analytics" not in st.session_state:
    st.session_state.hist_analytics = None
if "fitted_sigmas" not in st.session_state:
    st.session_state.fitted_sigmas = {"weather": None, "rework": None}


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    sb_header("Project Structure")
    floors = st.number_input("Total Floors", min_value=1, max_value=100, value=30,
        help="Total above-grade floors in scope")
    work_zones = st.number_input("Parallel Work Zones", min_value=1, max_value=5, value=2,
        help="Number of concurrent active formwork zones")
    cycle_days = st.number_input("Floor Cycle (days)", min_value=4, max_value=15, value=7,
        help="Target pour-to-pour cycle time per floor")

    sb_header("Risk Inputs")
    weather_risk = st.slider("Weather Risk %", min_value=0, max_value=30, value=5)
    rework_risk  = st.slider("Rework Risk %",  min_value=0, max_value=20, value=3)
    safety_buf   = st.slider("Safety Buffer %", min_value=5, max_value=25, value=10)

    # Show whether sigmas are data-fitted or defaults
    if st.session_state.fitted_sigmas["weather"] is not None:
        st.markdown(
            f'<div class="success-box">✓ Risk sigmas fitted from historical data<br>'
            f'Weather σ = {st.session_state.fitted_sigmas["weather"]:.2f} | '
            f'Rework σ = {st.session_state.fitted_sigmas["rework"]:.2f}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="note-box">ℹ Upload historical data in the History tab to fit risk sigmas automatically.</div>',
            unsafe_allow_html=True
        )

    sb_header("Cost Inputs")
    kit_cost        = st.number_input("Kit Cost (₹)",        min_value=5000,  max_value=500000, value=15000,  step=1000)
    labor_per_floor = st.number_input("Labor per Floor (₹)", min_value=10000, max_value=500000, value=25000, step=1000)

    sb_header("Optimization Strategy")
    strategy = st.radio("strategy_select",
        options=["Balanced", "Accelerated", "Cost-Minimized"],
        label_visibility="collapsed")
    st.markdown(
        '<div class="note-box">⚠ Cost-Minimized increases schedule risk.<br>Check overrun % after running.</div>',
        unsafe_allow_html=True)

    st.write("")
    run_btn = st.button("▶ Run Simulation", use_container_width=True, type="primary")


# ── Main tabs ─────────────────────────────────────────────────────────────────
main_tabs = st.tabs([
    "📊 Simulation",
    "📁 Historical Kitting Data",
    "🔁 Repetition Analytics",
    "📋 BoQ Comparison",
    "🏗️ Inventory Tracker",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: SIMULATION (original functionality)
# ═══════════════════════════════════════════════════════════════════
with main_tabs[0]:
    if run_btn:
        with st.spinner("Running 3,000 Monte Carlo iterations..."):
            time.sleep(0.6)
            metrics, sims = run_simulation(
                floors, work_zones, cycle_days,
                weather_risk, rework_risk, safety_buf,
                strategy, kit_cost, labor_per_floor,
                weather_sigma=st.session_state.fitted_sigmas["weather"],
                rework_sigma=st.session_state.fitted_sigmas["rework"],
            )

        # Data-fitted badge
        if st.session_state.fitted_sigmas["weather"] is not None:
            st.markdown(
                '<div class="success-box">✓ Simulation used data-fitted risk sigmas from your historical upload</div>',
                unsafe_allow_html=True
            )

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi("Recommended Kits", str(metrics["optimized_kits"]), f"base: {metrics['base_kits']} + buffer")
        with c2: kpi("Projected Saving", f"{metrics['saving_pct']}%", f"₹ {int(metrics['cost_savings']/1000):,}K")
        with c3: kpi("CVaR (95%)", f"₹{round(metrics['cvar_95']/100000,1)}L", "expected shortfall")
        with c4: kpi("Schedule Gain", f"{metrics['days_saved']}d", f"vs. {metrics['trad_days']}d traditional")
        with c5: kpi("Overrun Risk", f"{metrics['overrun_prob']}%", "vs. traditional baseline")

        st.markdown("---")

        tab_fin, tab_risk, tab_ops, tab_esg = st.tabs([
            "📊 Financials", "📈 Risk Distribution", "⚙ Operational", "🌱 ESG"
        ])

        with tab_fin:
            st.markdown('<div class="section-title">Cost Breakdown</div>', unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                df_compare = pd.DataFrame({
                    "Item": ["Inventory CapEx", "Labor OpEx", "Total Cost",
                             "Baseline (Traditional)", "Projected Saving",
                             "Holding Cost (4.5% p.a.)", "ROI vs Inventory", "Capital Efficiency"],
                    "AI Optimized": [
                        f"₹ {int(metrics['inventory_cost']):,}",
                        f"₹ {int(metrics['labor_total']):,}",
                        f"₹ {int(metrics['total_cost_ai']):,}",
                        f"₹ {int(metrics['manual_baseline']):,}",
                        f"₹ {int(metrics['cost_savings']):,} ({metrics['saving_pct']}%)",
                        f"₹ {int(metrics['holding_cost']):,}",
                        f"{metrics['roi_pct']}%",
                        f"{metrics['cap_efficiency']}%",
                    ]
                })
                st.dataframe(df_compare, hide_index=True, use_container_width=True)
            with cc2:
                fig_wf = go.Figure(go.Waterfall(
                    orientation="v", measure=["absolute", "absolute", "total", "absolute", "relative"],
                    x=["Inventory", "Labor", "AI Total", "Traditional", "Saving"],
                    y=[metrics["inventory_cost"], metrics["labor_total"], 0,
                       metrics["manual_baseline"], -metrics["cost_savings"]],
                    connector={"line": {"color": "#1e2130"}},
                    increasing={"marker": {"color": "#3b82f6"}},
                    decreasing={"marker": {"color": "#10b981"}},
                    totals={"marker": {"color": "#6366f1"}},
                ))
                fig_wf.update_layout(**dark_chart_layout("Cost Waterfall (₹)"))
                st.plotly_chart(fig_wf, use_container_width=True)

        with tab_risk:
            st.markdown('<div class="section-title">Monte Carlo Distribution (3,000 iterations)</div>', unsafe_allow_html=True)
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(x=sims, nbinsx=70, marker_color="#3b82f6", opacity=0.65))
                for val, color, label in [
                    (metrics["mean_cost"], "#10b981", "Mean"),
                    (metrics["var_95"],    "#ef4444", "95% VaR"),
                    (metrics["best_case"], "#60a5fa", "5th pct"),
                    (metrics["manual_baseline"], "#f59e0b", "Baseline"),
                ]:
                    fig_hist.add_vline(x=val, line_color=color, line_dash="dash",
                                       annotation_text=label, annotation_font_color=color)
                fig_hist.update_layout(**dark_chart_layout(), xaxis_title="Simulated Total Cost (₹)", yaxis_title="Frequency", showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            with rc2:
                st.markdown('<div class="section-title">Percentile Table</div>', unsafe_allow_html=True)
                df_pct = pd.DataFrame({
                    "Percentile": ["5th (best)", "10th", "50th (median)", "90th", "95th (VaR)", "99th", "CVaR >95%"],
                    "Cost (₹)": [
                        f"₹ {metrics['best_case']:,}", f"₹ {metrics['p10']:,}",
                        f"₹ {int(np.median(sims)):,}", f"₹ {metrics['var_90']:,}",
                        f"₹ {metrics['var_95']:,}", f"₹ {metrics['var_99']:,}",
                        f"₹ {metrics['cvar_95']:,}",
                    ]
                })
                st.dataframe(df_pct, hide_index=True, use_container_width=True)
                st.markdown("**Budget overrun risk:**")
                st.markdown(risk_badge(metrics["overrun_prob"]), unsafe_allow_html=True)
                st.caption(f"σ: ₹ {int(metrics['std_cost']):,}")

        with tab_ops:
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.markdown('<div class="section-title">Schedule</div>', unsafe_allow_html=True)
                st.metric("Estimated Duration",   f"{metrics['est_days']} days")
                st.metric("Traditional Estimate", f"{metrics['trad_days']} days")
                st.metric("Schedule Saving",      f"{metrics['days_saved']} days")
                st.metric("Total Man-Hours",       f"{metrics['man_hours']:,}")
            with oc2:
                st.markdown('<div class="section-title">Inventory</div>', unsafe_allow_html=True)
                st.metric("Recommended Kits", metrics["optimized_kits"])
                st.metric("Safety Stock",     f"{metrics['safety_stock']} kits")
                st.metric("Kit Utilization",  f"{metrics['kit_util']}%")
                st.metric("Holding Cost",     f"₹ {metrics['holding_cost']:,}")
            with oc3:
                st.markdown('<div class="section-title">Strategy Notes</div>', unsafe_allow_html=True)
                notes = {
                    "Balanced":       "Risk-adjusted. Suitable for most projects.",
                    "Accelerated":    "20% uplift. Lower schedule risk, higher CapEx.",
                    "Cost-Minimized": "Lean procurement. Lower holding cost, higher overrun risk.",
                }
                st.info(notes[strategy])
                if strategy == "Cost-Minimized" and metrics["overrun_prob"] > 30:
                    st.warning(f"Overrun probability is **{metrics['overrun_prob']}%**. Consider Balanced.")

        with tab_esg:
            st.markdown(
                '<div class="note-box">Rough estimates — internal benchmarks & CIDC 2022. '
                '<b>Do not cite in client sustainability reports</b> without sustainability cell review.</div>',
                unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)
            with ec1:
                st.metric("Carbon Saving (est.)",        f"{metrics['carbon_saved_kg']:,} kg CO₂e")
                st.metric("Transport Emissions Avoided", f"{metrics['transport_co2_saved']:,} kg CO₂e")
                st.metric("Ecological Offset (rough)",   f"≈ {metrics['trees_equiv']:,} trees/yr")
            with ec2:
                st.metric("Waste Avoided",  f"{metrics['waste_reduced_kg']:,} kg")
                st.metric("Water Savings",  f"{metrics['water_saved_l']:,} L")

        st.markdown("---")
        pdf_buf = build_pdf(
            metrics,
            {
                "strategy": strategy, "floors": floors, "work_zones": work_zones,
                "data_fitted": st.session_state.fitted_sigmas["weather"] is not None,
                "historical_records": len(st.session_state.hist_df) if st.session_state.hist_df is not None else 0,
            }
        )
        st.download_button(
            "📄 Download PDF Report", pdf_buf,
            file_name=f"FormOptimus_{strategy}_{floors}fl.pdf",
            mime="application/pdf",
        )
        st.caption("Includes all metrics, methodology notes, and ESG disclaimers.")

    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 0;color:#374151;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:48px;margin-bottom:16px;color:#1e2130;">[ ]</div>
            <div style="font-size:14px;color:#4b5563;">Configure parameters in the sidebar and click <b>Run Simulation</b>.</div>
            <div style="font-size:12px;color:#374151;margin-top:8px;">Tip: Upload historical kitting data in the <b>Historical Kitting Data</b> tab to improve simulation accuracy.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2: HISTORICAL KITTING DATA
# ═══════════════════════════════════════════════════════════════════
with main_tabs[1]:
    st.markdown('<div class="section-title">Historical Kitting Data Upload & Analysis</div>', unsafe_allow_html=True)

    col_upload, col_template = st.columns([2, 1])

    with col_template:
        st.markdown("**Expected CSV columns:**")
        st.markdown("""
        <div class="info-box">
        <b>Required:</b><br>
        • project<br>
        • floor<br>
        • element_type<br>
        • kits_planned<br>
        • kits_ordered<br>
        • kits_used<br>
        • cycle_days<br>
        • cost_planned<br>
        • cost_actual<br><br>
        <b>Optional (improve sigma fitting):</b><br>
        • weather_delay_days<br>
        • rework_flag (0/1)
        </div>
        """, unsafe_allow_html=True)

        # Download sample template
        sample_df = generate_sample_kitting_history().head(30)
        csv_buf = BytesIO()
        sample_df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button(
            "📥 Download Sample Template",
            csv_buf, file_name="formwork_history_template.csv",
            mime="text/csv", use_container_width=True
        )

    with col_upload:
        uploaded = st.file_uploader(
            "Upload historical kitting CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help="Your past project formwork kitting records"
        )

        use_demo = st.checkbox("Use built-in demo data (6 projects, ~900 records)", value=True)

        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)
                st.session_state.hist_df = df_raw
                st.success(f"✓ Loaded {len(df_raw):,} records from {uploaded.name}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        elif use_demo:
            if st.session_state.hist_df is None or st.button("🔄 Reload demo data"):
                with st.spinner("Generating demo data..."):
                    st.session_state.hist_df = generate_sample_kitting_history()
            st.markdown(
                f'<div class="success-box">✓ Demo data loaded — '
                f'{len(st.session_state.hist_df):,} records across 6 projects</div>',
                unsafe_allow_html=True
            )

    if st.session_state.hist_df is not None:
        df_h = st.session_state.hist_df

        with st.spinner("Analysing historical data..."):
            analytics = analyze_historical_data(df_h)
            st.session_state.hist_analytics = analytics
            # Update fitted sigmas in session state
            st.session_state.fitted_sigmas["weather"] = analytics["fitted_w_sigma"]
            st.session_state.fitted_sigmas["rework"]  = analytics["fitted_r_sigma"]

        st.markdown("---")

        # KPI row
        hk1, hk2, hk3, hk4, hk5 = st.columns(5)
        with hk1: kpi("Projects", str(analytics["total_projects"]), "in dataset")
        with hk2: kpi("Records", f"{analytics['total_records']:,}", "kitting entries")
        with hk3: kpi("Avg Over-Order", f"{analytics['avg_overorder']:.1f}%", "vs kits used")
        with hk4: kpi("Avg Wastage", f"{analytics['avg_wastage']:.1f}%", "of kits ordered unused")
        with hk5: kpi("Fitted Weather σ", f"{analytics['fitted_w_sigma']:.2f}", "replaces hardcoded 3.0")

        st.markdown("---")

        ht1, ht2 = st.tabs(["📊 Per-Project Summary", "🗂️ Raw Data Preview"])

        with ht1:
            proj_s = analytics["proj_summary"]
            if not proj_s.empty:
                # Color code cost overrun
                def style_overrun(val):
                    if isinstance(val, float):
                        if val > 20: return "color: #f87171"
                        elif val > 10: return "color: #fb923c"
                        else: return "color: #4ade80"
                    return ""

                display_cols = ["project", "floors_covered", "total_kits_ordered",
                                "total_kits_used", "avg_overorder_pct",
                                "avg_cycle_days", "rework_rate", "cost_overrun_pct"]
                st.dataframe(
                    proj_s[display_cols].rename(columns={
                        "project": "Project",
                        "floors_covered": "Floors",
                        "total_kits_ordered": "Kits Ordered",
                        "total_kits_used": "Kits Used",
                        "avg_overorder_pct": "Avg Over-Order %",
                        "avg_cycle_days": "Avg Cycle Days",
                        "rework_rate": "Rework Rate",
                        "cost_overrun_pct": "Cost Overrun %",
                    }),
                    hide_index=True, use_container_width=True
                )

                # Overorder by project chart
                fig_oo = go.Figure()
                fig_oo.add_trace(go.Bar(
                    x=proj_s["project"], y=proj_s["avg_overorder_pct"],
                    marker_color=["#ef4444" if v > 20 else "#f59e0b" if v > 10 else "#10b981"
                                  for v in proj_s["avg_overorder_pct"]],
                    text=[f"{v:.1f}%" for v in proj_s["avg_overorder_pct"]],
                    textposition="outside",
                ))
                fig_oo.update_layout(**dark_chart_layout("Average Over-Order % by Project", 280))
                fig_oo.update_layout(yaxis_title="Over-Order %")
                st.plotly_chart(fig_oo, use_container_width=True)

                # Wastage by floor band
                wb = analytics["wastage_by_band"]
                if not wb.empty:
                    fig_wb = go.Figure(go.Bar(
                        x=wb["floor_band"].astype(str), y=wb["avg_wastage_pct"],
                        marker_color="#6366f1",
                        text=[f"{v:.1f}%" for v in wb["avg_wastage_pct"]],
                        textposition="outside",
                    ))
                    fig_wb.update_layout(**dark_chart_layout("Avg Wastage % by Floor Band", 260))
                    st.plotly_chart(fig_wb, use_container_width=True)

        with ht2:
            st.dataframe(df_h.head(100), hide_index=True, use_container_width=True)
            st.caption(f"Showing first 100 of {len(df_h):,} records.")

    else:
        st.markdown("""
        <div style="text-align:center;padding:50px;color:#4b5563;">
            <div style="font-size:32px;margin-bottom:12px;">📁</div>
            <div>Upload your CSV / Excel above, or check "Use demo data" to get started.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3: REPETITION ANALYTICS
# ═══════════════════════════════════════════════════════════════════
with main_tabs[2]:
    st.markdown('<div class="section-title">Repetition Analytics & Kit Clustering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Identifies which element types repeat most across floors and projects, '
        'enabling the same formwork kit design to be reused — reducing custom fabrication costs and inventory.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.hist_analytics is None:
        st.warning("Load historical data in the **Historical Kitting Data** tab first.")
    else:
        analytics = st.session_state.hist_analytics
        rep_df = analytics["repetition"]

        if not rep_df.empty:
            # Reuse score chart
            rep_sorted = rep_df.sort_values("reuse_score", ascending=True)
            colors_list = ["#10b981" if v == "High" else "#f59e0b" if v == "Medium" else "#6b7280"
                           for v in rep_sorted["standardization_potential"]]

            fig_rep = go.Figure(go.Bar(
                y=rep_sorted["element_type"],
                x=rep_sorted["reuse_score"],
                orientation="h",
                marker_color=colors_list,
                text=[f'{v:.0f}% | {p}' for v, p in zip(rep_sorted["reuse_score"], rep_sorted["standardization_potential"])],
                textposition="outside",
            ))
            fig_rep.update_layout(
                **dark_chart_layout("Kit Reuse Score by Element Type (100 = most repeated)", 360),
                xaxis_title="Reuse Score", yaxis_title=""
            )
            st.plotly_chart(fig_rep, use_container_width=True)

            # Repetition table
            rc1, rc2 = st.columns([2, 1])
            with rc1:
                st.markdown('<div class="section-title">Element-Level Repetition Summary</div>', unsafe_allow_html=True)
                display = rep_df.rename(columns={
                    "element_type": "Element Type",
                    "total_occurrences": "Total Occurrences",
                    "avg_kits_ordered": "Avg Kits Ordered",
                    "avg_kits_used": "Avg Kits Used",
                    "avg_wastage_pct": "Avg Wastage %",
                    "avg_cycle_days": "Avg Cycle Days",
                    "projects_used_in": "Projects",
                    "reuse_score": "Reuse Score",
                    "standardization_potential": "Std. Potential",
                }).round(2)
                st.dataframe(display, hide_index=True, use_container_width=True)

            with rc2:
                st.markdown('<div class="section-title">Standardization Summary</div>', unsafe_allow_html=True)
                pot_counts = rep_df["standardization_potential"].value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=pot_counts.index, values=pot_counts.values,
                    marker_colors=["#10b981", "#f59e0b", "#6b7280"],
                    hole=0.5,
                ))
                fig_pie.update_layout(**dark_chart_layout(height=240))
                st.plotly_chart(fig_pie, use_container_width=True)

                high_std = rep_df[rep_df["standardization_potential"] == "High"]
                if not high_std.empty:
                    st.markdown("**High-potential elements:**")
                    for _, row in high_std.iterrows():
                        st.markdown(
                            f'<span class="tag tag-green">✓ {row["element_type"]}</span>',
                            unsafe_allow_html=True
                        )

            st.markdown("---")
            st.markdown('<div class="section-title">K-Means Kit Clustering</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-box">Groups elements with similar kit requirements and cycle times. '
                'Elements in the same cluster can share a standardized kit design.</div>',
                unsafe_allow_html=True
            )

            df_clustered = run_element_clustering(analytics["df_enriched"])
            if not df_clustered.empty:
                # Scatter plot of clusters
                fig_cl = px.scatter(
                    df_clustered.sample(min(500, len(df_clustered))),
                    x="cycle_days", y="kits_used",
                    color="cluster_name",
                    hover_data=["element_type"] if "element_type" in df_clustered.columns else None,
                    color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"],
                    title="Element Clusters by Cycle Days vs Kits Used",
                )
                fig_cl.update_layout(**dark_chart_layout(height=360))
                fig_cl.update_traces(marker=dict(size=5, opacity=0.7))
                st.plotly_chart(fig_cl, use_container_width=True)

                # Cluster summary
                cluster_summary = df_clustered.groupby("cluster_name").agg(
                    count=("kits_used", "count"),
                    avg_kits_used=("kits_used", "mean"),
                    avg_cycle_days=("cycle_days", "mean"),
                ).reset_index().round(2)

                st.dataframe(
                    cluster_summary.rename(columns={
                        "cluster_name": "Kit Group",
                        "count": "Elements in Group",
                        "avg_kits_used": "Avg Kits Needed",
                        "avg_cycle_days": "Avg Cycle Days",
                    }),
                    hide_index=True, use_container_width=True
                )
                st.caption(f"K={cluster_summary.shape[0]} clusters fitted. Each group is a candidate for a standardized kit design.")
            else:
                st.info("Not enough data for clustering. Need at least 10 records.")
        else:
            st.info("No element_type column found in data. Repetition analysis requires element_type.")


# ═══════════════════════════════════════════════════════════════════
# TAB 4: BoQ COMPARISON
# ═══════════════════════════════════════════════════════════════════
with main_tabs[3]:
    st.markdown('<div class="section-title">BoQ Planned vs Actual Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Compares planned Bill of Quantities against actual procurement. '
        'Pinpoints which element types are consistently over-ordered and by how much.</div>',
        unsafe_allow_html=True
    )

    if st.session_state.hist_df is None:
        st.warning("Load historical data in the **Historical Kitting Data** tab first.")
    else:
        boq = analyze_boq(st.session_state.hist_df)
        if not boq:
            st.warning("BoQ analysis requires element_type, kits_planned, kits_ordered, kits_used, cost_planned, cost_actual columns.")
        else:
            # Summary KPIs
            bk1, bk2, bk3, bk4 = st.columns(4)
            with bk1: kpi("Total Planned Cost", f"₹{boq['total_planned']//100000}L", "across all elements")
            with bk2: kpi("Total Actual Cost",  f"₹{boq['total_actual']//100000}L",  "actual procurement")
            with bk3: kpi("Wastage Value",       f"₹{boq['total_waste']//100000}L",   "excess kits ordered")
            with bk4: kpi("Wastage %",           f"{boq['waste_pct']}%",              "of total procurement")

            st.markdown("---")
            elem_boq = boq["elem_boq"]

            bc1, bc2 = st.columns(2)

            with bc1:
                # Planned vs actual grouped bar
                fig_boq = go.Figure()
                fig_boq.add_trace(go.Bar(
                    name="Planned Kits", x=elem_boq["element_type"], y=elem_boq["planned_kits"],
                    marker_color="#3b82f6",
                ))
                fig_boq.add_trace(go.Bar(
                    name="Ordered Kits", x=elem_boq["element_type"], y=elem_boq["actual_kits"],
                    marker_color="#f59e0b",
                ))
                fig_boq.add_trace(go.Bar(
                    name="Used Kits", x=elem_boq["element_type"], y=elem_boq["used_kits"],
                    marker_color="#10b981",
                ))
                fig_boq.update_layout(
                    **dark_chart_layout("Kits: Planned vs Ordered vs Actually Used", 340),
                    barmode="group", xaxis_tickangle=-30,
                )
                st.plotly_chart(fig_boq, use_container_width=True)

            with bc2:
                # Cost variance chart
                fig_cv = go.Figure(go.Bar(
                    x=elem_boq["element_type"],
                    y=elem_boq["cost_var_pct"],
                    marker_color=["#ef4444" if v > 20 else "#f59e0b" if v > 10 else "#10b981"
                                  for v in elem_boq["cost_var_pct"]],
                    text=[f"{v:.1f}%" for v in elem_boq["cost_var_pct"]],
                    textposition="outside",
                ))
                fig_cv.update_layout(
                    **dark_chart_layout("Cost Variance % (Actual vs Planned)", 340),
                    xaxis_tickangle=-30, yaxis_title="Variance %"
                )
                fig_cv.add_hline(y=0, line_color="#4b5563", line_dash="dot")
                st.plotly_chart(fig_cv, use_container_width=True)

            # Detailed BoQ table
            st.markdown('<div class="section-title">Element-Level BoQ Detail</div>', unsafe_allow_html=True)
            boq_display = elem_boq.copy()
            boq_display["planned_cost"]      = boq_display["planned_cost"].apply(lambda x: f"₹ {int(x):,}")
            boq_display["actual_cost"]        = boq_display["actual_cost"].apply(lambda x: f"₹ {int(x):,}")
            boq_display["savings_potential"]  = boq_display["savings_potential"].apply(lambda x: f"₹ {int(x):,}")
            boq_display["cost_variance"]      = boq_display["cost_variance"].apply(lambda x: f"₹ {int(x):,}")

            st.dataframe(
                boq_display.rename(columns={
                    "element_type": "Element Type",
                    "planned_kits": "Kits Planned",
                    "actual_kits": "Kits Ordered",
                    "used_kits": "Kits Used",
                    "overorder_kits": "Excess Kits",
                    "overorder_pct": "Over-Order %",
                    "planned_cost": "Planned Cost",
                    "actual_cost": "Actual Cost",
                    "cost_variance": "Cost Variance",
                    "cost_var_pct": "Variance %",
                    "savings_potential": "Savings Potential",
                }),
                hide_index=True, use_container_width=True
            )

            # Savings waterfall
            fig_sw = go.Figure(go.Waterfall(
                orientation="h", measure=["relative"] * len(elem_boq) + ["total"],
                y=list(elem_boq["element_type"]) + ["TOTAL"],
                x=list(-elem_boq["savings_potential"]) + [0],
                decreasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#3b82f6"}},
            ))
            fig_sw.update_layout(
                **dark_chart_layout("Savings Potential by Element Type (₹)", 340),
                xaxis_title="Savings (₹)"
            )
            st.plotly_chart(fig_sw, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 5: INVENTORY TRACKER
# ═══════════════════════════════════════════════════════════════════
with main_tabs[4]:
    st.markdown('<div class="section-title">On-Site Inventory Tracker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Live snapshot of formwork components on site — utilization, availability, '
        'damage rates, and reorder alerts.</div>',
        unsafe_allow_html=True
    )

    inv_col1, inv_col2 = st.columns([3, 1])
    with inv_col2:
        use_demo_inv = st.checkbox("Use demo inventory data", value=True)
        if st.button("🔄 Refresh Inventory", use_container_width=True):
            st.cache_data.clear()

    inv_df = generate_sample_inventory()

    # Summary KPIs
    total_value = inv_df["inventory_value_inr"].sum()
    avg_util    = inv_df["utilization_pct"].mean()
    low_stock   = (inv_df["available"] <= inv_df["reorder_point"]).sum()
    damaged_tot = inv_df["damaged"].sum()

    ik1, ik2, ik3, ik4 = st.columns(4)
    with ik1: kpi("Inventory Value",   f"₹{total_value//100000}L", "total on-site value")
    with ik2: kpi("Avg Utilization",   f"{avg_util:.1f}%",         "components in active use")
    with ik3: kpi("Low Stock Alerts",  str(low_stock),              "below reorder point")
    with ik4: kpi("Damaged Units",     str(damaged_tot),            "requiring replacement")

    st.markdown("---")

    ic1, ic2 = st.columns(2)

    with ic1:
        # Utilization bar chart
        inv_sorted = inv_df.sort_values("utilization_pct", ascending=True)
        fig_util = go.Figure(go.Bar(
            y=inv_sorted["component"], x=inv_sorted["utilization_pct"],
            orientation="h",
            marker_color=["#ef4444" if v > 90 else "#f59e0b" if v > 70 else "#10b981"
                          for v in inv_sorted["utilization_pct"]],
            text=[f'{v:.0f}%' for v in inv_sorted["utilization_pct"]],
            textposition="outside",
        ))
        fig_util.update_layout(
            **dark_chart_layout("Component Utilization %", 380),
            xaxis_title="Utilization %"
        )
        fig_util.add_vline(x=80, line_color="#ef4444", line_dash="dash",
                           annotation_text="80% threshold", annotation_font_color="#ef4444")
        st.plotly_chart(fig_util, use_container_width=True)

    with ic2:
        # Stock status donut
        in_use_total    = inv_df["in_use"].sum()
        available_total = inv_df["available"].sum()
        damaged_total   = inv_df["damaged"].sum()

        fig_donut = go.Figure(go.Pie(
            labels=["In Use", "Available", "Damaged"],
            values=[in_use_total, available_total, damaged_total],
            hole=0.55,
            marker_colors=["#3b82f6", "#10b981", "#ef4444"],
        ))
        fig_donut.update_layout(**dark_chart_layout("Overall Stock Status", 280))
        st.plotly_chart(fig_donut, use_container_width=True)

        # Reorder alerts
        st.markdown('<div class="section-title">⚠ Reorder Alerts</div>', unsafe_allow_html=True)
        alerts = inv_df[inv_df["available"] <= inv_df["reorder_point"]]
        if not alerts.empty:
            for _, row in alerts.iterrows():
                level = "tag-red" if row["available"] == 0 else "tag-amber"
                st.markdown(
                    f'<span class="tag {level}">⚠ {row["component"]}: {row["available"]} units left '
                    f'(reorder at {row["reorder_point"]})</span><br>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown('<span class="tag tag-green">✓ All components above reorder point</span>',
                        unsafe_allow_html=True)

    # Full inventory table
    st.markdown("---")
    st.markdown('<div class="section-title">Full Inventory Register</div>', unsafe_allow_html=True)

    inv_display = inv_df.copy()
    inv_display["inventory_value_inr"] = inv_display["inventory_value_inr"].apply(lambda x: f"₹ {int(x):,}")
    inv_display["unit_cost_inr"]       = inv_display["unit_cost_inr"].apply(lambda x: f"₹ {int(x):,}")
    inv_display["status"] = inv_df.apply(
        lambda r: "🔴 Reorder" if r["available"] <= r["reorder_point"] else (
            "🟡 High Use" if r["utilization_pct"] > 80 else "🟢 OK"), axis=1
    )

    st.dataframe(
        inv_display.rename(columns={
            "component": "Component",
            "total_units": "Total",
            "in_use": "In Use",
            "available": "Available",
            "damaged": "Damaged",
            "utilization_pct": "Utilization %",
            "unit_cost_inr": "Unit Cost",
            "inventory_value_inr": "Total Value",
            "reorder_point": "Reorder At",
            "last_updated": "Last Updated",
            "status": "Status",
        }),
        hide_index=True, use_container_width=True
    )

    # Export inventory
    inv_csv = BytesIO()
    inv_df.to_csv(inv_csv, index=False)
    inv_csv.seek(0)
    st.download_button(
        "📥 Export Inventory CSV", inv_csv,
        file_name=f"inventory_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )