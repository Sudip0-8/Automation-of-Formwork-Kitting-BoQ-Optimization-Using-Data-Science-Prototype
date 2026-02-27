"""
FormOptimus - Formwork Procurement Optimizer
=============================================
Started this as a quick weekend project to automate something our site
engineers were doing manually in Excel. Grew into this.

Author: Procurement Engineering Cell, L&T B&F
Last touched: Feb 2025

NOTE TO FUTURE DEV: The ESG section is a WIP. Current multipliers are rough
estimates based on a 2022 CIDC report and one internal study from the Pune
site. Do NOT present these as certified figures to clients until we get
sign-off from the sustainability team.

TODO:
  - Replace hardcoded sigma values with project-type lookup (infra vs residential)
  - Add regional weather data pull instead of manual slider (maybe OpenWeather API?)
  - The PDF footer disclaimer needs legal review before going to clients
  - Figure out why sometimes the CVaR calc blows up on very low sim counts
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
from io import BytesIO
from typing import Dict, Any, Tuple

import plotly.graph_objects as go

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

# Keeping the CSS minimal and ONLY targeting things that don't touch
# Streamlit's internal widget DOM. Previous attempts to style .stSlider labels
# etc. were breaking because Streamlit changes its internal class names between
# versions. Lesson learned: only style custom HTML elements we inject ourselves.
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

    /* Only styling our own injected HTML blocks below — no Streamlit internals */

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

    .stDataFrame { border: 1px solid #1e2130; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ── Constants ────────────────────────────────────────────────────────────────
# These came from averaging across ~40 L&T projects 2019-2023.
# Residential vs commercial split is roughly 60/40 in the dataset.
# DO NOT treat these as universal constants - regional variation is significant.
# Source: Internal Procurement Analytics Report, Q3 2023 (ask Meenakshi for the PDF)

HOLDING_COST_RATE        = 0.045  # 4.5% per annum - standard working capital rate
WORKERS_PER_FLOOR        = 12     # avg formwork gang size (based on IS 456 crew norms)
SHIFT_HOURS              = 8
TRAD_OVERORDER_MULT      = 1.35   # traditional procurement over-orders by ~35% on average
                                   # (was 40% in first draft, field feedback revised to 35%)
KG_CO2_PER_KIT_TRANSPORT = 38     # rough figure per kit for a 20-kit truck delivery
                                   # source: one fleet manager's estimate, needs proper LCA

# Monte Carlo sigma values - this is where I'm least confident.
# Ideally fitted to historical project data. For now, engineering judgment:
#   weather sigma=3% feels right for most of India except coastal Odisha/AP
#   rework sigma=2% is conservative; real projects can swing much more
WEATHER_SIGMA = 3.0
REWORK_SIGMA  = 2.0

STRATEGY_FACTORS = {
    "Balanced":       1.00,
    "Accelerated":    1.20,  # throw kits at the problem
    "Cost-Minimized": 0.85,  # aggressive; increases schedule risk
}


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
    n_sims: int = 3000,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Monte Carlo cost simulation for formwork procurement.

    Core model: kits = base_kits × risk_factor × strategy_factor × buffer
    We sample risk_factor stochastically across n_sims iterations.

    Known limitations:
    - Normal distribution for risk; reality is probably right-skewed
    - No correlation between weather and rework (they ARE correlated in practice)
    - Labor cost treated as fixed; weather delays increase it in reality
    """

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
    w_samples = np.clip(rng.normal(weather_risk, WEATHER_SIGMA, n_sims), 0, None)
    r_samples = np.clip(rng.normal(rework_risk,  REWORK_SIGMA,  n_sims), 0, None)

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
        "optimized_kits":      optimized_kits,
        "base_kits":           base_kits,
        "saving_pct":          saving_pct,
        "cost_savings":        cost_savings,
        "total_cost_ai":       total_cost_ai,
        "manual_baseline":     manual_baseline,
        "traditional_kits":    traditional_kits,
        "mean_cost":           mean_cost,
        "std_cost":            std_cost,
        "best_case":           best_case,
        "p10":                 p10,
        "var_90":              var_90,
        "var_95":              var_95,
        "var_99":              var_99,
        "cvar_95":             cvar_95,
        "max_variance":        max_variance,
        "overrun_prob":        overrun_prob,
        "est_days":            est_days,
        "trad_days":           trad_days,
        "days_saved":          days_saved,
        "man_hours":           man_hours,
        "kit_util":            kit_util,
        "safety_stock":        safety_stock,
        "inventory_cost":      inventory_cost,
        "labor_total":         labor_total,
        "holding_cost":        holding_cost,
        "cost_per_floor":      cost_per_floor,
        "roi_pct":             roi_pct,
        "cap_efficiency":      cap_efficiency,
        "carbon_saved_kg":     carbon_saved_kg,
        "trees_equiv":         trees_equiv,
        "transport_co2_saved": transport_co2_saved,
        "waste_reduced_kg":    waste_reduced_kg,
        "water_saved_l":       water_saved_l,
    }

    return metrics, simulations


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
    elems.append(Paragraph(
        f"The AI engine recommends procuring <b>{metrics['optimized_kits']} formwork kits</b> "
        f"(vs. {metrics['traditional_kits']} kits under a 35% over-order assumption). "
        f"Projected saving: <b>₹ {int(metrics['cost_savings']):,}</b> ({metrics['saving_pct']}%). "
        f"Schedule impact: <b>{metrics['days_saved']} fewer days</b> vs. traditional timelines.",
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
        ["Budget Overrun Probability vs. Baseline", f"{metrics['overrun_prob']}%"],
        ["Cost Volatility (1σ)", c(metrics["std_cost"])],
        ["Best Case (5th percentile)", c(metrics["best_case"])],
        ["Optimistic (10th percentile)", c(metrics["p10"])],
        ["VaR @ 90%", c(metrics["var_90"])],
        ["VaR @ 95%", c(metrics["var_95"])],
        ["VaR @ 99%", c(metrics["var_99"])],
        ["Conditional VaR / Expected Shortfall (>95%)", c(metrics["cvar_95"])],
        ["Max Simulated Cost Range", c(metrics["max_variance"])],
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
        ["Water Use Reduction (rough)", f"{n(metrics['water_saved_l'])} L"],
    ]

    tbl = Table(rows, colWidths=[4.7 * inch, 2.1 * inch])
    cmd = [
        ("BACKGROUND", (0,0), (1,0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR",  (0,0), (1,0), colors.white),
        ("FONTNAME",   (0,0), (1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (1,0), 10),
        ("ALIGN",      (0,0), (0,0), "LEFT"),
        ("ALIGN",      (1,0), (1,0), "RIGHT"),
        ("BOTTOMPADDING", (0,0), (1,0), 9),
        ("TOPPADDING",    (0,0), (1,0), 9),
        ("FONTNAME", (0,1), (1,-1), "Helvetica"),
        ("FONTSIZE", (0,1), (1,-1), 8.5),
        ("ALIGN",    (0,1), (0,-1), "LEFT"),
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
                ("ALIGN",      (0,i), (1,i), "LEFT"),
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
    elems.append(Spacer(1, 0.08 * inch))
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
    """Sidebar section divider — plain HTML so it doesn't interfere with widgets."""
    st.markdown(f'<div class="sb-section">{text}</div>', unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:8px;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;color:#f1f5f9;">
        FormOptimus
    </span>
    <span style="font-family:'IBM Plex Sans',sans-serif;font-size:13px;color:#4b5563;margin-left:12px;">
        Formwork Procurement Optimizer · Monte Carlo Risk Engine
    </span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
# Deliberately NOT using st.expander here anymore. Expanders + custom CSS
# caused label clipping at certain browser zoom levels. Flat layout with
# our own section dividers is simpler and more reliable.

with st.sidebar:

    sb_header("Project Structure")
    floors = st.number_input(
        "Total Floors",
        min_value=1, max_value=100, value=30,
        help="Total above-grade floors in scope"
    )
    work_zones = st.number_input(
        "Parallel Work Zones",
        min_value=1, max_value=5, value=2,
        help="Number of concurrent active formwork zones"
    )
    cycle_days = st.number_input(
        "Floor Cycle (days)",
        min_value=4, max_value=15, value=7,
        help="Target pour-to-pour cycle time per floor"
    )

    sb_header("Risk Inputs")
    weather_risk = st.slider(
        "Weather Risk %",
        min_value=0, max_value=30, value=5,
        help="Probability uplift from adverse weather"
    )
    rework_risk = st.slider(
        "Rework Risk %",
        min_value=0, max_value=20, value=3,
        help="Rework probability based on site QA history"
    )
    safety_buf = st.slider(
        "Safety Buffer %",
        min_value=5, max_value=25, value=10,
        help="Additional buffer on top of risk-adjusted quantity"
    )

    sb_header("Cost Inputs")
    kit_cost = st.number_input(
        "Kit Cost (₹)",
        min_value=5000, max_value=500000, value=15000, step=1000,
        help="Procurement cost per formwork kit"
    )
    labor_per_floor = st.number_input(
        "Labor per Floor (₹)",
        min_value=10000, max_value=500000, value=25000, step=1000,
        help="Estimated labour cost per floor"
    )

    sb_header("Optimization Strategy")
    strategy = st.radio(
        "strategy_select",
        options=["Balanced", "Accelerated", "Cost-Minimized"],
        label_visibility="collapsed",
        help="Balanced: risk-adjusted. Accelerated: more kits, faster. Cost-Minimized: lean but higher overrun risk."
    )

    st.markdown(
        '<div class="note-box">'
        '⚠ Cost-Minimized increases schedule risk.<br>'
        'Check overrun % after running.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")  # breathing room before button
    run_btn = st.button("▶ Run Simulation", use_container_width=True, type="primary")


# ── Main panel ────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running 3,000 Monte Carlo iterations..."):
        time.sleep(0.6)  # so spinner actually renders — Streamlit's fault not ours
        metrics, sims = run_simulation(
            floors, work_zones, cycle_days,
            weather_risk, rework_risk, safety_buf,
            strategy, kit_cost, labor_per_floor
        )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Recommended Kits", str(metrics["optimized_kits"]),
                  f"base: {metrics['base_kits']} + buffer")
    with c2: kpi("Projected Saving", f"{metrics['saving_pct']}%",
                  f"₹ {int(metrics['cost_savings']/1000):,}K")
    with c3: kpi("CVaR (95%)", f"₹{round(metrics['cvar_95']/100000,1)}L",
                  "expected shortfall")
    with c4: kpi("Schedule Gain", f"{metrics['days_saved']}d",
                  f"vs. {metrics['trad_days']}d traditional")
    with c5: kpi("Overrun Risk", f"{metrics['overrun_prob']}%",
                  "vs. traditional baseline")

    st.markdown("---")

    tab_fin, tab_risk, tab_ops, tab_esg = st.tabs([
        "📊 Financials", "📈 Risk Distribution", "⚙ Operational", "🌱 ESG"
    ])

    with tab_fin:
        st.markdown('<div class="section-title">Cost Breakdown</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)

        with cc1:
            df_compare = pd.DataFrame({
                "Item": [
                    "Inventory CapEx", "Labor OpEx", "Total Cost",
                    "Baseline (Traditional)", "Projected Saving",
                    "Holding Cost (4.5% p.a.)", "ROI vs Inventory", "Capital Efficiency",
                ],
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
                orientation="v",
                measure=["absolute", "absolute", "total", "absolute", "relative"],
                x=["Inventory", "Labor", "AI Total", "Traditional", "Saving"],
                y=[
                    metrics["inventory_cost"], metrics["labor_total"], 0,
                    metrics["manual_baseline"], -metrics["cost_savings"],
                ],
                connector={"line": {"color": "#1e2130"}},
                increasing={"marker": {"color": "#3b82f6"}},
                decreasing={"marker": {"color": "#10b981"}},
                totals={"marker": {"color": "#6366f1"}},
            ))
            fig_wf.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font=dict(color="#9ca3af", family="IBM Plex Sans"),
                margin=dict(l=10, r=10, t=30, b=10), height=300,
                title=dict(text="Cost Waterfall (₹)", font=dict(size=12, color="#6b7280")),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    with tab_risk:
        st.markdown('<div class="section-title">Monte Carlo Distribution (3,000 iterations)</div>',
                    unsafe_allow_html=True)
        rc1, rc2 = st.columns([2, 1])

        with rc1:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=sims, nbinsx=70, marker_color="#3b82f6", opacity=0.65,
            ))
            fig_hist.add_vline(x=metrics["mean_cost"],      line_color="#10b981",
                                line_dash="solid",    annotation_text="Mean",
                                annotation_font_color="#10b981")
            fig_hist.add_vline(x=metrics["var_95"],         line_color="#ef4444",
                                line_dash="dash",     annotation_text="95% VaR",
                                annotation_font_color="#ef4444")
            fig_hist.add_vline(x=metrics["best_case"],      line_color="#60a5fa",
                                line_dash="dot",      annotation_text="5th pct",
                                annotation_font_color="#60a5fa")
            fig_hist.add_vline(x=metrics["manual_baseline"], line_color="#f59e0b",
                                line_dash="longdash", annotation_text="Baseline",
                                annotation_font_color="#f59e0b")
            fig_hist.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font=dict(color="#9ca3af", family="IBM Plex Sans"),
                xaxis_title="Simulated Total Cost (₹)", yaxis_title="Frequency",
                showlegend=False, margin=dict(l=10, r=10, t=20, b=10), height=340,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with rc2:
            st.markdown('<div class="section-title">Percentile Table</div>',
                        unsafe_allow_html=True)
            df_pct = pd.DataFrame({
                "Percentile": ["5th (best)", "10th", "50th (median)",
                                "90th", "95th (VaR)", "99th", "CVaR >95%"],
                "Cost (₹)": [
                    f"₹ {metrics['best_case']:,}",
                    f"₹ {metrics['p10']:,}",
                    f"₹ {int(np.median(sims)):,}",
                    f"₹ {metrics['var_90']:,}",
                    f"₹ {metrics['var_95']:,}",
                    f"₹ {metrics['var_99']:,}",
                    f"₹ {metrics['cvar_95']:,}",
                ]
            })
            st.dataframe(df_pct, hide_index=True, use_container_width=True)
            st.markdown("**Budget overrun risk:**")
            st.markdown(risk_badge(metrics["overrun_prob"]), unsafe_allow_html=True)
            st.caption(f"Probability of exceeding ₹ {int(metrics['manual_baseline']):,} baseline")
            st.caption(f"Cost volatility (σ): ₹ {int(metrics['std_cost']):,}")

    with tab_ops:
        oc1, oc2, oc3 = st.columns(3)

        with oc1:
            st.markdown('<div class="section-title">Schedule</div>', unsafe_allow_html=True)
            st.metric("Estimated Duration",   f"{metrics['est_days']} days")
            st.metric("Traditional Estimate", f"{metrics['trad_days']} days")
            st.metric("Schedule Saving",      f"{metrics['days_saved']} days")
            st.metric("Total Man-Hours",      f"{metrics['man_hours']:,}")

        with oc2:
            st.markdown('<div class="section-title">Inventory</div>', unsafe_allow_html=True)
            st.metric("Recommended Kits",  metrics["optimized_kits"])
            st.metric("Safety Stock",      f"{metrics['safety_stock']} kits")
            st.metric("Kit Utilization",   f"{metrics['kit_util']}%")
            st.metric("Holding Cost",      f"₹ {metrics['holding_cost']:,}")

        with oc3:
            st.markdown('<div class="section-title">Strategy Notes</div>', unsafe_allow_html=True)
            notes = {
                "Balanced":       "Risk-adjusted quantity. Suitable for most projects.",
                "Accelerated":    "20% uplift on kits. Lower schedule risk, higher CapEx.",
                "Cost-Minimized": "Lean procurement. Lower holding cost, higher overrun risk.",
            }
            st.info(notes[strategy])
            if strategy == "Cost-Minimized" and metrics["overrun_prob"] > 30:
                st.warning(
                    f"Overrun probability is **{metrics['overrun_prob']}%**. "
                    "Consider switching to Balanced for this risk profile."
                )

    with tab_esg:
        st.markdown("""
        <div class="note-box">
        These are rough estimates based on internal benchmarks and CIDC 2022.
        <b>Do not cite in client sustainability reports</b> without sustainability cell review.
        </div>
        """, unsafe_allow_html=True)
        ec1, ec2 = st.columns(2)
        with ec1:
            st.metric("Carbon Saving (est.)",       f"{metrics['carbon_saved_kg']:,} kg CO₂e")
            st.metric("Transport Emissions Avoided", f"{metrics['transport_co2_saved']:,} kg CO₂e")
            st.metric("Ecological Offset (rough)",   f"≈ {metrics['trees_equiv']:,} trees/yr")
        with ec2:
            st.metric("Waste Avoided",    f"{metrics['waste_reduced_kg']:,} kg")
            st.metric("Water Savings",    f"{metrics['water_saved_l']:,} L")
            st.caption(
                "Carbon: ~0.021 kg CO₂e/₹ avoided spend (CIDC 2022, Table 4). "
                "Water factor: ±50% uncertainty. "
                "Waste: 150 kg/over-ordered kit, 3-project observation."
            )

    st.markdown("---")
    st.markdown('<div class="section-title">Export</div>', unsafe_allow_html=True)
    pdf_buf = build_pdf(metrics, {"strategy": strategy, "floors": floors, "work_zones": work_zones})
    st.download_button(
        "Download PDF Report", pdf_buf,
        file_name=f"FormOptimus_{strategy}_{floors}fl.pdf",
        mime="application/pdf",
    )
    st.caption("Includes all metrics, methodology notes, and ESG disclaimers.")

else:
    st.markdown("""
    <div style="text-align:center;padding:80px 0;color:#374151;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:48px;margin-bottom:16px;color:#1e2130;">[ ]</div>
        <div style="font-size:14px;color:#4b5563;">Configure parameters in the sidebar and click <b>Run Simulation</b>.</div>
    </div>
    """, unsafe_allow_html=True)