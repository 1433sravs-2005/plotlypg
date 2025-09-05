# app.py
import io
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio

# =========================
# Page config & CSS styling
# =========================
st.set_page_config(page_title="AI-Powered Plotly playground(APPP)", layout="wide")

# --- Global CSS: PowerBI-like card shadows & clean UI ---
CARD_CSS = """
<style>
/* App-wide tweaks */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px;}
/* Shadow card wrapper */
.shadow-card {
  background: var(--card-bg, #ffffff);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow:
     0 2px 6px rgba(0,0,0,0.06),
     0 10px 20px rgba(0,0,0,0.06);
  border: 1px solid rgba(0,0,0,0.05);
  margin-bottom: 16px;
}
/* KPI look */
.kpi {
  display:flex; flex-direction:column; gap:4px;
}
.kpi .label {font-size: 0.85rem; opacity: 0.7;}
.kpi .value {font-size: 1.6rem; font-weight: 700; line-height: 1.1;}
/* Improve expander look */
.streamlit-expanderHeader {font-weight: 600;}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

st.title("🤖 AI-Powered Plotly Playground (APPP)")
st.caption("Upload a dataset (or use demo). The inner AI cleans it, suggests charts, and you can edit them. Add charts to your final dashboard and download it as HTML.")

# =========================
# Themes & palettes
# =========================
QUAL_PALETTES = {
    "Plotly (default)": px.colors.qualitative.Plotly,
    "Bold": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel,
    "Prism": px.colors.qualitative.Prism,
    "Safe": px.colors.qualitative.Safe,
    "Vivid": px.colors.qualitative.Vivid,
    "D3": px.colors.qualitative.D3,
    "Alphabet": px.colors.qualitative.Alphabet,
}

PLOTLY_TEMPLATES = [
    "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn",
    "simple_white", "none", "presentation"
]

# Map weekday -> (template, palette)
THEME_OF_DAY = {
    0: ("plotly_white", "Bold"),       # Monday
    1: ("ggplot2", "Pastel"),          # Tuesday
    2: ("seaborn", "Prism"),           # Wednesday
    3: ("presentation", "Vivid"),      # Thursday
    4: ("plotly", "D3"),               # Friday
    5: ("plotly_dark", "Safe"),        # Saturday
    6: ("simple_white", "Alphabet"),   # Sunday
}

def todays_theme():
    wd = datetime.now().weekday()
    return THEME_OF_DAY.get(wd, ("plotly_white", "Plotly (default)"))

# Apply theme to a figure
def apply_theme(fig, template_name: str, font_family: str = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"):
    fig.update_layout(template=template_name, font=dict(family=font_family))
    return fig

# =========================
# Demo data
# =========================
def make_demo_sales(n_rows: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    regions = {
        "Asia": ["India", "China", "Japan", "Singapore", "Indonesia"],
        "Europe": ["United Kingdom", "Germany", "France", "Spain", "Italy"],
        "North America": ["United States", "Canada", "Mexico"],
        "South America": ["Brazil", "Argentina", "Chile", "Peru"],
        "Africa": ["Nigeria", "South Africa", "Egypt", "Kenya"],
        "Oceania": ["Australia", "New Zealand"],
    }
    categories = {
        "Technology": ["Laptops", "Smartphones", "Accessories", "Tablets"],
        "Furniture": ["Chairs", "Desks", "Storage", "Bookcases"],
        "Office Supplies": ["Paper", "Binders", "Art", "Labels"],
    }
    start = pd.Timestamp(2022, 1, 1)
    end = pd.Timestamp(2025, 8, 1)
    all_dates = pd.date_range(start, end, freq="D")
    rows = []
    for order_id in range(10000, 10000 + n_rows):
        region = np.random.choice(list(regions.keys()))
        country = np.random.choice(regions[region])
        category = np.random.choice(list(categories.keys()))
        subcat = np.random.choice(categories[category])
        product = f"{subcat} {np.random.randint(100,999)}"
        order_date = np.random.choice(all_dates)
        qty = int(np.random.choice([1,1,1,2,2,3,4,5], p=[0.25,0.25,0.1,0.2,0.1,0.06,0.03,0.01]))
        price = float(np.round(np.random.uniform(5, 1500), 2))
        sales = float(np.round(price * qty, 2))
        margin = float(np.round(np.random.normal(0.18, 0.15), 3))
        profit = float(np.round(sales * margin, 2))
        rows.append({
            "OrderID": order_id,
            "OrderDate": order_date,
            "Region": region,
            "Country": country,
            "Category": category,
            "SubCategory": subcat,
            "Product": product,
            "Quantity": qty,
            "UnitPrice": price,
            "Sales": sales,
            "Profit": profit,
        })
    df = pd.DataFrame(rows)
    return df

# =========================
# Data loading
# =========================
def load_uploaded(file) -> pd.DataFrame:
    """Reads CSV or Excel into DataFrame."""
    if file is None:
        return pd.DataFrame()
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(file)
        # Fallback: try csv
        return pd.read_csv(file)
    except Exception as e:
        st.warning(f"Could not read file as CSV/Excel. Error: {e}")
        return pd.DataFrame()

# =========================
# AI cleaning (rule-based)
# =========================
def auto_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    report = {}
    if df.empty:
        return df, {"status": "No data."}

    original_cols = list(df.columns)
    # 1) Trim column names
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    if original_cols != list(df.columns):
        report["column_names"] = "Trimmed whitespace in column names."

    # 2) Strip whitespace in string cells
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        try:
            df[c] = df[c].astype(str).str.strip()
        except Exception:
            pass

    # 3) Convert obvious dates or convertible objects
    date_converted = []
    for c in df.columns:
        if "date" in str(c).lower():
            df[c] = pd.to_datetime(df[c], errors="coerce")
            date_converted.append(c)
        else:
            if df[c].dtype == "object":
                try_dt = pd.to_datetime(df[c], errors="coerce")
                if try_dt.notna().mean() > 0.8:
                    df[c] = try_dt
                    date_converted.append(c)
    if date_converted:
        report["dates"] = f"Converted to datetime: {', '.join(date_converted)}"

    # 4) Convert numeric-like strings to numbers
    num_like_converted = []
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                try_num = pd.to_numeric(df[c].str.replace(",", "", regex=False), errors="coerce")
                if try_num.notna().mean() > 0.8:
                    df[c] = try_num
                    num_like_converted.append(c)
            except Exception:
                pass
    if num_like_converted:
        report["numbers"] = f"Converted to numeric: {', '.join(num_like_converted)}"

    # 5) Drop fully empty columns
    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    dropped = before_cols - df.shape[1]
    if dropped:
        report["dropped_columns"] = f"Dropped {dropped} fully-empty column(s)."

    # 6) Handle missing values
    filled_info = []
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                if df[c].isna().any():
                    med = df[c].median()
                    df[c] = df[c].fillna(med)
                    filled_info.append(f"{c}: median={med:.3f}")
            elif pd.api.types.is_datetime64_any_dtype(df[c]):
                if df[c].isna().any():
                    mode = df[c].mode()
                    fill_val = mode.iloc[0] if not mode.empty else df[c].dropna().min()
                    df[c] = df[c].fillna(fill_val)
                    filled_info.append(f"{c}: datetime fill={fill_val}")
            else:
                if df[c].isna().any():
                    mode = df[c].mode()
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                    df[c] = df[c].fillna(fill_val)
                    filled_info.append(f"{c}: mode='{fill_val}'")
        except Exception:
            pass
    if filled_info:
        report["missing"] = "Filled missing values → " + "; ".join(filled_info)

    # 7) Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    dups = before - len(df)
    if dups:
        report["duplicates"] = f"Removed {dups} duplicate row(s)."

    # 8) Add Year/Month if any datetime col exists
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    if len(dt_cols) > 0:
        dt = dt_cols[0]
        if "Year" not in df.columns:
            df["Year"] = df[dt].dt.year
        if "Month" not in df.columns:
            df["Month"] = df[dt].dt.to_period("M").dt.to_timestamp()
        report["time_features"] = f"Added Year/Month from '{dt}'."

    if not report:
        report["status"] = "Data looked clean. No changes."
    return df, report

# =========================
# Suggestions
# =========================
def suggest_charts(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return []
    nums = df.select_dtypes(include=np.number).columns.tolist()
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dts  = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()

    suggestions = []
    if dts and nums:
        suggestions.append("Line (Time vs Numeric)")
    if cats and nums:
        suggestions.append("Bar (Category vs Numeric)")
        suggestions.append("Pie (Category share)")
        suggestions.append("Box (Numeric by Category)")
    if len(nums) >= 2:
        suggestions.append("Scatter (Numeric vs Numeric)")
        suggestions.append("Histogram (Numeric)")
        suggestions.append("Correlation Heatmap")
    return suggestions[:4] or ["Bar (Category vs Numeric)"]

# =========================
# Helpers
# =========================
def pick_columns(df: pd.DataFrame):
    nums = df.select_dtypes(include=np.number).columns.tolist()
    cats = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dts  = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    return nums, cats, dts

def kpi_value(series: pd.Series, fn: str) -> Optional[float]:
    if series.empty:
        return None
    if fn == "sum":
        return float(series.sum())
    if fn == "mean":
        return float(series.mean())
    if fn == "median":
        return float(series.median())
    if fn == "min":
        return float(series.min())
    if fn == "max":
        return float(series.max())
    if fn == "count":
        return float(series.count())
    return None

# =========================
# Sidebar: Data input & Theme
# =========================
st.sidebar.header("1) Data source")
use_demo = st.sidebar.toggle("Use demo sales data", value=True)
uploaded = st.sidebar.file_uploader("…or upload CSV/Excel", type=["csv", "xlsx", "xls"])

if use_demo:
    raw_df = make_demo_sales()
else:
    raw_df = load_uploaded(uploaded)

if raw_df.empty:
    st.info("Upload a dataset or enable the demo to continue.")
    st.stop()

st.sidebar.header("2) Theme")
auto_theme = st.sidebar.toggle("Use today's theme", value=True)
if auto_theme:
    default_template, default_palette = todays_theme()
else:
    default_template, default_palette = ("plotly_white", "Plotly (default)")
template_choice = st.sidebar.selectbox("Plotly Template", PLOTLY_TEMPLATES, index=max(0, PLOTLY_TEMPLATES.index(default_template)))
palette_choice  = st.sidebar.selectbox("Palette", list(QUAL_PALETTES.keys()), index=list(QUAL_PALETTES.keys()).index(default_palette))

# Dashboard grid choice
st.sidebar.header("3) Dashboard Layout")
grid_cols = st.sidebar.slider("Columns in Final Dashboard", 1, 4, 2)

# Init session state for final dashboard
if "final_charts" not in st.session_state:
    st.session_state["final_charts"] = []
if "final_titles" not in st.session_state:
    st.session_state["final_titles"] = []

# =========================
# Clean data
# =========================
st.sidebar.header("4) Clean & preprocess")
if st.sidebar.button("Run AI Cleaning", type="primary"):
    st.session_state["cleaned"], st.session_state["report"] = auto_clean(raw_df.copy())
if "cleaned" not in st.session_state:
    st.session_state["cleaned"], st.session_state["report"] = auto_clean(raw_df.copy())

df = st.session_state["cleaned"]
report = st.session_state["report"]

with st.expander("🔎 Data preview (first 250 rows)", expanded=False):
    st.dataframe(df.head(250), use_container_width=True)

with st.expander("🧽 Cleaning report", expanded=False):
    for k, v in report.items():
        st.write(f"**{k}**: {v}")

# Download cleaned
st.download_button(
    "⬇️ Download cleaned data (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_data.csv",
    mime="text/csv",
)

st.markdown("---")

# =========================
# KPI row (auto, optional)
# =========================
nums, cats, dts = pick_columns(df)
kpi_num = nums[0] if nums else None
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="shadow-card kpi">', unsafe_allow_html=True)
        st.markdown('<div class="label">Rows</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="value">{len(df):,}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="shadow-card kpi">', unsafe_allow_html=True)
        st.markdown('<div class="label">Columns</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="value">{df.shape[1]:,}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="shadow-card kpi">', unsafe_allow_html=True)
        st.markdown('<div class="label">Numeric Fields</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="value">{len(nums)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="shadow-card kpi">', unsafe_allow_html=True)
        st.markdown('<div class="label">Categorical Fields</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="value">{len(cats)}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# =========================
# Chart selection
# =========================
st.sidebar.header("5) Choose charts")
default_suggestions = suggest_charts(df)
all_options = [
    "Bar (Category vs Numeric)",
    "Line (Time vs Numeric)",
    "Pie (Category share)",
    "Scatter (Numeric vs Numeric)",
    "Histogram (Numeric)",
    "Box (Numeric by Category)",
    "Correlation Heatmap",
]
selected_charts = st.sidebar.multiselect("Select chart types", options=all_options, default=default_suggestions)
agg_options = ["sum", "mean", "count", "median", "min", "max"]

# ---------- Utility: card wrapper ----------
def card_start(title: str):
    st.markdown('<div class="shadow-card">', unsafe_allow_html=True)
    st.markdown(f"#### {title}")

def card_end():
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Render selected charts with edit controls
# =========================
for idx_chart, chart in enumerate(selected_charts):
    card_start(chart)

    if chart == "Bar (Category vs Numeric)":
        c1, c2, c3 = st.columns(3)
        x_cat = c1.selectbox("Category (X)", cats or df.columns.tolist(), index=0 if cats else 0, key=f"bar_x_{idx_chart}")
        y_num = c2.selectbox("Numeric (Y)", nums or df.select_dtypes(include=np.number).columns.tolist(), key=f"bar_y_{idx_chart}")
        agg = c3.selectbox("Aggregation", agg_options, index=0, key=f"bar_agg_{idx_chart}")
        c4, c5 = st.columns(2)
        color = c4.selectbox("Color (optional)", ["None"] + cats, index=0, key=f"bar_color_{idx_chart}")
        local_palette = c5.selectbox("Palette (override)", ["Use global"] + list(QUAL_PALETTES.keys()), index=0, key=f"bar_pal_{idx_chart}")

        dgroup = df.groupby([x_cat] + ([color] if color != "None" else []), as_index=False)[y_num].agg(agg)
        fig = px.bar(
            dgroup, x=x_cat, y=y_num,
            color=None if color == "None" else color,
            title=f"{agg.title()} of {y_num} by {x_cat}",
            color_discrete_sequence=None if local_palette == "Use global" else QUAL_PALETTES[local_palette],
            text_auto=".2s"
        )
        fig = apply_theme(fig, template_choice)
        if local_palette == "Use global":
            fig.update_layout(colorway=QUAL_PALETTES[palette_choice])
        st.plotly_chart(fig, use_container_width=True, key=f"bar_fig_{idx_chart}")

        c_add1, c_add2 = st.columns([1, 3])
        with c_add1:
            add_title = st.text_input("Chart title (for dashboard)", value=f"Bar: {y_num} by {x_cat}", key=f"bar_title_{idx_chart}")
        with c_add2:
            if st.button("➕ Add to Dashboard", key=f"bar_add_{idx_chart}"):
                st.session_state.final_charts.append(fig)
                st.session_state.final_titles.append(add_title)
                st.success("Added to final dashboard.")

    elif chart == "Line (Time vs Numeric)":
        # Handle datasets with no datetime
        if not dts:
            st.info("No datetime column found. Try Bar/Scatter.")
            card_end(); continue
        c1, c2, c3 = st.columns(3)
        x_time = c1.selectbox("Time column (X)", dts, key=f"line_x_{idx_chart}")
        y_num  = c2.selectbox("Numeric (Y)", nums, key=f"line_y_{idx_chart}")
        color  = c3.selectbox("Group/Color (optional)", ["None"] + cats, key=f"line_color_{idx_chart}")
        freq = st.selectbox("Resample frequency", ["None","D","W","M","Q","Y"], index=2, help="D=day, W=week, M=month, etc.", key=f"line_freq_{idx_chart}")
        local_palette = st.selectbox("Palette (override)", ["Use global"] + list(QUAL_PALETTES.keys()), index=0, key=f"line_pal_{idx_chart}")

        temp = df[[x_time, y_num] + ([color] if color!="None" else [])].dropna()
        if freq != "None":
            if color == "None":
                temp = temp.set_index(x_time).resample(freq)[y_num].sum().reset_index()
                fig = px.line(temp, x=x_time, y=y_num, markers=True)
            else:
                temp = temp.set_index(x_time).groupby(color).resample(freq)[y_num].sum().reset_index()
                fig = px.line(temp, x=x_time, y=y_num, color=color, markers=True)
        else:
            fig = px.line(temp, x=x_time, y=y_num, color=None if color=="None" else color, markers=True)

        fig.update_layout(title=f"{y_num} over time")
        fig = apply_theme(fig, template_choice)
        if local_palette == "Use global":
            fig.update_layout(colorway=QUAL_PALETTES[palette_choice])
        else:
            fig.update_layout(colorway=QUAL_PALETTES[local_palette])
        st.plotly_chart(fig, use_container_width=True, key=f"line_fig_{idx_chart}")

        c_add1, c_add2 = st.columns([1, 3])
        with c_add1:
            add_title = st.text_input("Chart title (for dashboard)", value=f"Line: {y_num} over time", key=f"line_title_{idx_chart}")
        with c_add2:
            if st.button("➕ Add to Dashboard", key=f"line_add_{idx_chart}"):
                st.session_state.final_charts.append(fig)
                st.session_state.final_titles.append(add_title)
                st.success("Added to final dashboard.")

    elif chart == "Pie (Category share)":
        if not cats:
            st.info("No categorical column found for pie chart.")
            card_end(); continue
        c1, c2 = st.columns(2)
        names = c1.selectbox("Category (names)", cats, key=f"pie_names_{idx_chart}")
        values = c2.selectbox("Value (numeric)", nums, key=f"pie_vals_{idx_chart}")
        hole = st.slider("Donut hole", 0.0, 0.6, 0.3, 0.05, key=f"pie_hole_{idx_chart}")
        local_palette = st.selectbox("Palette (override)", ["Use global"] + list(QUAL_PALETTES.keys()), index=0, key=f"pie_pal_{idx_chart}")

        dsum = df.groupby(names, as_index=False)[values].sum()
        fig = px.pie(dsum, names=names, values=values, hole=hole)
        fig.update_layout(title=f"Share of {values} by {names}")
        fig = apply_theme(fig, template_choice)
        if local_palette == "Use global":
            fig.update_layout(colorway=QUAL_PALETTES[palette_choice])
        else:
            fig.update_layout(colorway=QUAL_PALETTES[local_palette])
        st.plotly_chart(fig, use_container_width=True, key=f"pie_fig_{idx_chart}")

        c_add1, c_add2 = st.columns([1, 3])
        with c_add1:
            add_title = st.text_input("Chart title (for dashboard)", value=f"Pie: {values} by {names}", key=f"pie_title_{idx_chart}")
        with c_add2:
            if st.button("➕ Add to Dashboard", key=f"pie_add_{idx_chart}"):
                st.session_state.final_charts.append(fig)
                st.session_state.final_titles.append(add_title)
                st.success("Added to final dashboard.")

    elif chart == "Scatter (Numeric vs Numeric)":
        if len(nums) < 2:
            st.info("Need at least two numeric columns for scatter.")
            card_end(); continue
        c1, c2, c3 = st.columns(3)
        x_num = c1.selectbox("X (numeric)", nums, key=f"sc_x_{idx_chart}")
        y_num = c2.selectbox("Y (numeric)", nums, index=min(1, len(nums)-1), key=f"sc_y_{idx_chart}")
        color = c3.selectbox("Color (optional)", ["None"] + cats, key=f"sc_color_{idx_chart}")
        local_palette = st.selectbox("Palette (override)", ["Use global"] + list(QUAL_PALETTES.keys()), index=0, key=f"sc_pal_{idx_chart}")

        fig = px.scatter(
            df, x=x_num, y=y_num, color=None if color=="None" else color, trendline=None
        )
        fig.update_layout(title=f"{y_num} vs {x_num}")
        fig = apply_theme(fig, template_choice)
        if local_palette == "Use global":
            fig.update_layout(colorway=QUAL_PALETTES[palette_choice])
        else:
            fig.update_layout(colorway=QUAL_PALETTES[local_palette])
        st.plotly_chart(fig, use_container_width=True, key=f"sc_fig_{idx_chart}")

        c_add1, c_add2 = st.columns([1, 3])
        with c_add1:
            add_title = st.text_input("Chart title (for dashboard)", value=f"Scatter: {y_num} vs {x_num}", key=f"sc_title_{idx_chart}")
        with c_add2:
            if st.button("➕ Add to Dashboard", key=f"sc_add_{idx_chart}"):
                st.session_state.final_charts.append(fig)
                st.session_state.final_titles.append(add_title)
                st.success("Added to final dashboard.")

    elif chart == "Histogram (Numeric)":
        if not nums:
            st.info("No numeric column found for histogram.")
            card_end(); continue
        c1, c2 = st.columns(2)
        num_col = c1.selectbox("Numeric column", nums, key=f"hist_col_{idx_chart}")
        bins = c2.slider("Bins", 5, 100, 30, key=f"hist_bins_{idx_chart}")
        local_palette = st.selectbox("Palette (override)", ["Use global"] + list(QUAL_PALETTES.keys()), index=0, key=f"hist_pal_{idx_chart}")

        fig = px.histogram(df, x=num_col, nbins=bins)
        fig.update_layout(title=f"Distribution of {num_col}")
        fig = apply_theme(fig, template_choice)
        if local_palette == "Use global":
            fig.update_layout(colorway=QUAL_PALETTES[palette_choice])
        else:
            fig.update_layout(colorway=QUAL_PALETTES[local_palette])
        st.plotly_chart(fig, use_container_width=True, key=f"hist_fig_{idx_chart}")

        c_add1, c_add2 = st.columns([1, 3])
        with c_add1:
            add_title = st.text_input("Chart title (for dashboard)", value=f"Histogram: {num_col}", key=f"hist_title_{idx_chart}")
        with c_add2:
            if st.button("➕ Add to Dashboard", key=f"hist_add_{idx_chart}"):
                st.session_state.final_charts.append(fig)
                st.session_state.final_titles.append(add_title)
                st.success("Added to final dashboard.")

    elif chart == "Box (Numeric by Category)":
        if not nums or not cats:
            st.info("Need one numeric and one categorical column for box plot.")
            card_end(); continue
        c1, c2 = st.columns(2)
        y_num = c1.selectbox("Numeric (Y)", nums, key=f"box_y_{idx_chart}")
        x_cat = c2.selectbox("Category (X)", cats, key=f"box_x_{idx_chart}")
        local_palette = st.selectbox("Palette (override)", ["Use global"] + list(QUAL_PALETTES.keys()), index=0, key=f"box_pal_{idx_chart}")

        fig = px.box(df, x=x_cat, y=y_num, color=x_cat)
        fig.update_layout(title=f"{y_num} by {x_cat}")
        fig = apply_theme(fig, template_choice)
        if local_palette == "Use global":
            fig.update_layout(colorway=QUAL_PALETTES[palette_choice])
        else:
            fig.update_layout(colorway=QUAL_PALETTES[local_palette])
        st.plotly_chart(fig, use_container_width=True, key=f"box_fig_{idx_chart}")

        c_add1, c_add2 = st.columns([1, 3])
        with c_add1:
            add_title = st.text_input("Chart title (for dashboard)", value=f"Box: {y_num} by {x_cat}", key=f"box_title_{idx_chart}")
        with c_add2:
            if st.button("➕ Add to Dashboard", key=f"box_add_{idx_chart}"):
                st.session_state.final_charts.append(fig)
                st.session_state.final_titles.append(add_title)
                st.success("Added to final dashboard.")

    elif chart == "Correlation Heatmap":
        nums_only = df.select_dtypes(include=np.number)
        if nums_only.shape[1] < 2:
            st.info("Need at least two numeric columns for correlation heatmap.")
            card_end(); continue
        corr = nums_only.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        fig.update_layout(title="Correlation Heatmap")
        fig = apply_theme(fig, template_choice)
        st.plotly_chart(fig, use_container_width=True, key=f"corr_fig_{idx_chart}")

        c_add1, c_add2 = st.columns([1, 3])
        with c_add1:
            add_title = st.text_input("Chart title (for dashboard)", value="Correlation Heatmap", key=f"corr_title_{idx_chart}")
        with c_add2:
            if st.button("➕ Add to Dashboard", key=f"corr_add_{idx_chart}"):
                st.session_state.final_charts.append(fig)
                st.session_state.final_titles.append(add_title)
                st.success("Added to final dashboard.")

    card_end()

# =========================
# Finalize Dashboard Actions
# =========================
st.markdown("---")
final_c1, final_c2, final_c3 = st.columns([1.2, 1, 2])

with final_c1:
    if st.button("✅ Finalize Dashboard", type="primary"):
        st.session_state["show_final"] = True

with final_c2:
    if st.button("🗑️ Clear Dashboard"):
        st.session_state.final_charts = []
        st.session_state.final_titles = []
        st.session_state["show_final"] = False
        st.success("Cleared saved dashboard charts.")

with final_c3:
    st.caption("Finalize to assemble your saved charts into a clean grid. You can download as a single HTML file.")

# Render Final Dashboard grid
if st.session_state.get("show_final", False):
    st.subheader("📊 Final Dashboard View")

    if st.session_state.final_charts:
        # Auto grid layout
        cols = st.columns(grid_cols)
        for i, (fig, title) in enumerate(zip(st.session_state.final_charts, st.session_state.final_titles)):
            with cols[i % grid_cols]:
                st.markdown('<div class="shadow-card">', unsafe_allow_html=True)
                st.markdown(f"**{title}**")
                # UNIQUE KEY is crucial to avoid duplicate element IDs
                st.plotly_chart(fig, use_container_width=True, theme=None, key=f"final_chart_{i}")
                st.markdown("</div>", unsafe_allow_html=True)

        # Download combined HTML
        html_parts = [pio.to_html(fig, full_html=False, include_plotlyjs=(i == 0))  # include JS only once
                      for i, fig in enumerate(st.session_state.final_charts)]
        full_html = "<html><head><meta charset='utf-8'></head><body>" + "".join(html_parts) + "</body></html>"
        st.download_button(
            label="📥 Download Dashboard (HTML)",
            data=full_html,
            file_name="dashboard.html",
            mime="text/html"
        )
    else:
        st.warning("No charts added yet. Add some charts and click Finalize again.")

st.success("✅ Done! Scroll up/down to tweak chart settings. Tip: use the demo data first, then try your CSV/Excel.")
