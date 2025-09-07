# app.py
# Periodization Coach — All-in-One (Coach-first UI with Insights side page)
# Python 3.10+
# -----------------------------------------------------------------------------
# Features
# - Main page: 8 large buttons for workouts; on click: show full per-set coaching (all exercises).
# - Side page ("Insights"): weekly muscle volume (LOW/OK/HIGH), 1RM library, safety rules.
# - Upload Strong CSV (Date, Exercise Name, Weight, Reps) + optional Advice Excel (sheet "advice").
# - Advice precedence (ISO Year/Week/Workout/Exercise/SetNumber), else fallbacks by muscle & status.
# - DUP scaling (H/M/L) adjusts %1RM on the Coach page.
# - Rounded target loads (DB 2.0 kg, bar 2.5 kg), dumbbell/bar inference by exercise name.
# - A/B alternation by ISO week parity remains supported; user can override by picking any workout.
# -----------------------------------------------------------------------------

import io
from datetime import datetime, date
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------- CONFIG --------------------------------------------------

TARGET_MIN = 10
TARGET_MAX = 20
ROLLING_WEEKS = 4

DB_STEP = 2.0
BAR_STEP = 2.5

WORKOUTS = ["1A Push","2A Push","1B Push","2B Push","1A Pull","2A Pull","1B Pull","2B Pull"]

TEMPLATES = {
    # --- PUSH ---
    "1A Push": [
        ("Incline Bench Press (Dumbbell)", 2),
        ("Chest Fly (Cable)", 2),
        ("Lateral Raise (Dumbbell)", 2),
        ("Tricep Pushdown (Cable)", 2),
        ("Skullcrusher (Dumbbell)", 2),
        ("Cable Crunch", 2),
    ],
    "2A Push": [
        ("Bench Press (Dumbbell)", 3),
        ("Chest Fly (Dumbbell)", 2),
        ("Lateral Raise (Dumbbell)", 2),
        ("Tricep Pushdown (Cable - Straight Bar)", 2),
        ("Cable Crunch", 2),
    ],
    "1B Push": [
        ("Bench Press (Cable)", 3),
        ("Overhead Press (Dumbbell)", 1),  # trimmed as discussed
        ("Lateral Raise (Dumbbell)", 2),
        ("Chest Fly (Dumbbell)", 2),
        ("Tricep Pushdown (Cable)", 2),
        ("Triceps Extension (Cable)", 2),
        ("Hanging Knee Raise", 2),
    ],
    "2B Push": [
        ("Incline Bench Press (Dumbbell)", 3),
        ("Seated Overhead Press (Dumbbell)", 1),  # trimmed as discussed
        ("Lateral Raise (Cable)", 2),
        ("Triceps Extension (Cable)", 2),
        ("Cable Crunch", 2),
    ],
    # --- PULL ---
    "1A Pull": [
        ("Lat Pulldown (Cable)", 3),            # +1 applied
        ("Incline Row (Dumbbell)", 3),
        ("Cable Crunch", 2),
        ("Reverse Fly (Cable)", 2),
        ("Incline DB Hammer Curl", 3),          # upgraded
    ],
    "2A Pull": [
        ("Lat Pulldown (Cable)", 2),
        ("Incline Row (Dumbbell)", 3),          # +1 applied
        ("Cable Crunch", 2),
        ("Y-Raise", 2),                         # swap for DB reverse fly
        ("Bicep Curl (Dumbbell)", 3),           # upgraded
    ],
    "1B Pull": [
        ("Lat Pulldown (Cable)", 2),
        ("Seated Row (Cable)", 3),              # +1 applied
        ("Face Pull (Cable)", 3),               # +1 applied
        ("Cable Crunch", 2),
        ("LOW CABLE BICEP CURL (Cable)", 3),    # upgraded
    ],
    "2B Pull": [
        ("Lat Pulldown (Cable)", 3),            # +1 applied
        ("Seated Row (Cable)", 2),
        ("Hanging Knee Raise", 2),
        ("Face Pull (Cable)", 2),
        ("EZ BAR CURL (Cable)", 3),             # upgraded
    ],
}

MUSCLE_MAP = {
    # Chest
    "Bench Press (Dumbbell)": "Chest",
    "Bench Press (Cable)": "Chest",
    "Incline Bench Press (Dumbbell)": "Chest",
    "Chest Fly (Dumbbell)": "Chest",
    "Chest Fly (Cable)": "Chest",
    # Shoulders
    "Overhead Press (Dumbbell)": "Shoulders",
    "Seated Overhead Press (Dumbbell)": "Shoulders",
    "Arnold Press (Dumbbell)": "Shoulders",
    "Front Raise (Cable)": "Shoulders",
    "Lateral Raise (Dumbbell)": "Shoulders",
    "Lateral Raise (Cable)": "Shoulders",
    "Reverse Fly (Dumbbell)": "Shoulders",
    "Reverse Fly (Cable)": "Shoulders",
    "Face Pull (Cable)": "Shoulders",
    "Y-Raise": "Shoulders",
    # Triceps
    "Tricep Pushdown (Cable)": "Triceps",
    "Tricep Pushdown (Cable - Straight Bar)": "Triceps",
    "Triceps Extension (Cable)": "Triceps",
    "Skullcrusher (Dumbbell)": "Triceps",
    # Back
    "Lat Pulldown (Cable)": "Back",
    "Incline Row (Dumbbell)": "Back",
    "Seated Row (Cable)": "Back",
    # Biceps
    "Incline DB Hammer Curl": "Biceps",
    "Bicep Curl (Dumbbell)": "Biceps",
    "LOW CABLE BICEP CURL (Cable)": "Biceps",
    "EZ BAR CURL (Cable)": "Biceps",
    # Core
    "Cable Crunch": "Core",
    "Hanging Knee Raise": "Core",
}

FALLBACK = {
    # Defaults aim at hypertrophy with your %1RM preference
    "Chest":     {"TopPct": 80, "TopReps": 6,  "BackPct": 70, "BackReps": 10},
    "Back":      {"TopPct": 75, "TopReps": 8,  "BackPct": 65, "BackReps": 12},
    "Shoulders": {"TopPct": 72, "TopReps": 8,  "BackPct": 60, "BackReps": 15},
    "Triceps":   {"TopPct": 70, "TopReps":10,  "BackPct": 62, "BackReps": 14},
    "Biceps":    {"TopPct": 70, "TopReps":10,  "BackPct": 62, "BackReps": 14},
    "Core":      {"TopPct": 80, "TopReps":10,  "BackPct": 65, "BackReps": 15},
    "Other":     {"TopPct": 70, "TopReps":10,  "BackPct": 60, "BackReps": 12},
}

DUP_SCALE = {"H": 1.00, "M": 0.96, "L": 0.92}

# ----------------------- HELPERS -------------------------------------------------

def to_date(s):
    if isinstance(s, (date, datetime)): return pd.to_datetime(s).date()
    return pd.to_datetime(s, errors="coerce").date()

def is_dumbbell(name: str) -> bool:
    return ("Dumbbell" in name) or ("DB" in name)

def round_load(x: float, is_db: bool) -> float:
    step = DB_STEP if is_db else BAR_STEP
    return round(round(x / step) * step, 2)

# 1RM estimators
def epley_1rm(w, r):   return w * (1 + r / 30)
def brzycki_1rm(w, r): return (w * (36 / (37 - r))) if r < 37 else np.nan
def blended_1rm(w, r):
    if r <= 1: return w
    return np.nanmean([epley_1rm(w, r), brzycki_1rm(w, r)])

def best_1rm(df: pd.DataFrame, weeks=8) -> pd.Series:
    if df.empty: return pd.Series(dtype=float)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = df["Date"].max() - pd.Timedelta(weeks=weeks)
    df = df[df["Date"] >= cutoff]
    df["est_1rm"] = df.apply(lambda r: blended_1rm(r["Weight"], r["Reps"]), axis=1)
    return df.groupby("Exercise Name")["est_1rm"].max()

def status_tag(n_sets: float) -> str:
    if n_sets < TARGET_MIN: return "LOW"
    if n_sets > TARGET_MAX: return "HIGH"
    return "OK"

def coach_for_set(muscle: str, set_idx: int, muscle_status: Dict[str,str]):
    base = FALLBACK.get(muscle, FALLBACK["Other"]).copy()
    status = muscle_status.get(muscle, "OK")
    if status == "LOW":
        base["TopPct"] += 2; base["BackPct"] += 2; base["BackReps"] += 2
    elif status == "HIGH":
        base["TopPct"] -= 3; base["BackPct"] -= 3; base["BackReps"] -= 2
    if set_idx == 1: return base["TopPct"], base["TopReps"]
    return base["BackPct"], base["BackReps"]

# ----------------------- APP LAYOUT ---------------------------------------------

st.set_page_config(page_title="Periodization Coach — All-in-One", layout="wide")

# Sidebar (uploads + selections)
st.sidebar.header("Uploads")
strong_file = st.sidebar.file_uploader("Strong CSV", type=["csv"])
advice_file = st.sidebar.file_uploader("Advice Excel (sheet: advice)", type=["xlsx"])

st.sidebar.header("Parameters")
today = pd.Timestamp.today()
default_year = int(today.isocalendar().year)
default_week = int(today.isocalendar().week)

year = st.sidebar.number_input("ISO Year", min_value=2000, max_value=2100, value=default_year, step=1)
week = st.sidebar.number_input("ISO Week", min_value=1, max_value=53, value=default_week, step=1)
dup_choice = st.sidebar.selectbox("DUP (H/M/L)", ["H","M","L"], index=0)

st.sidebar.markdown("---")
st.sidebar.caption("Ma: Push • Di: Pull • Do: Push • Vr/Zat: Pull")

# Simple page switcher (Coach vs Insights)
page = st.sidebar.radio("Pagina", ["Coach", "Insights"], index=0)

# ----------------------- DATA LOADING -------------------------------------------

df = pd.DataFrame()
advice_df = None
if strong_file is not None:
    df = pd.read_csv(strong_file)
    # Validate
    required = {"Date","Exercise Name","Weight","Reps"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"CSV mist verplichte kolommen: {missing}")
        st.stop()
    # Normalize
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Exercise Name"] = df["Exercise Name"].astype(str)
    df["Reps"] = pd.to_numeric(df["Reps"], errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["ISO_Year"] = df["Date"].dt.isocalendar().year.astype(int)
    df["ISO_Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Muscle"] = df["Exercise Name"].map(MUSCLE_MAP).fillna("Other")

    # Volume per week per spiergroep
    vol = (
        df.assign(Set=1)
          .groupby(["ISO_Year","ISO_Week","Muscle"], as_index=False)["Set"].sum()
    )
else:
    vol = pd.DataFrame(columns=["ISO_Year","ISO_Week","Muscle","Set"])

if advice_file is not None:
    try:
        advice_df = pd.read_excel(advice_file, sheet_name="advice")
        # Normalize numeric columns
        for c in ["ISO_Year","ISO_Week","SetNumber","%1RM","Reps","SetsOverride"]:
            if c in advice_df.columns:
                advice_df[c] = pd.to_numeric(advice_df[c], errors="coerce")
        if "Workout" in advice_df.columns:
            advice_df["Workout"] = advice_df["Workout"].astype(str)
        if "Exercise" in advice_df.columns:
            advice_df["Exercise"] = advice_df["Exercise"].astype(str)
    except Exception as e:
        st.error(f"Kon sheet 'advice' niet lezen: {e}")

# Build weekly status table for selected week
week_summary = pd.DataFrame({"Muscle": sorted(set(MUSCLE_MAP.values())), "Set": 0})
if not vol.empty:
    sel = vol[(vol["ISO_Year"] == year) & (vol["ISO_Week"] == week)]
    if not sel.empty:
        week_summary = sel.groupby("Muscle", as_index=False)["Set"].sum()
        # ensure all muscles present
        all_m = sorted(set(MUSCLE_MAP.values()))
        week_summary = (week_summary
                        .set_index("Muscle")
                        .reindex(all_m, fill_value=0)
                        .reset_index())
week_summary["Status"] = week_summary["Set"].apply(status_tag)
muscle_status_dict = {r["Muscle"]: r["Status"] for _, r in week_summary.iterrows()}

# 1RM library (last 8 weeks)
ones = best_1rm(df) if not df.empty else pd.Series(dtype=float)

# ----------------------- COACH PAGE ---------------------------------------------

def lookup_advice(ex: str, workout: str, set_idx: int) -> Tuple[float, int, int | None]:
    """
    Returns (%1RM, reps, sets_override) if present in advice_df for current year/week, else (np.nan, None, None).
    Set-specific advice preferred; if SetsOverride present (any row for this ex/workout), return it too.
    """
    if advice_df is None: return np.nan, None, None
    q = advice_df[
        ((advice_df["ISO_Year"].isna()) | (advice_df["ISO_Year"] == year)) &
        ((advice_df["ISO_Week"].isna()) | (advice_df["ISO_Week"] == week)) &
        ((advice_df["Workout"].isna())  | (advice_df["Workout"] == workout)) &
        (advice_df["Exercise"] == ex)
    ].copy()
    if q.empty: return np.nan, None, None

    set_match = q[q["SetNumber"] == set_idx]
    row = set_match.iloc[0] if not set_match.empty else q.iloc[0]

    pct = float(row["%1RM"]) if not np.isnan(row.get("%1RM", np.nan)) else np.nan
    reps = int(row["Reps"])   if not np.isnan(row.get("Reps", np.nan))   else None

    sets_override = None
    qo = q[q["SetsOverride"].notna()]
    if not qo.empty:
        sets_override = int(qo.iloc[0]["SetsOverride"])
    return pct, reps, sets_override

def build_plan_for_workout(workout: str, dup: str) -> pd.DataFrame:
    rows = []
    template = TEMPLATES.get(workout, [])
    scale = DUP_SCALE.get(dup, 1.0)

    # first pass: read sets_override from advice if provided
    ex_to_sets = {}
    for ex, base_sets in template:
        _, _, sets_override = lookup_advice(ex, workout, set_idx=1)
        ex_to_sets[ex] = sets_override if sets_override is not None else base_sets

    for ex, _ in template:
        n_sets = ex_to_sets[ex]
        muscle = MUSCLE_MAP.get(ex, "Other")
        est1rm = ones.get(ex, np.nan)

        for s in range(1, n_sets + 1):
            adv_pct, adv_reps, _ = lookup_advice(ex, workout, s)
            if not np.isnan(adv_pct) and adv_reps is not None:
                pct, reps = adv_pct, adv_reps
            else:
                pct, reps = coach_for_set(muscle, s, muscle_status_dict)

            pct = pct * scale
            load = np.nan
            if not np.isnan(est1rm):
                load = round_load(est1rm * (pct / 100.0), is_dumbbell(ex))

            rows.append({
                "Exercise": ex,
                "Set": s,
                "%1RM": round(pct, 1),
                "Reps": reps,
                "Target Load (kg)": None if np.isnan(load) else load,
                "Est. 1RM (kg)": None if np.isnan(est1rm) else round(float(est1rm), 1),
                "Muscle": muscle,
                "RPE/RIR hint": "Top-set ≈RPE8 (RIR~2)" if s == 1 else "Back-off ≈RPE7–8 (RIR~2–3)",
            })
    dfw = pd.DataFrame(rows)
    dfw = dfw.sort_values(["Exercise","Set"]).reset_index(drop=True)
    return dfw

def big_button(label: str, key: str):
    c = st.container()
    with c:
        st.markdown(
            f"""
            <style>
            .btn-{key} button {{
                border: 2px solid #1f77b4;
                border-radius: 10px;
                padding: 18px 12px;
                font-size: 18px;
                width: 100%;
                height: 78px;
                background: white;
            }}
            .btn-{key} button:hover {{
                background: #eaf3ff;
            }}
            </style>
            """, unsafe_allow_html=True
        )
        return st.container().button(label, key=key, use_container_width=True)

if page == "Coach":
    st.title("Workout Coach")

    if strong_file is None:
        st.info("Upload Strong CSV en (optioneel) Advice Excel in de sidebar.")
        st.stop()

    st.markdown("**Kies je workout**")
    # Grid met 8 knoppen (4 x 2)
    sel = None
    rows = [
        ["1A Push","2A Push","1B Push","2B Push"],
        ["1A Pull","2A Pull","1B Pull","2B Pull"]
    ]
    for r in rows:
        cols = st.columns(len(r), gap="large")
        for i, w in enumerate(r):
            with cols[i]:
                if st.button(w, key=f"btn_{w}", use_container_width=True):
                    sel = w

    # Als geen knop is gedrukt: voorstel op basis van weekpariteit (A/B)
    if sel is None:
        is_even = (week % 2 == 0)
        sel = "1B Push" if is_even else "1A Push"

    st.subheader(f"🧭 Sessie: {sel}  •  Week {week}  •  DUP: {dup_choice}")
    st.caption("Advies uit Excel prevaleert; bij geen match vallen we terug op spiergroep-fallbacks en weekstatus (LOW/OK/HIGH).")

    plan = build_plan_for_workout(sel, dup_choice)

    # Weergave: alle oefeningen tegelijk, per oefening in een 'card'
    for ex in plan["Exercise"].unique():
        sub = plan[plan["Exercise"] == ex].copy()
        m = sub["Muscle"].iloc[0]
        st.markdown(f"#### {ex}  —  *{m}*")
        # Compacte tabel: Set | %1RM | Reps | Target Load | Est.1RM | Hint
        show = sub[["Set","%1RM","Reps","Target Load (kg)","Est. 1RM (kg)","RPE/RIR hint"]]
        st.dataframe(show, hide_index=True, use_container_width=True)

    # Export voor gym-quickview
    out = plan[["Exercise","Set","%1RM","Reps","Target Load (kg)"]].copy()
    buf = io.StringIO(); out.to_csv(buf, index=False)
    st.download_button("Export sessie (CSV)", data=buf.getvalue(),
                       file_name=f"{sel.replace(' ','_').lower()}_wk{week}_plan.csv", mime="text/csv")

# ----------------------- INSIGHTS PAGE ------------------------------------------

if page == "Insights":
    st.title("Insights")
    st.markdown("Overzicht van **weekvolume per spiergroep**, **LOW/OK/HIGH**, **1RM-bibliotheek**, en **veiligheidsregels**.")

    st.subheader(f"Weekvolume per spiergroep — week {week}, {year}")
    if week_summary.empty:
        st.info("Geen sets gevonden voor de geselecteerde week in de Strong CSV.")
    st.dataframe(week_summary.sort_values("Muscle").reset_index(drop=True), use_container_width=True)

    if not week_summary.empty:
        fig = px.bar(week_summary, x="Muscle", y="Set", color="Status",
                     color_discrete_map={"LOW":"#F39C12","OK":"#27AE60","HIGH":"#E74C3C"},
                     title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Geschatte 1RM per oefening (beste uit laatste 8 weken)")
    if ones.empty:
        st.info("Nog geen 1RM-schattingen beschikbaar (upload Strong CSV).")
    else:
        disp = ones.sort_values(ascending=False).reset_index()
        disp.columns = ["Exercise","Est_1RM (kg)"]
        st.dataframe(disp, use_container_width=True)

    st.subheader("Veiligheidsregels")
    st.markdown(
        "- LOW/OK/HIGH vergeleken met **10–20 sets/week**.\n"
        "- **DUP** schaalt %1RM (H: 100%, M: 96%, L: 92%).\n"
        "- Fallback-coaching per spiergroep, aangepast op weekstatus.\n"
        "- Loads afgerond op **DB 2.0 kg** / **bar 2.5 kg**.\n"
        "- **4-week rolling average** begrenst volume-sprongen."
    )
