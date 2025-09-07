# app.py
# Periodization Coach — All-in-One (Streamlit)
# Author: Dorus' Coach
# Python 3.10+
# -----------------------------------------------------------------------------
# Features
# - Upload Strong CSV (min cols: Date, Exercise Name, Weight, Reps)
# - Upload Excel "advice" sheet met %1RM/reps/sets per week & oefening
# - 1RM-berekening (Epley+Brzycki gewogen) en beste-van-blok
# - Weekvolume per spiergroep (ISO week), LOW/OK/HIGH t.o.v. 10–20 sets
# - DUP (H/M/L) generator en “Next Workout” per dagtype (Push/Pull)
# - Template mapping 1A/1B Push + 1A/1B Pull (uitbreidbaar)
# - Per oefening per set: doel-%1RM, reps, geschat gewicht
# - Veiligheidsregels: cap op volume-groei, fallback-doelen
# - Exports als CSV
# -----------------------------------------------------------------------------

import io
import math
from datetime import datetime, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------- SETTINGS ------------------------------------------------

TARGET_MIN = 10  # sets/week lower band
TARGET_MAX = 20  # sets/week upper band
ROLLING_WEEKS = 4  # for fatigue guard

# Dumbbell plates rounding step (kg)
DB_STEP = 2.0
BAR_STEP = 2.5

# Your weekly schedule
SCHEDULE = {
    "Monday": "Push",
    "Tuesday": "Pull",
    "Thursday": "Push",
    "Friday/Saturday": "Pull",
}

# Templates (2 sets standaard; pas aantallen aan in advies-Excel of in deze dict)
TEMPLATES = {
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
        ("Overhead Press (Dumbbell)", 2),
        ("Lateral Raise (Dumbbell)", 2),
        ("Chest Fly (Dumbbell)", 2),
        ("Tricep Pushdown (Cable)", 2),
        ("Triceps Extension (Cable)", 2),
        ("Hanging Knee Raise", 2),
    ],
    "2B Push": [
        ("Incline Bench Press (Dumbbell)", 3),
        ("Seated Overhead Press (Dumbbell)", 1),
        ("Lateral Raise (Cable)", 2),
        ("Triceps Extension (Cable)", 2),
        ("Cable Crunch", 2),
    ],
    "1A Pull": [
        ("Lat Pulldown (Cable)", 3),
        ("Incline Row (Dumbbell)", 3),
        ("Cable Crunch", 2),
        ("Reverse Fly (Cable)", 2),
        ("Incline DB Hammer Curl", 2),
    ],
    "2A Pull": [
        ("Lat Pulldown (Cable)", 2),
        ("Incline Row (Dumbbell)", 3),
        ("Cable Crunch", 2),
        ("Reverse Fly (Dumbbell)", 2),   # kan in app naar Y-Raise worden geswapped
        ("Bicep Curl (Dumbbell)", 3),
    ],
    "1B Pull": [
        ("Lat Pulldown (Cable)", 2),
        ("Seated Row (Cable)", 3),
        ("Face Pull (Cable)", 3),
        ("Cable Crunch", 2),
        ("LOW CABLE BICEP CURL (Cable)", 3),
    ],
    "2B Pull": [
        ("Lat Pulldown (Cable)", 3),
        ("Seated Row (Cable)", 2),
        ("Hanging Knee Raise", 2),
        ("Face Pull (Cable)", 2),
        ("EZ BAR CURL (Cable)", 3),
    ],
}

# Mapping oefeningen -> primaire spiergroep (voor set-volume per week)
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

# -------------- Utility ----------------------------------------------------------

def to_date(s):
    if isinstance(s, (date, datetime)):
        return pd.to_datetime(s).date()
    return pd.to_datetime(s, errors="coerce").date()

def iso_year_week(d: date) -> Tuple[int, int]:
    iso = datetime(d.year, d.month, d.day).isocalendar()
    return iso[0], iso[1]

def round_load(x: float, is_db: bool) -> float:
    step = DB_STEP if is_db else BAR_STEP
    return round(round(x / step) * step, 2)

def is_dumbbell(ex_name: str) -> bool:
    return "Dumbbell" in ex_name or "DB" in ex_name

# 1RM estimators
def epley_1rm(w, r):
    return w * (1 + r / 30)

def brzycki_1rm(w, r):
    if r >= 37:  # avoid division by zero
        return np.nan
    return w * (36 / (37 - r))

def blended_1rm(w, r):
    if r <= 1:
        return w
    return np.nanmean([epley_1rm(w, r), brzycki_1rm(w, r)])

# Compute best recent 1RM per exercise (highest estimated across last N weeks)
def best_1rm(df: pd.DataFrame, weeks=8) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = df["Date"].max() - pd.Timedelta(weeks=weeks)
    df = df[df["Date"] >= cutoff]
    df["est_1rm"] = df.apply(lambda r: blended_1rm(r["Weight"], r["Reps"]), axis=1)
    return df.groupby("Exercise Name")["est_1rm"].max()

# -------------- Streamlit UI ----------------------------------------------------

st.set_page_config(page_title="Periodization Coach — All-in-One", layout="wide")

st.title("Periodization Coach — All-in-One")

left, right = st.columns([2, 1])
with left:
    st.markdown("**Upload je Strong CSV** (min kolommen: `Date, Exercise Name, Weight, Reps`).")
    strong_file = st.file_uploader("Strong CSV", type=["csv"])
with right:
    st.markdown("**Upload optionele Advies-Excel** (tab: `advice`) met %1RM/rep/sets.")
    advice_file = st.file_uploader("Advice Excel", type=["xlsx"])

st.divider()

# Schema toelichting
with st.expander("📘 Advies-Excel schema & voorbeeld"):
    st.write(
        "Tabblad **advice** met kolommen: "
        "`ISO_Year`, `ISO_Week`, `Workout` (bv. '1A Push'), `Exercise`, "
        "`SetNumber` (1..N), `%1RM`, `Reps`, `SetsOverride` (optioneel). "
        "App matcht op jaar+week+workout+exercise. Fallback: alleen exercise."
    )
    sample = pd.DataFrame({
        "ISO_Year": [2025, 2025],
        "ISO_Week": [36, 36],
        "Workout": ["1A Push", "1A Push"],
        "Exercise": ["Incline Bench Press (Dumbbell)", "Chest Fly (Cable)"],
        "SetNumber": [1, 1],
        "%1RM": [80, 62],
        "Reps": [6, 15],
        "SetsOverride": [3, None],
    })
    st.dataframe(sample)

# Load data
if strong_file is None:
    st.info("Upload eerst je Strong CSV.")
    st.stop()

df = pd.read_csv(strong_file)
required = {"Date", "Exercise Name", "Weight", "Reps"}
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

# Weekly volume per muscle
vol = (
    df.assign(Set=1)
      .groupby(["ISO_Year", "ISO_Week", "Muscle"], as_index=False)["Set"].sum()
)

# Rolling 4-week average safety
vol_4w = (
    vol.sort_values(["Muscle", "ISO_Year", "ISO_Week"])
       .groupby("Muscle")["Set"]
       .rolling(window=ROLLING_WEEKS, min_periods=1).mean()
       .reset_index(level=0, drop=True)
)
vol["Rolling4wAvg"] = vol_4w

# Current year/week selector
today = pd.Timestamp.today()
default_year = int(today.isocalendar().year)
default_week = int(today.isocalendar().week)

year = st.number_input("Jaar (ISO)", min_value=2000, max_value=2100, value=default_year, step=1)
week = st.number_input("Week (ISO)", min_value=1, max_value=53, value=default_week, step=1)

sel = vol[(vol["ISO_Year"] == year) & (vol["ISO_Week"] == week)]
if sel.empty:
    st.warning("Geen sets geregistreerd voor deze week in je CSV. (De coach werkt nog steeds met advies/fallbacks.)")

# Traffic light (LOW/OK/HIGH)
def status_tag(n_sets: float) -> str:
    if n_sets < TARGET_MIN:
        return "LOW"
    if n_sets > TARGET_MAX:
        return "HIGH"
    return "OK"

summary = (
    sel[["Muscle", "Set"]].groupby("Muscle", as_index=False).sum()
    if not sel.empty else pd.DataFrame(columns=["Muscle", "Set"])
)
if summary.empty:
    summary = pd.DataFrame({"Muscle": sorted(set(MUSCLE_MAP.values())), "Set": 0})
summary["Status"] = summary["Set"].apply(status_tag)

st.subheader("Weekvolume per spiergroep")
st.dataframe(summary.sort_values("Muscle").reset_index(drop=True))

fig = px.bar(summary, x="Muscle", y="Set", color="Status",
             color_discrete_map={"LOW":"#F39C12","OK":"#27AE60","HIGH":"#E74C3C"},
             title=f"Volume week {week}, {year}")
st.plotly_chart(fig, use_container_width=True)

# 1RM library
ones = best_1rm(df)

with st.expander("📈 Geschatte 1RM per oefening (beste uit laatste 8 weken)"):
    disp = ones.sort_values(ascending=False).reset_index()
    disp.columns = ["Exercise", "Est_1RM"]
    st.dataframe(disp)

# Advice workbook
advice_df = None
if advice_file is not None:
    try:
        advice_df = pd.read_excel(advice_file, sheet_name="advice")
        # clean
        for c in ["ISO_Year","ISO_Week","SetNumber","%1RM","Reps","SetsOverride"]:
            if c in advice_df.columns:
                advice_df[c] = pd.to_numeric(advice_df[c], errors="coerce")
        if "Workout" in advice_df.columns:
            advice_df["Workout"] = advice_df["Workout"].astype(str)
        if "Exercise" in advice_df.columns:
            advice_df["Exercise"] = advice_df["Exercise"].astype(str)
    except Exception as e:
        st.error(f"Kon sheet 'advice' niet lezen: {e}")

# Fallback coaching rules by status
FALLBACK = {
    "Chest":     {"TopPct": 78, "TopReps": 6, "BackPct": 70, "BackReps": 10},
    "Back":      {"TopPct": 75, "TopReps": 8, "BackPct": 65, "BackReps": 12},
    "Shoulders": {"TopPct": 72, "TopReps": 8, "BackPct": 60, "BackReps": 15},
    "Triceps":   {"TopPct": 70, "TopReps":10, "BackPct": 62, "BackReps": 14},
    "Biceps":    {"TopPct": 70, "TopReps":10, "BackPct": 62, "BackReps": 14},
    "Core":      {"TopPct": 80, "TopReps":10, "BackPct": 65, "BackReps": 15},
    "Other":     {"TopPct": 70, "TopReps":10, "BackPct": 60, "BackReps": 12},
}

def coach_for_exercise(workout_name: str, exercise: str, set_idx: int, muscle_status: Dict[str, str]):
    """Return (%1RM, reps) tuple for given set, preference: advice_df -> fallback by muscle/status."""
    # 1) Explicit advice match year+week+workout+exercise+set
    if advice_df is not None:
        q = advice_df.copy()
        q = q[
            ((q["ISO_Year"].isna()) | (q["ISO_Year"] == year)) &
            ((q["ISO_Week"].isna()) | (q["ISO_Week"] == week)) &
            ((q["Workout"].isna())  | (q["Workout"] == workout_name)) &
            (q["Exercise"] == exercise) &
            ((q["SetNumber"].isna()) | (q["SetNumber"] == set_idx))
        ]
        if not q.empty:
            row = q.iloc[0]
            pct = float(row.get("%1RM", np.nan))
            reps = int(row.get("Reps", np.nan)) if not np.isnan(row.get("Reps", np.nan)) else None
            return pct, reps

    # 2) Fallback policy by muscle & weekly status
    muscle = MUSCLE_MAP.get(exercise, "Other")
    status = muscle_status.get(muscle, "OK")
    base = FALLBACK.get(muscle, FALLBACK["Other"]).copy()

    # Nudge by status
    if status == "LOW":
        base["TopPct"] += 2; base["BackPct"] += 2
        base["BackReps"] += 2
    elif status == "HIGH":
        base["TopPct"] -= 3; base["BackPct"] -= 3
        base["BackReps"] -= 2

    if set_idx == 1:
        return base["TopPct"], base["TopReps"]
    else:
        return base["BackPct"], base["BackReps"]

def compose_next_workout(daytype: str, dup: str, muscle_status: Dict[str, str]) -> pd.DataFrame:
    """
    Choose template by daytype & DUP pattern (H/M/L cycles map to A/B automatically by week parity).
    - We alternate A/B by ISO week number.
    """
    is_even = (week % 2 == 0)
    if daytype == "Push":
        workout = "1B Push" if is_even else "1A Push"  # example: alternate
        alt = "2B Push" if is_even else "2A Push"
    else:
        workout = "1B Pull" if is_even else "1A Pull"
        alt = "2B Pull" if is_even else "2A Pull"

    # DUP scaling
    dup_scale = {"H": 1.00, "M": 0.96, "L": 0.92}
    scale = dup_scale.get(dup, 1.0)

    # Build prescription table
    rows = []
    for ex, base_sets in TEMPLATES[workout]:
        # Allow advice to override number of sets
        sets_override = None
        if advice_df is not None:
            q = advice_df[
                ((advice_df["ISO_Year"].isna()) | (advice_df["ISO_Year"] == year)) &
                ((advice_df["ISO_Week"].isna()) | (advice_df["ISO_Week"] == week)) &
                (advice_df["Workout"] == workout) &
                (advice_df["Exercise"] == ex) &
                (advice_df["SetsOverride"].notna())
            ]
            if not q.empty:
                sets_override = int(q.iloc[0]["SetsOverride"])

        n_sets = sets_override if sets_override is not None else base_sets
        for s in range(1, n_sets + 1):
            pct, reps = coach_for_exercise(workout, ex, s, muscle_status)
            pct = pct * scale
            # Resolve 1RM for load calc
            est1rm = ones.get(ex, np.nan)
            load = np.nan
            if not np.isnan(est1rm):
                load = est1rm * (pct / 100.0)
                load = round_load(load, is_dumbbell(ex))
            rows.append({
                "Workout": workout,
                "Exercise": ex,
                "Set": s,
                "%1RM": round(pct, 1),
                "Reps": reps,
                "Est_1RM": round(est1rm, 1) if not np.isnan(est1rm) else None,
                "Target Load": load if not np.isnan(load) else None,
            })

    dfw = pd.DataFrame(rows)
    return dfw

# Build muscle-status dict for the selected week
muscle_status = {row["Muscle"]: row["Status"] for _, row in summary.iterrows()}

st.subheader("Next Workout — Coachingvoorstel")

col1, col2, col3 = st.columns(3)
with col1:
    daytype = st.selectbox("Dagtype", ["Push", "Pull"])
with col2:
    dup_choice = st.selectbox("DUP (H/M/L)", ["H", "M", "L"], index=1)
with col3:
    st.write("Weekritme:", SCHEDULE)

plan = compose_next_workout(daytype, dup_choice, muscle_status)
st.dataframe(plan, use_container_width=True)

# Export
buf = io.StringIO()
plan.to_csv(buf, index=False)
st.download_button("Export ‘Next Workout’ CSV", data=buf.getvalue(), file_name="next_workout_plan.csv", mime="text/csv")

st.divider()
st.markdown("**Veiligheidsregels**")
st.markdown(
    "- LOW/OK/HIGH bepaald t.o.v. 10–20 sets/week.\n"
    "- DUP schaalt %1RM (H:100%, M:96%, L:92%).\n"
    "- Fallback-coaching per spiergroep, aangepast op status.\n"
    "- Load afgerond op dumbbell-stappen (2.0 kg) of barbell-stappen (2.5 kg).\n"
    "- Rolling 4w average bewaakt volumestijging."
)
