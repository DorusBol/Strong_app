import datetime as dt
import math
import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Basisconfig
# =========================
st.set_page_config(page_title="Periodization Coach — MVP", layout="wide")
st.title("Periodization Coach — MVP")

st.caption(
    "Upload een **Strong App** CSV-export. De app berekent weekvolume per spiergroep, "
    "markeert LOW/OK/HIGH t.o.v. 10–20 sets/week en geeft een ‘Next Workout’ voorstel."
)

# =========================
# Parameters
# =========================
LOWER_BOUND = 10  # sets/week
UPPER_BOUND = 20

# Map jouw oefeningnamen (zoals in Strong) naar spiergroep + bewegingstype
# ➜ Breid dit gaandeweg uit met jouw exacte namen.
MUSCLE_MAP = {
    "Incline Bench Press (Dumbbell)": ("chest", "press"),
    "Bench Press (Dumbbell)": ("chest", "press"),
    "Bench Press (Cable)": ("chest", "press"),
    "Chest Fly": ("chest", "fly"),

    "Overhead Press (Dumbbell)": ("shoulders_front", "press"),
    "Seated Overhead Press (Dumbbell)": ("shoulders_front", "press"),
    "Arnold Press (Dumbbell)": ("shoulders_front", "press"),
    "Lateral Raise (Dumbbell)": ("shoulders_side", "raise"),

    "Reverse Fly (Cable)": ("shoulders_rear", "fly"),
    "Face Pull (Cable)": ("shoulders_rear", "retraction"),

    "Lat Pulldown (Cable)": ("back", "pulldown"),
    "Seated Row (Cable)": ("back", "row"),
    "Incline Row (Dumbbell)": ("back", "row"),

    "Tricep Pushdown (Cable)": ("triceps", "extension"),
    "Skullcrusher (Dumbbell)": ("triceps", "extension"),
    "Cable Kickback": ("triceps", "extension"),

    "Bicep Curl (Dumbbell)": ("biceps", "curl"),
    "LOW CABLE BICEP CURL (Cable)": ("biceps", "curl"),
    "EZ BAR CURL": ("biceps", "curl"),

    "Cable Crunch": ("core", "flexion"),
    "Hanging Knee Raise": ("core", "flexion"),
}

# Rep-target ➜ intensiteitsband (simpele DUP mapping)
def dup_band(rep_target: int):
    if rep_target <= 6:
        return 0.82, "Heavy"
    if rep_target <= 10:
        return 0.72, "Moderate"
    return 0.65, "Light"

# Epley 1RM schatter
def epley_1rm(weight, reps):
    try:
        w = float(weight); r = float(reps)
        if r <= 1:
            return w
        return w * (1 + r/30.0)
    except Exception:
        return np.nan

def status_flag(sets: int):
    if sets < LOWER_BOUND:
        return "LOW"
    if sets > UPPER_BOUND:
        return "HIGH"
    return "OK"

# =========================
# Upload
# =========================
with st.sidebar:
    st.header("Upload")
    f = st.file_uploader("Upload Strong CSV", type=["csv"])
    st.markdown(
        "- Minimale kolommen: **Date**, **Exercise Name**, **Weight**, **Reps**.\n"
        "- Elke rij wordt als 1 werkset geïnterpreteerd."
    )

if not f:
    st.info("Upload een Strong CSV om te starten.")
    st.stop()

# =========================
# Data inlezen en normaliseren
# =========================
try:
    df = pd.read_csv(f)
except Exception as e:
    st.error(f"Kon CSV niet lezen: {e}")
    st.stop()

df.columns = [c.strip() for c in df.columns]
expected = {"Date", "Exercise Name", "Weight", "Reps"}
missing = expected - set(df.columns)
if missing:
    st.error(f"Ontbrekende kolommen in CSV: {missing}")
    st.stop()

# parse datums en basisvelden
df["Date"] = pd.to_datetime(df["Date"])
df["Sets"] = 1  # elke rij = 1 set (Strong export)
df["Exercise Name"] = df["Exercise Name"].astype(str)

# week & jaar voor groeperen
df["ISO_Year"] = df["Date"].dt.isocalendar().year.astype(int)
df["ISO_Week"] = df["Date"].dt.isocalendar().week.astype(int)

# spiergroep/patroon mapping
df["muscle_group"] = df["Exercise Name"].map(lambda x: MUSCLE_MAP.get(x, (None, None))[0])
df["pattern"] = df["Exercise Name"].map(lambda x: MUSCLE_MAP.get(x, (None, None))[1])

work = df[~df["muscle_group"].isna()].copy()

# =========================
# Weekselectie
# =========================
today = dt.date.today()
iy, iw, _ = today.isocalendar()

colA, colB = st.columns(2)
with colA:
    year = st.selectbox("Jaar", sorted(work["ISO_Year"].unique()), index=max(0, len(sorted(work["ISO_Year"].unique()))-1))
with colB:
    weeks_for_year = sorted(work.loc[work["ISO_Year"] == year, "ISO_Week"].unique())
    default_week_idx = weeks_for_year.index(iw) if iw in weeks_for_year else len(weeks_for_year)-1
    week = st.selectbox("Week (ISO)", weeks_for_year, index=max(0, default_week_idx))

wkdata = work[(work["ISO_Year"] == year) & (work["ISO_Week"] == week)].copy()

# =========================
# Volume per spiergroep
# =========================
st.subheader("Weekvolume per spiergroep")

if wkdata.empty:
    st.warning("Geen data voor de gekozen week.")
else:
    vol = wkdata.groupby("muscle_group", as_index=False)["Sets"].sum()
    vol["Status"] = vol["Sets"].apply(status_flag)
    vol = vol.sort_values(["Status", "Sets"], ascending=[True, False])  # LOW/HIGH vallen op
    st.dataframe(vol, use_container_width=True)

    # advies op basis van status
    st.subheader("Volume-advies (t.o.v. 10–20 sets/week)")
    advice_rows = []
    for _, r in vol.iterrows():
        mg, sets, stat = r["muscle_group"], int(r["Sets"]), r["Status"]
        if stat == "LOW":
            advice = "Voeg +1 set toe aan 1–2 oefeningen in deze groep (start met compounds)."
        elif stat == "HIGH":
            advice = "Verlaag −1 set op 1–2 oefeningen (prioriteer isolaties of overlap)."
        else:
            advice = "Behoud volume."
        advice_rows.append((mg, sets, stat, advice))
    st.dataframe(pd.DataFrame(advice_rows, columns=["Spiergroep", "Sets", "Status", "Advies"]), use_container_width=True)

# =========================
# Next Workout — voorstel
# =========================
st.subheader("Next Workout — voorstel")

col1, col2, col3 = st.columns(3)
with col1:
    day_type = st.selectbox("Dagtype", ["Push", "Pull"])
with col2:
    periodization = st.selectbox("Periodisering", ["DUP (H/M/L)", "Linear"])
with col3:
    rep_default = st.number_input("Default rep-target (fallback)", min_value=6, max_value=20, value=8, step=1)

# laatste prestatie per oefening (uit hele dataset, recentste rij per oefening)
last = work.sort_values("Date").groupby("Exercise Name").tail(1)

# kies oefeningen voor dagtype
if day_type == "Push":
    target_groups = {"chest", "shoulders_front", "shoulders_side", "triceps"}
else:
    target_groups = {"back", "shoulders_rear", "biceps", "core"}

candidate_exercises = [e for e, (mg, _) in MUSCLE_MAP.items() if mg in target_groups]

plan_rows = []
for ex in candidate_exercises:
    mg, pattern = MUSCLE_MAP.get(ex, (None, None))
    if mg is None:
        continue

    # rep-target op basis van simpele heuristiek
    if periodization == "DUP (H/M/L)":
        if mg in {"chest", "back", "shoulders_front"}:
            rep_target = 8
        elif mg in {"shoulders_side", "shoulders_rear", "biceps", "triceps", "core"}:
            rep_target = 12
        else:
            rep_target = rep_default
    else:
        rep_target = rep_default

    pct_1rm, inten_name = dup_band(rep_target)

    row = last[last["Exercise Name"] == ex]
    if not row.empty:
        lw = float(row.iloc[0].get("Weight", np.nan))
        lr = float(row.iloc[0].get("Reps", np.nan))
        est1rm = epley_1rm(lw, lr) if (lw > 0 and lr > 0) else np.nan
    else:
        lw = lr = est1rm = np.nan

    # doelgewicht
    if not math.isnan(est1rm):
        target_w = round((est1rm * pct_1rm) / 2.5) * 2.5  # afronden per 2.5 kg
    elif not math.isnan(lw):
        target_w = lw  # behoud laatste gewicht
    else:
        target_w = np.nan

    # set-advies (afgeleid van weekvolume-status)
    status_lookup = {}
    if wkdata.shape[0] > 0:
        vdict = wkdata.groupby("muscle_group")["Sets"].count().to_dict()  # presence check
        if "muscle_group" in vol.columns:
            status_lookup = dict(zip(vol["muscle_group"], vol["Status"]))
    st_flag = status_lookup.get(mg, "OK")
    sets = 3 if st_flag == "LOW" else 2  # simpel: verhoog naar 3 sets als LOW

    plan_rows.append([ex, mg, inten_name, rep_target, sets, target_w])

plan = pd.DataFrame(plan_rows, columns=["Exercise", "Muscle", "Intensity", "Rep target", "Sets", "Target weight (kg)"])
if plan.empty:
    st.info("Geen voorstel — breid je MUSCLE_MAP uit met jouw oefeningnamen uit Strong.")
else:
    st.dataframe(plan, use_container_width=True)

st.success(
    "Tip: beheer alleen **gewicht + reps** in Strong. Laat deze coach **sets** en **rep-targets** sturen via LOW/OK/HIGH."
)
