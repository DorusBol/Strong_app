# app.py
# Periodization Coach — kaart-/blokweergave per set
# Vereist: streamlit, pandas, numpy. Optioneel: openpyxl (voor .xlsx inlezen)

import io
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, Iterable, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# ---------- UI ----------
# =========================

st.set_page_config(page_title="Periodization Coach — Kaartenweergave", layout="wide")

st.title("Next Workout — Coachingvoorstel (Kaartenweergave)")
st.caption(
    "Upload je Strong-export (CSV). Optioneel: Advies-Excel (tab **advice**) met `%1RM`, `Reps`, `SetsOverride`. "
    "Resultaat toont per oefening grote blokken per set met %1RM, reps en doel-kg."
)

col_u1, col_u2 = st.columns(2)
with col_u1:
    strong_file = st.file_uploader("Strong CSV", type=["csv"])
with col_u2:
    advice_xlsx = st.file_uploader("Advies-Excel (tab: advice)", type=["xlsx"])

with st.expander("Weergave & berekening — instellingen", expanded=True):
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        kg_step = st.number_input("Rondingsstap (kg)", min_value=0.25, max_value=5.0, step=0.25, value=2.5)
    with col_s2:
        round_mode = st.selectbox("Rondingsmodus", ["nearest", "down", "up"], index=0)
    with col_s3:
        safety_delta = st.number_input("Veiligheidsmarge op %1RM (-=zwaarder, +=lichter)", value=0.0, step=1.0, format="%.1f")
    with col_s4:
        lookback_weeks = st.number_input("Max. lookback (weken) voor e1RM", min_value=1, max_value=52, value=8, step=1)

    col_s5, col_s6, col_s7 = st.columns(3)
    with col_s5:
        default_workout_name = st.text_input("Dagtype/Workout", value="Push")
    with col_s6:
        show_warmups = st.checkbox("Toon automatische warm-up sets", value=False)
    with col_s7:
        bar_weight = st.number_input("Stanggewicht (kg) — warm-up hint", min_value=0.0, max_value=30.0, value=20.0, step=0.5)

# ==============================
# ---------- Helpers ----------
# ==============================

def _norm(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    return " ".join(s.split())

def _safe_int(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == ""):
            return None
        return int(float(x))
    except Exception:
        return None

def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        return None

def _round_weight(kg: float, step: float, mode: str) -> float:
    if step <= 0:
        return round(kg, 1)
    q = kg / step
    if mode == "up":
        return round(math.ceil(q) * step, 3)
    if mode == "down":
        return round(math.floor(q) * step, 3)
    return round(round(q) * step, 3)  # nearest

def _epley_1rm(weight: float, reps: int) -> float:
    # Epley: 1RM = w*(1 + r/30)
    return weight * (1.0 + reps / 30.0)

def _isocalendar_tuple(ts: pd.Timestamp) -> Tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)

def _read_strong_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Minimale vereisten
    req = {"Date", "Exercise Name", "Weight", "Reps"}
    missing = req - set(df.columns)
    if missing:
        st.error(f"Strong CSV mist kolommen: {sorted(missing)}")
        st.stop()
    # Parse datum
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Exercise"] = df["Exercise Name"].astype(str)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Reps"] = pd.to_numeric(df["Reps"], errors="coerce")
    df["ISO_Year"], df["ISO_Week"] = zip(*df["Date"].map(_isocalendar_tuple))
    return df

def _read_advice_xlsx(file) -> pd.DataFrame:
    try:
        df = pd.read_excel(io.BytesIO(file.read()), sheet_name="advice", engine="openpyxl")
    except ModuleNotFoundError:
        st.error("Openen van .xlsx vereist **openpyxl**. Installeer met `pip install openpyxl`.")
        st.stop()
    except ValueError:
        st.error("Sheet **'advice'** niet gevonden. Hernoem tabblad naar exact `advice`.")
        st.stop()
    # Zorg voor aanwezigheid kolommen
    for c in ["ISO_Year", "ISO_Week", "Workout", "Exercise", "SetNumber", "%1RM", "Reps", "SetsOverride"]:
        if c not in df.columns:
            df[c] = np.nan
    # Normalisatie
    df["Workout_norm"] = df["Workout"].map(_norm)
    df["Exercise_norm"] = df["Exercise"].map(_norm)
    df["SetNumber_int"] = df["SetNumber"].map(_safe_int)
    df["ISO_Year_int"] = df["ISO_Year"].map(_safe_int)
    df["ISO_Week_int"] = df["ISO_Week"].map(_safe_int)
    df["spec"] = (
        df["ISO_Year_int"].notna().astype(int)
        + df["ISO_Week_int"].notna().astype(int)
        + (df["Workout_norm"] != "").astype(int)
        + (df["Exercise_norm"] != "").astype(int)
        + df["SetNumber_int"].notna().astype(int)
    )
    df["recency"] = list(zip(df["ISO_Year_int"].fillna(-1e9), df["ISO_Week_int"].fillna(-1e9)))
    return df

def _best_match(adf: pd.DataFrame,
                iso_year: Optional[int],
                iso_week: Optional[int],
                workout: str,
                exercise: str,
                setno: Optional[int]) -> Optional[pd.Series]:
    if adf is None or adf.empty:
        return None
    w = _norm(workout)
    e = _norm(exercise)
    s = setno
    picks: List[pd.Series] = []

    def pick(mask):
        sub = adf[mask]
        if not sub.empty:
            sub = sub.sort_values(by=["spec", "recency"], ascending=[False, False], kind="mergesort")
            picks.append(sub.iloc[0])

    # 1) Y+W+workout+exercise+set
    pick((adf["ISO_Year_int"] == iso_year) & (adf["ISO_Week_int"] == iso_week) &
         (adf["Workout_norm"] == w) & (adf["Exercise_norm"] == e) & (adf["SetNumber_int"] == s))
    # 2) Y+W+workout+exercise
    pick((adf["ISO_Year_int"] == iso_year) & (adf["ISO_Week_int"] == iso_week) &
         (adf["Workout_norm"] == w) & (adf["Exercise_norm"] == e) & adf["SetNumber_int"].isna())
    # 3) Y+W+exercise
    pick((adf["ISO_Year_int"] == iso_year) & (adf["ISO_Week_int"] == iso_week) &
         (adf["Exercise_norm"] == e) & (adf["Exercise_norm"] == e))
    # 4) workout+exercise+set (jaar/week leeg)
    pick(adf["ISO_Year_int"].isna() & adf["ISO_Week_int"].isna() &
         (adf["Workout_norm"] == w) & (adf["Exercise_norm"] == e) & (adf["SetNumber_int"] == s))
    # 5) workout+exercise
    pick(adf["ISO_Year_int"].isna() & adf["ISO_Week_int"].isna() &
         (adf["Workout_norm"] == w) & (adf["Exercise_norm"] == e) & adf["SetNumber_int"].isna())
    # 6) exercise+set
    pick(adf["ISO_Year_int"].isna() & adf["ISO_Week_int"].isna() &
         (adf["Workout_norm"] == "") & (adf["Exercise_norm"] == e) & (adf["SetNumber_int"] == s))
    # 7) exercise
    pick(adf["ISO_Year_int"].isna() & adf["ISO_Week_int"].isna() &
         (adf["Workout_norm"] == "") & (adf["Exercise_norm"] == e) & adf["SetNumber_int"].isna())

    return picks[0] if picks else None

def _compute_e1rm(df: pd.DataFrame, lookback_weeks: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Exercise", "e1RM"])
    last_date = df["Date"].max()
    cutoff = last_date - pd.Timedelta(weeks=int(lookback_weeks))
    sub = df[df["Date"] >= cutoff].copy()

    sub["est_1rm"] = _epley_1rm(sub["Weight"].astype(float), sub["Reps"].astype(int))
    agg = sub.groupby("Exercise", as_index=False)["est_1rm"].max().rename(columns={"est_1rm": "e1RM"})
    return agg

def _apply_fallback(row: pd.Series, fallback_tbl: pd.DataFrame) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    if fallback_tbl is None or fallback_tbl.empty:
        return (None, None, None)
    mg = _norm(row.get("MuscleGroup"))
    ls = _norm(row.get("LoadStatus"))
    hit = fallback_tbl[(fallback_tbl["MuscleGroup_norm"] == mg) & (fallback_tbl["LoadStatus_norm"] == ls)]
    if hit.empty:
        return (None, None, None)
    r = hit.iloc[0]
    return (_safe_float(r.get("%1RM")), _safe_int(r.get("Reps")), _safe_int(r.get("Sets")))

def _prep_fallback_matrix(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    # Verwacht kolommen: MuscleGroup, LoadStatus, %1RM, Reps, Sets
    if raw is None:
        return pd.DataFrame(columns=["MuscleGroup","LoadStatus","%1RM","Reps","Sets","MuscleGroup_norm","LoadStatus_norm"])
    df = raw.copy()
    for c in ["MuscleGroup", "LoadStatus"]:
        if c not in df.columns: df[c] = ""
    df["MuscleGroup_norm"] = df["MuscleGroup"].map(_norm)
    df["LoadStatus_norm"] = df["LoadStatus"].map(_norm)
    for c in ["%1RM", "Reps", "Sets"]:
        if c not in df.columns: df[c] = np.nan
    return df

def _calc_target_weight(e1rm: Optional[float], pct: Optional[float], step: float, mode: str) -> Optional[float]:
    if e1rm is None or pct is None:
        return None
    raw = e1rm * (pct / 100.0)
    return _round_weight(raw, step, mode)

# ==============================
# ---------- Pipeline ----------
# ==============================

if strong_file is None:
    st.info("Upload eerst je **Strong CSV** om een voorstel te genereren.")
    st.stop()

df_strong = _read_strong_csv(strong_file)
e1rm_tbl = _compute_e1rm(df_strong, lookback_weeks)

advice_df = None
if advice_xlsx is not None:
    advice_df = _read_advice_xlsx(advice_xlsx)

# Voorbeeld: je bestaande template voor de "volgende sessie".
# In veel implementaties wordt dit al elders afgeleid. Hier construeren we een minimale template
# uit de meest recente training per oefening: 3 sets default, reps = laatst uitgevoerde reps.
last_by_ex = df_strong.sort_values("Date").groupby("Exercise", as_index=False).tail(1)
template_rows = []
for r in last_by_ex.itertuples(index=False):
    for s in range(1, 4):  # default 3 sets
        template_rows.append({
            "Workout": default_workout_name,
            "Exercise": getattr(r, "Exercise"),
            "SetNumber": s,
            # optioneel gevuld in jouw data:
            "MuscleGroup": "",          # kan door jou worden gevuld
            "LoadStatus": "ok",         # "low"/"ok"/"high" — placeholder
            "Sets": 3                   # default sets per oefening (kan overridden worden)
        })
template = pd.DataFrame(template_rows)

# Merge e1RM
template = template.merge(e1rm_tbl, on="Exercise", how="left")

# Advies-matching / fallback
fallback_matrix = _prep_fallback_matrix(None)  # <- vul met eigen matrix indien gewenst

out_rows = []
today = datetime.now()
isoY, isoW = today.isocalendar().year, today.isocalendar().week

for r in template.itertuples(index=False):
    ex = getattr(r, "Exercise")
    setno = getattr(r, "SetNumber")
    workout = getattr(r, "Workout")

    match = None
    if advice_df is not None:
        match = _best_match(advice_df, isoY, isoW, workout, ex, setno)

    pct = _safe_float(match["%1RM"]) if match is not None else None
    reps = _safe_int(match["Reps"]) if match is not None else None
    sets_override = _safe_int(match["SetsOverride"]) if match is not None else None

    # Fallback indien geen match
    if pct is None or reps is None:
        fb_pct, fb_reps, fb_sets = _apply_fallback(pd.Series(r._asdict()), fallback_matrix)
        pct = fb_pct if pct is None else pct
        reps = fb_reps if reps is None else reps
        # neem fb_sets als default voor Sets (alleen informatief)
        if fb_sets is not None and "Sets" in template.columns:
            pass

    # Veiligheidsmarge op %1RM
    if pct is not None and safety_delta != 0.0:
        pct = max(0.0, pct + float(safety_delta))

    # Doel-kg
    e1 = _safe_float(getattr(r, "e1RM"))
    tgt_kg = _calc_target_weight(e1, pct, kg_step, round_mode) if (e1 is not None and pct is not None) else None

    out_rows.append({
        "Workout": workout,
        "Exercise": ex,
        "SetNumber": setno,
        "%1RM": pct,
        "Reps": reps,
        "TargetKg": tgt_kg,
        "e1RM": e1,
        "SetsOverride": sets_override
    })

plan = pd.DataFrame(out_rows)

# Indien SetsOverride aanwezig is, toon precies zoveel set-blokken als override
def _apply_sets_override_to_display(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ex, grp in df.groupby("Exercise", sort=False):
        max_set = grp["SetNumber"].max()
        override_vals = grp["SetsOverride"].dropna()
        if not override_vals.empty and override_vals.iloc[0] and override_vals.iloc[0] > 0:
            n_sets = int(override_vals.iloc[0])
        else:
            n_sets = int(max_set)
        # Neem de eerste n_sets sets; indien minder rijen dan n_sets, vul aan door laatste set te kopiëren
        g = grp.sort_values("SetNumber").copy()
        if len(g) >= n_sets:
            g = g.head(n_sets)
        else:
            if len(g) > 0:
                last = g.iloc[-1].to_dict()
                for s in range(len(g)+1, n_sets+1):
                    last["SetNumber"] = s
                    g = pd.concat([g, pd.DataFrame([last])], ignore_index=True)
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else df

plan_disp = _apply_sets_override_to_display(plan)

# ==============================
# ---------- Styling ----------
# ==============================

CUSTOM_CSS = """
<style>
.section-title {
    margin-top: 0.5rem;
    margin-bottom: 0.25rem;
    font-size: 1.25rem;
    font-weight: 700;
}
.exercise-card {
    padding: 0.75rem 1rem;
    margin: 0.25rem 0 0.75rem 0;
    border-radius: 12px;
    border: 1px solid rgba(120,120,120,0.25);
    background: linear-gradient(180deg, rgba(250,250,250,0.9), rgba(245,245,245,0.9));
}
.set-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
}
.set-card {
    border: 1px solid rgba(100,100,100,0.25);
    border-radius: 12px;
    padding: 14px;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.set-header {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 6px;
}
.kv {
    display: grid;
    grid-template-columns: 1fr auto;
    row-gap: 6px;
    column-gap: 12px;
    font-size: 0.95rem;
}
.kv .k { color: #555; }
.kv .v { font-weight: 700; }
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #124;
    background: #e9f3ff;
    border: 1px solid #cfe4ff;
}
.hint {
    margin-top: 8px;
    font-size: 0.8rem;
    color: #666;
}
.divider {
    height: 1px;
    background: rgba(0,0,0,0.07);
    margin: 8px 0 12px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================
# ---------- Render ----------
# ==============================

# Groepeer per oefening binnen workout
if plan_disp.empty:
    st.warning("Geen adviesregels beschikbaar.")
    st.stop()

# Bovenaan: context
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Unieke oefeningen", plan_disp["Exercise"].nunique())
with c2:
    st.metric("Totaal set-blokken", len(plan_disp))
with c3:
    st.metric("Lookback (weken)", lookback_weeks)
with c4:
    st.metric("Ronding (kg)", kg_step)

# Render per oefening
for ex, grp in plan_disp.groupby("Exercise", sort=False):
    grp = grp.sort_values("SetNumber")
    e1 = grp["e1RM"].dropna().max()
    header = f"{ex}  "
    sub = f"(e1RM: {e1:.1f} kg)" if e1 is not None and not np.isnan(e1) else "(e1RM: onbekend)"
    st.markdown(f"<div class='section-title'>{header}<span class='badge'>{sub}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='exercise-card'>", unsafe_allow_html=True)

    # Grid van set-kaarten
    st.markdown("<div class='set-grid'>", unsafe_allow_html=True)
    for r in grp.itertuples(index=False):
        pct = getattr(r, "_1RM") if hasattr(r, "_1RM") else getattr(r, "%1RM")
        reps = getattr(r, "Reps")
        tgt = getattr(r, "TargetKg")
        setn = getattr(r, "SetNumber")

        # Kaart
        st.markdown("<div class='set-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='set-header'>Set {int(setn)}</div>", unsafe_allow_html=True)
        # key-values
        st.markdown("<div class='kv'>", unsafe_allow_html=True)
        st.markdown("<div class='k'>%1RM</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='v'>{'' if pct is None else f'{pct:.0f}%'}</div>", unsafe_allow_html=True)
        st.markdown("<div class='k'>Reps</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='v'>{'' if reps is None else int(reps)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='k'>Doel-kg</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='v'>{'' if tgt is None or np.isnan(tgt) else f'{tgt:.1f} kg'}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)  # kv

        # Optionele hints/warm-ups
        if show_warmups and (tgt is not None) and (tgt > 0):
            # simpele ladder naar ~85%, 70%, 50% van doel
            w3 = _round_weight(tgt * 0.85, kg_step, round_mode)
            w2 = _round_weight(tgt * 0.70, kg_step, round_mode)
            w1 = _round_weight(max(bar_weight, tgt * 0.50), kg_step, round_mode)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='hint'><b>Warm-up</b>: "
                        f"{w1:.1f} × 5  →  {w2:.1f} × 3  →  {w3:.1f} × 1–2</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # set-card
    st.markdown("</div>", unsafe_allow_html=True)  # set-grid
    st.markdown("</div>", unsafe_allow_html=True)  # exercise-card

# ==============================
# ---------- Exports ----------
# ==============================

with st.expander("Exporteren"):
    # Exporteer het plan in 'platte' vorm als CSV voor logging/controle
    exp = plan_disp.copy()
    exp = exp[["Workout","Exercise","SetNumber","%1RM","Reps","TargetKg","e1RM","SetsOverride"]]
    csv_bytes = exp.to_csv(index=False).encode("utf-8")
    st.download_button("Download voorstel als CSV", data=csv_bytes, file_name="coaching_voorstel.csv", mime="text/csv")

    st.markdown(
        "- CSV is bedoeld als audit trail. De **kaart-weergave** hierboven is leidend voor de uitvoering.\n"
        "- Pas instellingen aan (ronding, lookback, veiligheidsmarge) om de blokken realtime te herberekenen."
    )
