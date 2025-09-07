# app.py
# Periodization Coach — kaartenweergave met Advies-Excel + Workouts-planner (Excel of in-app)
# Vereist: streamlit, pandas, numpy. Optioneel: openpyxl (voor .xlsx inlezen)

import io
import math
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# ---------- UI ----------
# =========================

st.set_page_config(page_title="Periodization Coach — Kaartenweergave", layout="wide")

st.title("Next Workout — Coachingvoorstel (Kaartenweergave)")
st.caption(
    "Upload je Strong-export (CSV). Optioneel: Advies-Excel (tab **advice**) met `%1RM`, `Reps`, `SetsOverride` "
    "en/of een Workouts-planner (tab **workouts**). Je kunt ook een **in-app planner** gebruiken."
)

col_u1, col_u2 = st.columns(2)
with col_u1:
    strong_file = st.file_uploader("Strong CSV", type=["csv"])
with col_u2:
    advice_xlsx = st.file_uploader("Advies-Excel (advice/workouts)", type=["xlsx"])

with st.expander("Weergave & berekening — instellingen", expanded=True):
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        kg_step = st.number_input("Rondingsstap (kg)", min_value=0.25, max_value=5.0, step=0.25, value=1.0)
    with col_s2:
        round_mode = st.selectbox("Rondingsmodus", ["nearest", "down", "up"], index=0)
    with col_s3:
        safety_delta = st.number_input("Veiligheidsmarge op %1RM (-=zwaarder, +=lichter)", value=0.0, step=1.0, format="%.1f")
    with col_s4:
        lookback_weeks = st.number_input("Max. lookback (weken) voor e1RM", min_value=1, max_value=52, value=8, step=1)

    col_s5, col_s6, col_s7 = st.columns(3)
    with col_s5:
        default_workout_name = st.text_input("Dagtype/Workout (default indien geen planner)", value="Push")
    with col_s6:
        show_warmups = st.checkbox("Toon automatische warm-up sets", value=False)
    with col_s7:
        bar_weight = st.number_input("Stanggewicht (kg) — warm-up hint", min_value=0.0, max_value=30.0, value=20.0, step=0.5)

    col_s8, col_s9 = st.columns(2)
    with col_s8:
        use_global_defaults = st.checkbox("Gebruik defaults als geen advies/fallback", value=True,
                                          help="Zet %1RM=70 en Reps=8 als er geen match of fallback is.")
    with col_s9:
        show_debug = st.checkbox("Toon debug-diagnostiek", value=False)

with st.expander("Workouts plannen (in-app) — optioneel", expanded=False):
    use_inapp = st.checkbox("Gebruik handmatige planner", value=False)
    planner_df = pd.DataFrame()
    if use_inapp:
        st.markdown(
            "Voeg regels toe: **Workout** (tekst), **Exercise** (selecteer), **Sets** (aantal). "
            "Je selectie krijgt voorrang op Excel en default."
        )

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

def _coerce_pct(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s == "":
        return None
    s = s.replace("%", "").strip()
    try:
        return float(s)
    except Exception:
        return None

def _fmt_pct(x: Optional[float]) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0f}%"

def _fmt_int(x: Optional[int]) -> str:
    return "" if x is None else f"{int(x)}"

def _fmt_kg(x: Optional[float]) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f} kg"

def _round_weight(kg: float, step: float, mode: str) -> float:
    if step <= 0:
        return round(kg, 1)
    q = kg / step
    if mode == "up":
        return round(math.ceil(q) * step, 3)
    if mode == "down":
        return round(math.floor(q) * step, 3)
    return round(round(q) * step, 3)

def _epley_1rm(weight: float, reps: int) -> float:
    return weight * (1.0 + reps / 30.0)

def _isocalendar_tuple(ts: pd.Timestamp) -> Tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)

def _read_strong_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    req = {"Date", "Exercise Name", "Weight", "Reps"}
    missing = req - set(df.columns)
    if missing:
        st.error(f"Strong CSV mist kolommen: {sorted(missing)}")
        st.stop()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Exercise"] = df["Exercise Name"].astype(str)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Reps"] = pd.to_numeric(df["Reps"], errors="coerce")
    df["ISO_Year"], df["ISO_Week"] = zip(*df["Date"].map(_isocalendar_tuple))
    return df

def _read_sheet(upload, sheet: str) -> Optional[pd.DataFrame]:
    try:
        b = upload.read()
        xls = io.BytesIO(b)
        df = pd.read_excel(xls, sheet_name=sheet, engine="openpyxl")
        return df
    except ValueError:
        return None

def _read_advice_xlsx(upload) -> Optional[pd.DataFrame]:
    df = _read_sheet(upload, "advice")
    if df is None:
        return None
    for c in ["ISO_Year", "ISO_Week", "Workout", "Exercise", "SetNumber", "%1RM", "Reps", "SetsOverride"]:
        if c not in df.columns:
            df[c] = np.nan
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
    df["%1RM"] = df["%1RM"].map(_coerce_pct)
    df["Reps"] = df["Reps"].map(_safe_int)
    df["SetsOverride"] = df["SetsOverride"].map(_safe_int)
    return df

def _read_workouts_xlsx(upload) -> Optional[pd.DataFrame]:
    df = _read_sheet(upload, "workouts")
    if df is None:
        return None
    for c in ["Workout", "Exercise", "Sets", "ISO_Year", "ISO_Week"]:
        if c not in df.columns:
            df[c] = np.nan
    df["Workout"] = df["Workout"].fillna("").astype(str)
    df["Exercise"] = df["Exercise"].fillna("").astype(str)
    df["Sets"] = df["Sets"].map(_safe_int).fillna(3)
    df["ISO_Year"] = df["ISO_Year"].map(_safe_int)
    df["ISO_Week"] = df["ISO_Week"].map(_safe_int)
    # Expandeer per oefening naar individuele SetNumber regels
    rows = []
    for r in df.itertuples(index=False):
        for s in range(1, int(getattr(r, "Sets") or 0) + 1):
            rows.append({
                "Workout": getattr(r, "Workout"),
                "Exercise": getattr(r, "Exercise"),
                "SetNumber": s,
                "ISO_Year": getattr(r, "ISO_Year"),
                "ISO_Week": getattr(r, "ISO_Week")
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Workout","Exercise","SetNumber","ISO_Year","ISO_Week"])

def _best_match(adf: Optional[pd.DataFrame],
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

    # Strikte lagen
    pick((adf["ISO_Year_int"] == iso_year) & (adf["ISO_Week_int"] == iso_week) &
         (adf["Workout_norm"] == w) & (adf["Exercise_norm"] == e) & (adf["SetNumber_int"] == s))
    pick((adf["ISO_Year_int"] == iso_year) & (adf["ISO_Week_int"] == iso_week) &
         (adf["Workout_norm"] == w) & (adf["Exercise_norm"] == e) & adf["SetNumber_int"].isna())
    pick((adf["ISO_Year_int"] == iso_year) & (adf["ISO_Week_int"] == iso_week) &
         (adf["Exercise_norm"] == e))
    pick(adf["ISO_Year_int"].isna() & adf["ISO_Week_int"].isna() &
         (adf["Workout_norm"] == w) & (adf["Exercise_norm"] == e) & (adf["SetNumber_int"] == s))
    pick(adf["ISO_Year_int"].isna() & adf["ISO_Week_int"].isna() &
         (adf["Workout_norm"] == w) & (adf["Exercise_norm"] == e) & adf["SetNumber_int"].isna())
    # Losser
    pick((adf["Exercise_norm"] == e) & (adf["SetNumber_int"] == s))
    pick((adf["Exercise_norm"] == e) & adf["SetNumber_int"].isna())

    return picks[0] if picks else None

def _compute_strength_reference(df: pd.DataFrame, lookback_weeks: int) -> pd.DataFrame:
    def e1rm_table(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.dropna(subset=["Weight", "Reps"])
        sub = sub[(sub["Weight"] > 0) & (sub["Reps"] > 0)].copy()
        if sub.empty:
            return pd.DataFrame(columns=["Exercise", "e1RM"])
        sub["est_1rm"] = _epley_1rm(sub["Weight"].astype(float), sub["Reps"].astype(int))
        return sub.groupby("Exercise", as_index=False)["est_1rm"].max().rename(columns={"est_1rm": "e1RM"})

    last_date = df["Date"].max()
    cutoff = last_date - pd.Timedelta(weeks=int(lookback_weeks))
    recent = df[df["Date"] >= cutoff].copy()
    e_recent = e1rm_table(recent).rename(columns={"e1RM": "e1RM_recent"})
    e_all = e1rm_table(df).rename(columns={"e1RM": "e1RM_all"})
    nz = df[df["Weight"] > 0].sort_values("Date").groupby("Exercise", as_index=False).tail(1)
    nz = nz[["Exercise", "Weight"]].rename(columns={"Weight": "last_nz_weight"})
    ref = pd.merge(e_recent, e_all, on="Exercise", how="outer")
    ref = pd.merge(ref, nz, on="Exercise", how="outer")
    return ref

def _choose_e1rm(row: pd.Series) -> Optional[float]:
    e_recent = _safe_float(row.get("e1RM_recent"))
    e_all = _safe_float(row.get("e1RM_all"))
    return e_recent if (e_recent is not None and not np.isnan(e_recent)) else e_all

def _calc_target(e1rm: Optional[float], pct: Optional[float], last_nz: Optional[float],
                 step: float, mode: str) -> Optional[float]:
    if pct is None:
        return None
    if e1rm is not None and not np.isnan(e1rm) and e1rm > 0:
        raw = e1rm * (pct / 100.0)
        return _round_weight(raw, step, mode)
    if last_nz is not None and not np.isnan(last_nz) and last_nz > 0:
        raw = last_nz * (pct / 100.0)
        return _round_weight(raw, step, mode)
    return None

# ==============================
# ---------- Pipeline ----------
# ==============================

if strong_file is None:
    st.info("Upload eerst je **Strong CSV** om een voorstel te genereren.")
    st.stop()

df_strong = _read_strong_csv(strong_file)
ref_tbl = _compute_strength_reference(df_strong, lookback_weeks)

advice_df = _read_advice_xlsx(advice_xlsx) if advice_xlsx is not None else None
workouts_plan = _read_workouts_xlsx(advice_xlsx) if advice_xlsx is not None else None

# In-app planner (optioneel)
if use_inapp:
    # opties voor Exercise uit je Strong
    exercises_list = sorted(df_strong["Exercise"].dropna().unique().tolist())
    default_rows = pd.DataFrame({
        "Workout": [default_workout_name]*3,
        "Exercise": exercises_list[:3] if len(exercises_list)>=3 else exercises_list,
        "Sets": [3]*min(3, len(exercises_list))
    })
    planner_df = st.data_editor(
        default_rows,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Workout": st.column_config.TextColumn("Workout"),
            "Exercise": st.column_config.SelectboxColumn("Exercise", options=exercises_list, required=True),
            "Sets": st.column_config.NumberColumn("Sets", min_value=1, max_value=10, step=1, required=True)
        ],
        key="planner_editor"
    )
    apply_plan = st.button("Plan toepassen")
else:
    apply_plan = False

# ---- Template bouwen (prioriteit): In-app plan → Excel workouts → default laatste sessies ----
template = None

if use_inapp and apply_plan and not planner_df.empty:
    rows = []
    for r in planner_df.itertuples(index=False):
        w = str(getattr(r, "Workout") or default_workout_name)
        ex = str(getattr(r, "Exercise"))
        n = int(getattr(r, "Sets") or 0)
        for s in range(1, n+1):
            rows.append({"Workout": w, "Exercise": ex, "SetNumber": s})
    template = pd.DataFrame(rows)

elif workouts_plan is not None and not workouts_plan.empty:
    # Als ISO_Year/Week in plan staan en afwijken van vandaag, we negeren dat hier (matching gebeurt later)
    template = workouts_plan[["Workout","Exercise","SetNumber"]].copy()

else:
    # fallback: laatste uitvoering per oefening ⇒ default 3 sets
    last_by_ex = df_strong.sort_values("Date").groupby("Exercise", as_index=False).tail(1)
    rows = []
    for r in last_by_ex.itertuples(index=False):
        for s in range(1, 3 + 1):
            rows.append({"Workout": default_workout_name, "Exercise": getattr(r, "Exercise"), "SetNumber": s})
    template = pd.DataFrame(rows)

# Merge referenties (e1RM, last weight)
template = template.merge(ref_tbl, on="Exercise", how="left")

# Advies-matching en doel-kg berekenen
today = datetime.now()
isoY, isoW = today.isocalendar().year, today.isocalendar().week

out_rows = []
matched_count = 0

for _, row in template.iterrows():
    ex = str(row["Exercise"])
    setno = int(row["SetNumber"])
    workout = str(row["Workout"])

    match = _best_match(advice_df, isoY, isoW, workout, ex, setno) if advice_df is not None else None
    if match is not None:
        matched_count += 1

    pct = _coerce_pct(match["%1RM"]) if match is not None else None
    reps = _safe_int(match["Reps"]) if match is not None else None
    sets_override = _safe_int(match["SetsOverride"]) if match is not None else None

    if (pct is None or reps is None) and use_global_defaults:
        pct = 70.0 if pct is None else pct
        reps = 8 if reps is None else reps

    if pct is not None and safety_delta != 0.0:
        pct = max(0.0, pct + float(safety_delta))

    e1rm_choice = _choose_e1rm(row)
    last_nz = _safe_float(row.get("last_nz_weight"))
    tgt_kg = _calc_target(e1rm_choice, pct, last_nz, kg_step, round_mode)

    out_rows.append({
        "Workout": workout,
        "Exercise": ex,
        "SetNumber": setno,
        "%1RM": pct,
        "Reps": reps,
        "TargetKg": tgt_kg,
        "e1RM": e1rm_choice,
        "last_nz_weight": last_nz,
        "SetsOverride": sets_override,
        "_Matched": match is not None
    })

plan = pd.DataFrame(out_rows)

# SetsOverride (indien gezet in advice)
def _apply_sets_override_to_display(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for ex, grp in df.groupby("Exercise", sort=False):
        max_set = grp["SetNumber"].max()
        ov = grp["SetsOverride"].dropna()
        n_sets = int(ov.iloc[0]) if (not ov.empty and ov.iloc[0] and ov.iloc[0] > 0) else int(max_set)
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
    margin-top: 0.75rem;
    margin-bottom: 0.25rem;
    font-size: 1.25rem;
    font-weight: 700;
}
.exercise-card {
    padding: 1rem;
    margin: 0.25rem 0 1rem 0;
    border-radius: 14px;
    border: 1px solid rgba(120,120,120,0.25);
    background: linear-gradient(180deg, rgba(250,250,250,0.95), rgba(242,242,242,0.95));
}
.set-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 14px;
}
.set-card {
    border: 1px solid rgba(100,100,100,0.25);
    border-radius: 12px;
    padding: 14px;
    background: #ffffff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.set-header {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 10px;
}
.pill-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}
.pill {
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 10px;
    padding: 10px 12px;
    background: #fafafa;
}
.pill .label {
    font-size: 0.75rem;
    color: #666;
    margin-bottom: 4px;
}
.pill .value {
    font-size: 1.1rem;
    font-weight: 700;
    color: #111;
}
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
    margin-top: 10px;
    font-size: 0.8rem;
    color: #666;
}
.divider {
    height: 1px;
    background: rgba(0,0,0,0.07);
    margin: 10px 0 12px 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==============================
# ---------- Header ----------
# ==============================

if plan_disp.empty:
    st.warning("Geen adviesregels beschikbaar.")
    st.stop()

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Unieke oefeningen", plan_disp["Exercise"].nunique())
with c2:
    st.metric("Totaal set-blokken", len(plan_disp))
with c3:
    st.metric("Lookback (weken)", lookback_weeks)
with c4:
    st.metric("Ronding (kg)", kg_step)
with c5:
    st.metric("Advies-matches", int((plan["_Matched"] == True).sum()))

# ==============================
# ---------- Render ----------
# ==============================

for ex, grp in plan_disp.groupby("Exercise", sort=False):
    grp = grp.sort_values("SetNumber")
    e1 = grp["e1RM"].dropna().max()
    header = f"{ex}  "
    sub = f"(e1RM: {e1:.1f} kg)" if isinstance(e1, (int, float)) and not np.isnan(e1) else "(e1RM: onbekend)"
    st.markdown(f"<div class='section-title'>{header}<span class='badge'>{sub}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='exercise-card'>", unsafe_allow_html=True)

    st.markdown("<div class='set-grid'>", unsafe_allow_html=True)
    for _, rr in grp.iterrows():
        pct = _safe_float(rr.get("%1RM"))
        reps = _safe_int(rr.get("Reps"))
        tgt = _safe_float(rr.get("TargetKg"))
        setn = _safe_int(rr.get("SetNumber"))

        st.markdown("<div class='set-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='set-header'>Set {'' if setn is None else int(setn)}</div>", unsafe_allow_html=True)

        st.markdown("<div class='pill-row'>", unsafe_allow_html=True)
        st.markdown("<div class='pill'><div class='label'>%1RM</div>"
                    f"<div class='value'>{_fmt_pct(pct)}</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='pill'><div class='label'>Reps</div>"
                    f"<div class='value'>{_fmt_int(reps)}</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='pill'><div class='label'>Doel-kg</div>"
                    f"<div class='value'>{_fmt_kg(tgt)}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if show_warmups and (tgt is not None) and not (isinstance(tgt, float) and np.isnan(tgt)) and tgt > 0:
            w3 = _round_weight(tgt * 0.85, kg_step, round_mode)
            w2 = _round_weight(tgt * 0.70, kg_step, round_mode)
            w1 = _round_weight(max(bar_weight, tgt * 0.50), kg_step, round_mode)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='hint'><b>Warm-up</b>: "
                        f"{w1:.1f} × 5  →  {w2:.1f} × 3  →  {w3:.1f} × 1–2</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# ---------- Export & Debug ----------
# ==============================

with st.expander("Exporteren"):
    exp = plan_disp.copy()
    exp = exp[["Workout","Exercise","SetNumber","%1RM","Reps","TargetKg","e1RM","last_nz_weight","SetsOverride"]]
    csv_bytes = exp.to_csv(index=False).encode("utf-8")
    st.download_button("Download voorstel als CSV", data=csv_bytes, file_name="coaching_voorstel.csv", mime="text/csv")

if show_debug:
    st.subheader("Diagnostiek")
    st.write("- Adviesregels (advice):", 0 if advice_df is None else len(advice_df))
    st.write("- Workouts-plan regels (workouts):", 0 if workouts_plan is None else len(workouts_plan))
    st.write("- Template regels:", len(template))
    st.write("- Matches:", int((plan["_Matched"] == True).sum()))
    st.dataframe(plan.head(50))
