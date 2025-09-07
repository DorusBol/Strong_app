# app.py — Next Workout (Kaartenweergave) met 2-weken schema, A/B-varianten en topfilters
# Vereist: streamlit, pandas, numpy, openpyxl (voor .xlsx)
import io
import math
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- UI ----------------
st.set_page_config(page_title="Next Workout — Kaartenweergave", layout="wide")
st.title("Next Workout — Coachingvoorstel (Kaartenweergave)")
st.caption(
    "Upload je Strong-export (CSV). Optioneel: Excel met tabs **workouts**, **schedule**, **advice**. "
    "Filter bovenaan op Week/Block/Workout. Per set tonen we %1RM, Reps en Doel-kg."
)

c_up1, c_up2 = st.columns(2)
with c_up1:
    strong_file = st.file_uploader("Strong CSV", type=["csv"])
with c_up2:
    plan_xlsx = st.file_uploader("Planner-Excel (workouts/schedule/advice)", type=["xlsx"])

with st.expander("Instellingen", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kg_step = st.number_input("Rondingsstap (kg)", min_value=0.25, max_value=5.0, step=0.25, value=1.0)
    with c2:
        round_mode = st.selectbox("Rondingsmodus", ["nearest", "down", "up"], index=0)
    with c3:
        safety_delta = st.number_input("Veiligheidsmarge op %1RM (±%)", value=0.0, step=1.0, format="%.1f")
    with c4:
        lookback_weeks = st.number_input("Lookback e1RM (weken)", min_value=1, max_value=52, value=8, step=1)

    c5, c6, c7 = st.columns(3)
    with c5:
        default_workout_name = st.text_input("Default Workout (fallback)", value="Push")
    with c6:
        show_warmups = st.checkbox("Toon warm-up hints", value=False)
    with c7:
        bar_weight = st.number_input("Stanggewicht (kg)", min_value=0.0, max_value=30.0, value=20.0, step=0.5)

    c8, c9 = st.columns(2)
    with c8:
        use_global_defaults = st.checkbox("Defaults bij geen advies", value=True,
                                          help="Gebruik 70% × 8 wanneer er geen advies/fallback is.")
    with c9:
        show_debug = st.checkbox("Toon diagnostiek", value=False)

# -------------- Helpers --------------
def _norm(s) -> str:
    if pd.isna(s): return ""
    return " ".join(str(s).strip().lower().split())

def _safe_int(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == ""): return None
        return int(float(x))
    except: return None

def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(x, str) and x.strip() == ""): return None
        return float(x)
    except: return None

def _coerce_pct(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)): return None
    s = str(val).replace("%","").strip()
    try: return float(s)
    except: return None

def _fmt_pct(x): return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0f}%"
def _fmt_int(x): return "" if x is None else f"{int(x)}"
def _fmt_kg(x):  return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f} kg"

def _round_weight(kg: float, step: float, mode: str) -> float:
    q = kg / step if step > 0 else kg
    if step <= 0: return round(kg, 1)
    if mode == "up": return round(math.ceil(q) * step, 3)
    if mode == "down": return round(math.floor(q) * step, 3)
    return round(round(q) * step, 3)

def _epley_1rm(weight: float, reps: int) -> float: return weight * (1.0 + reps / 30.0)

def _isocalendar_tuple(ts: pd.Timestamp) -> Tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)

def _read_strong_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    req = {"Date","Exercise Name","Weight","Reps"}
    missing = req - set(df.columns)
    if missing:
        st.error(f"Strong CSV mist kolommen: {sorted(missing)}"); st.stop()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Exercise"] = df["Exercise Name"].astype(str)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Reps"]   = pd.to_numeric(df["Reps"], errors="coerce")
    df["ISO_Year"], df["ISO_Week"] = zip(*df["Date"].map(_isocalendar_tuple))
    return df

def _read_plan_bytes(upload) -> Optional[bytes]:
    if upload is None: return None
    try: return upload.read()
    except: return None

def _read_sheet(bytes_, sheet) -> Optional[pd.DataFrame]:
    if not bytes_: return None
    try: return pd.read_excel(io.BytesIO(bytes_), sheet_name=sheet, engine="openpyxl")
    except ValueError: return None

def _prep_advice(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None: return None
    for c in ["ISO_Year","ISO_Week","Workout","Exercise","SetNumber","%1RM","Reps","SetsOverride"]:
        if c not in df.columns: df[c] = np.nan
    df["Workout_norm"]  = df["Workout"].map(_norm)
    df["Exercise_norm"] = df["Exercise"].map(_norm)
    df["SetNumber_int"] = df["SetNumber"].map(_safe_int)
    df["ISO_Year_int"]  = df["ISO_Year"].map(_safe_int)
    df["ISO_Week_int"]  = df["ISO_Week"].map(_safe_int)
    df["%1RM"]          = df["%1RM"].map(_coerce_pct)
    df["Reps"]          = df["Reps"].map(_safe_int)
    df["SetsOverride"]  = df["SetsOverride"].map(_safe_int)
    df["spec"] = (df["ISO_Year_int"].notna().astype(int)
                + df["ISO_Week_int"].notna().astype(int)
                + (df["Workout_norm"]!="").astype(int)
                + (df["Exercise_norm"]!="").astype(int)
                + df["SetNumber_int"].notna().astype(int))
    df["recency"] = list(zip(df["ISO_Year_int"].fillna(-1e9), df["ISO_Week_int"].fillna(-1e9)))
    return df

def _prep_workouts(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None: return None
    for c in ["Workout","Variant","Exercise","Sets"]:
        if c not in df.columns: df[c] = np.nan
    df["Workout"] = df["Workout"].fillna("").astype(str)
    df["Variant"] = df["Variant"].fillna("").astype(str)
    df["Exercise"]= df["Exercise"].fillna("").astype(str)
    df["Sets"]    = df["Sets"].map(_safe_int).fillna(3)
    # Expand naar individuele SetNumber-rijen
    rows = []
    for r in df.itertuples(index=False):
        for s in range(1, int(getattr(r,"Sets") or 0)+1):
            rows.append({"Workout":r.Workout,"Variant":r.Variant,"Exercise":r.Exercise,"SetNumber":s})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Workout","Variant","Exercise","SetNumber"])

def _prep_schedule(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None: return None
    for c in ["ISO_Year","ISO_Week","Block","Workout","Variant"]:
        if c not in df.columns: df[c] = np.nan
    df["ISO_Year"] = df["ISO_Year"].map(_safe_int)
    df["ISO_Week"] = df["ISO_Week"].map(_safe_int)
    df["Block"]    = df["Block"].fillna("").astype(str)
    df["Workout"]  = df["Workout"].fillna("").astype(str)
    df["Variant"]  = df["Variant"].fillna("").astype(str)
    df["WorkoutKey"] = df["Workout"].str.strip()+" "+df["Variant"].str.strip()
    return df

def _best_match(adf: Optional[pd.DataFrame], iso_year, iso_week, workout, exercise, setno) -> Optional[pd.Series]:
    if adf is None or adf.empty: return None
    w, e, s = _norm(workout), _norm(exercise), setno
    picks: List[pd.Series] = []
    def pick(mask):
        sub = adf[mask]
        if not sub.empty:
            sub = sub.sort_values(by=["spec","recency"], ascending=[False, False], kind="mergesort")
            picks.append(sub.iloc[0])
    # strikte lagen
    pick((adf["ISO_Year_int"]==iso_year)&(adf["ISO_Week_int"]==iso_week)&
         (adf["Workout_norm"]==w)&(adf["Exercise_norm"]==e)&(adf["SetNumber_int"]==s))
    pick((adf["ISO_Year_int"]==iso_year)&(adf["ISO_Week_int"]==iso_week)&
         (adf["Workout_norm"]==w)&(adf["Exercise_norm"]==e)&adf["SetNumber_int"].isna())
    pick((adf["ISO_Year_int"]==iso_year)&(adf["ISO_Week_int"]==iso_week)&(adf["Exercise_norm"]==e))
    pick(adf["ISO_Year_int"].isna()&adf["ISO_Week_int"].isna()&
         (adf["Workout_norm"]==w)&(adf["Exercise_norm"]==e)&(adf["SetNumber_int"]==s))
    pick(adf["ISO_Year_int"].isna()&adf["ISO_Week_int"].isna()&
         (adf["Workout_norm"]==w)&(adf["Exercise_norm"]==e)&adf["SetNumber_int"].isna())
    # losser
    pick((adf["Exercise_norm"]==e)&(adf["SetNumber_int"]==s))
    pick((adf["Exercise_norm"]==e)&adf["SetNumber_int"].isna())
    return picks[0] if picks else None

def _compute_strength_reference(df: pd.DataFrame, lookback_weeks: int) -> pd.DataFrame:
    def e1rm_table(sub: pd.DataFrame):
        sub = sub.dropna(subset=["Weight","Reps"])
        sub = sub[(sub["Weight"]>0)&(sub["Reps"]>0)].copy()
        if sub.empty: return pd.DataFrame(columns=["Exercise","e1RM"])
        sub["est_1rm"] = _epley_1rm(sub["Weight"].astype(float), sub["Reps"].astype(int))
        return sub.groupby("Exercise", as_index=False)["est_1rm"].max().rename(columns={"est_1rm":"e1RM"})
    last_date = df["Date"].max()
    cutoff = last_date - pd.Timedelta(weeks=int(lookback_weeks))
    e_recent = e1rm_table(df[df["Date"]>=cutoff]).rename(columns={"e1RM":"e1RM_recent"})
    e_all    = e1rm_table(df).rename(columns={"e1RM":"e1RM_all"})
    nz = df[df["Weight"]>0].sort_values("Date").groupby("Exercise", as_index=False).tail(1)[["Exercise","Weight"]]
    nz = nz.rename(columns={"Weight":"last_nz_weight"})
    ref = e_recent.merge(e_all, on="Exercise", how="outer").merge(nz, on="Exercise", how="outer")
    return ref

def _choose_e1rm(row): 
    er = _safe_float(row.get("e1RM_recent")); ea = _safe_float(row.get("e1RM_all"))
    return er if (er is not None and not np.isnan(er)) else ea

def _calc_target(e1rm, pct, last_nz, step, mode):
    if pct is None: return None
    if e1rm is not None and not np.isnan(e1rm) and e1rm>0:
        return _round_weight(e1rm*(pct/100.0), step, mode)
    if last_nz is not None and not np.isnan(last_nz) and last_nz>0:
        return _round_weight(last_nz*(pct/100.0), step, mode)
    return None

# -------------- Pipeline --------------
if strong_file is None:
    st.info("Upload eerst je **Strong CSV**."); st.stop()

df_strong = _read_strong_csv(strong_file)
ref_tbl = _compute_strength_reference(df_strong, lookback_weeks)

file_bytes = _read_plan_bytes(plan_xlsx)
workouts_raw = _read_sheet(file_bytes, "workouts")
schedule_raw = _read_sheet(file_bytes, "schedule")
advice_raw   = _read_sheet(file_bytes, "advice")

workouts_df = _prep_workouts(workouts_raw)
schedule_df = _prep_schedule(schedule_raw)
advice_df   = _prep_advice(advice_raw)

# --------- Filters (bovenaan) ----------
flt_cols = st.columns([1,1,1,1,2])
week_options = []
block_options = []
wkkey_options = []
if schedule_df is not None and not schedule_df.empty:
    schedule_df["WeekStr"] = schedule_df["ISO_Year"].astype(str) + "-W" + schedule_df["ISO_Week"].astype(str)
    week_options = sorted(schedule_df["WeekStr"].dropna().unique().tolist())
    sel_week = flt_cols[0].selectbox("Week (ISO)", options=week_options, index=0 if week_options else None)
    # blocks for selected week
    block_options = schedule_df.loc[schedule_df["WeekStr"]==sel_week,"Block"].tolist() if week_options else []
    block_options = [b for b in ["1A","1B","1C","1D","2A","2B","2C","2D"] if b in block_options] or sorted(block_options)
    sel_block = flt_cols[1].selectbox("Block", options=block_options, index=0 if block_options else None)
    # derive WorkoutKey for selection + manual override
    sub = schedule_df[(schedule_df["WeekStr"]==sel_week)&(schedule_df["Block"]==sel_block)]
    wkkey = (sub["Workout"].iloc[0].strip()+" "+sub["Variant"].iloc[0].strip()) if not sub.empty else ""
    wkkey_options = sorted((schedule_df["Workout"].str.strip()+" "+schedule_df["Variant"].str.strip()).unique().tolist())
    sel_wkkey = flt_cols[2].selectbox("Workout+Variant", options=wkkey_options, index=wkkey_options.index(wkkey) if wkkey in wkkey_options else 0)
else:
    sel_week = None
    sel_block = None
    # fallback: build from workouts_df or default
    if workouts_df is not None and not workouts_df.empty:
        wkkey_options = sorted((workouts_df["Workout"].str.strip()+" "+workouts_df["Variant"].str.strip()).unique().tolist())
        sel_wkkey = flt_cols[2].selectbox("Workout+Variant", options=wkkey_options, index=0)
    else:
        sel_wkkey = flt_cols[2].text_input("Workout+Variant", value=f"{default_workout_name} A")

# -------------- Template bouwen --------------
if workouts_df is not None and not workouts_df.empty:
    wk, var = [s.strip() for s in sel_wkkey.split(" ", 1)] if " " in sel_wkkey else (sel_wkkey.strip(), "")
    template = workouts_df[(workouts_df["Workout"]==wk)&(workouts_df["Variant"]==var)].copy()
    if template.empty and not workouts_df.empty:
        # fallback: neem alle varianten voor gekozen workout
        template = workouts_df[workouts_df["Workout"]==wk].copy()
else:
    # fallback op laatste sessies — 3 sets
    last_by_ex = df_strong.sort_values("Date").groupby("Exercise", as_index=False).tail(1)
    rows = []
    for r in last_by_ex.itertuples(index=False):
        for s in range(1, 4):
            rows.append({"Workout": default_workout_name, "Variant":"", "Exercise": r.Exercise, "SetNumber": s})
    template = pd.DataFrame(rows)

# Merge e1RM/last weight
template = template.merge(ref_tbl, on="Exercise", how="left")

# Advies + targets
today = datetime.now(); isoY, isoW = today.isocalendar().year, today.isocalendar().week
out_rows = []
for _, row in template.iterrows():
    ex, setno = str(row["Exercise"]), int(row["SetNumber"])
    wk = str(row.get("Workout", default_workout_name))

    match = _best_match(advice_df, isoY, isoW, wk, ex, setno) if advice_df is not None else None
    pct  = _coerce_pct(match["%1RM"]) if match is not None else None
    reps = _safe_int(match["Reps"])   if match is not None else None
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
        "Workout": wk, "Variant": row.get("Variant",""),
        "Exercise": ex, "SetNumber": setno,
        "%1RM": pct, "Reps": reps, "TargetKg": tgt_kg,
        "e1RM": e1rm_choice, "last_nz_weight": last_nz,
        "SetsOverride": sets_override
    })

plan = pd.DataFrame(out_rows)

# SetsOverride toepassen
def _apply_sets_override(df: pd.DataFrame) -> pd.DataFrame:
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

plan_disp = _apply_sets_override(plan)

# -------------- Styling --------------
CUSTOM_CSS = """
<style>
.section-title { margin-top: .75rem; margin-bottom: .25rem; font-size: 1.25rem; font-weight: 700; }
.exercise-card { padding: 1rem; margin: .25rem 0 1rem 0; border-radius: 14px; border: 1px solid rgba(120,120,120,.25);
  background: linear-gradient(180deg, rgba(250,250,250,.95), rgba(242,242,242,.95)); }
.set-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; }
.set-card { border: 1px solid rgba(100,100,100,.25); border-radius: 12px; padding: 14px; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
.set-header { font-weight: 700; font-size: 1rem; margin-bottom: 10px; }
.pill-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
.pill { border: 1px solid rgba(0,0,0,.1); border-radius: 10px; padding: 10px 12px; background: #fafafa; }
.pill .label { font-size: .75rem; color: #666; margin-bottom: 4px; }
.pill .value { font-size: 1.1rem; font-weight: 700; color: #111; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:700; color:#124; background:#e9f3ff; border:1px solid #cfe4ff; }
.hint { margin-top:10px; font-size:.8rem; color:#666; }
.divider { height:1px; background: rgba(0,0,0,.07); margin:10px 0 12px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------- Header metrics --------------
if plan_disp.empty:
    st.warning("Geen adviesregels beschikbaar."); st.stop()

m1, m2, m3, m4, m5 = st.columns(5)
with m1: st.metric("Unieke oefeningen", plan_disp["Exercise"].nunique())
with m2: st.metric("Totaal set-blokken", len(plan_disp))
with m3: st.metric("Lookback (weken)", lookback_weeks)
with m4: st.metric("Ronding (kg)", kg_step)
with m5:
    if schedule_df is not None and not schedule_df.empty and week_options:
        st.metric("Selectie", f"{sel_week} · {sel_block} · {sel_wkkey}")
    else:
        st.metric("Selectie", sel_wkkey)

# -------------- Render --------------
for ex, grp in plan_disp.groupby("Exercise", sort=False):
    grp = grp.sort_values("SetNumber")
    e1 = grp["e1RM"].dropna().max()
    wk = grp["Workout"].iloc[0] if "Workout" in grp.columns else default_workout_name
    var = grp["Variant"].iloc[0] if "Variant" in grp.columns else ""
    header = f"{ex}"
    sub = f"(e1RM: {e1:.1f} kg) · {wk} {var}".strip()
    st.markdown(f"<div class='section-title'>{header} <span class='badge'>{sub}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='exercise-card'>", unsafe_allow_html=True)

    st.markdown("<div class='set-grid'>", unsafe_allow_html=True)
    for _, rr in grp.iterrows():
        pct = _safe_float(rr.get("%1RM")); reps = _safe_int(rr.get("Reps"))
        tgt = _safe_float(rr.get("TargetKg")); setn = _safe_int(rr.get("SetNumber"))

        st.markdown("<div class='set-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='set-header'>Set {'' if setn is None else int(setn)}</div>", unsafe_allow_html=True)
        st.markdown("<div class='pill-row'>", unsafe_allow_html=True)
        st.markdown("<div class='pill'><div class='label'>%1RM</div><div class='value'>"+_fmt_pct(pct)+"</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='pill'><div class='label'>Reps</div><div class='value'>"+_fmt_int(reps)+"</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='pill'><div class='label'>Doel-kg</div><div class='value'>"+_fmt_kg(tgt)+"</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if show_warmups and tgt not in (None, np.nan) and tgt and tgt>0:
            w3 = _round_weight(tgt*0.85, kg_step, round_mode)
            w2 = _round_weight(tgt*0.70, kg_step, round_mode)
            w1 = _round_weight(max(bar_weight, tgt*0.50), kg_step, round_mode)
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='hint'><b>Warm-up</b>: {w1:.1f} × 5 → {w2:.1f} × 3 → {w3:.1f} × 1–2</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------- Export & Debug --------------
with st.expander("Exporteren"):
    exp = plan_disp[["Workout","Variant","Exercise","SetNumber","%1RM","Reps","TargetKg","e1RM","last_nz_weight","SetsOverride"]].copy()
    st.download_button("Download voorstel (CSV)", data=exp.to_csv(index=False).encode("utf-8"),
                       file_name="coaching_voorstel.csv", mime="text/csv")

if show_debug:
    st.subheader("Diagnostiek")
    st.write("- Workouts-rows:", 0 if workouts_df is None else len(workouts_df))
    st.write("- Schedule-rows:", 0 if schedule_df is None else len(schedule_df))
    st.write("- Advice-rows:", 0 if advice_df is None else len(advice_df))
    st.write("- Template-rows:", len(template))
    st.dataframe(plan.head(50))
