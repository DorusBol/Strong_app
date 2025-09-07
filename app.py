# app.py — Periodized planner (CSV-only)
# Vereist: streamlit, pandas, numpy
import io, math
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- UI ----------------
st.set_page_config(page_title="Next Workout — Periodized (CSV)", layout="wide")
st.title("Next Workout — Periodized Coaching (CSV-only)")

st.caption(
    "Upload je **Strong CSV** en daarnaast **meerdere CSV-bestanden** voor het schema: "
    "`schedule.csv`, `workouts.csv`, `advice_periodization.csv`. "
    "De app herkent ze ook als de bestandsnamen afwijken, zolang de kolommen kloppen."
)

c_up1, c_up2 = st.columns(2)
with c_up1:
    strong_file = st.file_uploader("Strong CSV", type=["csv"])
with c_up2:
    plan_files = st.file_uploader(
        "Periodized CSV’s (meerdere selecteren tegelijk)",
        type=["csv"],
        accept_multiple_files=True
    )

with st.expander("Instellingen", expanded=True):
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        kg_step = st.number_input("Rondingsstap (kg)", 0.25, 5.0, 1.0, step=0.25)
    with s2:
        round_mode = st.selectbox("Rondingsmodus", ["nearest", "down", "up"], index=0)
    with s3:
        safety_delta = st.number_input("Veiligheidsmarge op %1RM (±%)", value=0.0, step=1.0, format="%.1f")
    with s4:
        lookback_weeks = st.number_input("Lookback e1RM (weken)", 1, 52, 8, step=1)
    s5, s6 = st.columns(2)
    with s5:
        use_defaults = st.checkbox("Defaults bij ontbrekend advies (70% × 8)", True)
    with s6:
        show_debug = st.checkbox("Toon diagnostiek", False)

# ---------------- Helpers ----------------
def _norm(s):
    if pd.isna(s): return ""
    return " ".join(str(s).strip().lower().split())

def _safe_int(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x,float) and np.isnan(x)) or (isinstance(x,str) and x.strip()==""): return None
        return int(float(x))
    except: return None

def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x,float) and np.isnan(x)) or (isinstance(x,str) and x.strip()==""): return None
        return float(x)
    except: return None

def _coerce_pct(val) -> Optional[float]:
    if val is None or (isinstance(val,float) and np.isnan(val)): return None
    try: return float(str(val).replace("%","").strip())
    except: return None

def _fmt_pct(x): return "" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.0f}%"
def _fmt_int(x): return "" if x is None else f"{int(x)}"
def _fmt_kg(x):  return "" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.1f} kg"

def _round_weight(kg: float, step: float, mode: str) -> float:
    if step <= 0: return round(kg, 1)
    q = kg / step
    if mode == "up":   return round(math.ceil(q) * step, 3)
    if mode == "down": return round(math.floor(q) * step, 3)
    return round(round(q) * step, 3)

def _epley_1rm(w: float, r: int) -> float:
    return w * (1.0 + r / 30.0)

def _isocalendar_tuple(ts: pd.Timestamp) -> Tuple[int, int]:
    iso = ts.isocalendar()
    return int(iso.year), int(iso.week)

def _read_strong_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    req = {"Date", "Exercise Name", "Weight", "Reps"}
    miss = req - set(df.columns)
    if miss:
        st.error(f"Strong CSV mist kolommen: {sorted(miss)}")
        st.stop()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Exercise"] = df["Exercise Name"].astype(str)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["Reps"] = pd.to_numeric(df["Reps"], errors="coerce")
    df["ISO_Year"], df["ISO_Week"] = zip(*df["Date"].map(_isocalendar_tuple))
    return df

def _compute_refs(df: pd.DataFrame, weeks:int) -> pd.DataFrame:
    def e1tbl(sub):
        sub = sub.dropna(subset=["Weight","Reps"])
        sub = sub[(sub["Weight"]>0) & (sub["Reps"]>0)]
        if sub.empty: return pd.DataFrame(columns=["Exercise","e1RM"])
        sub = sub.copy()
        sub["est_1rm"] = _epley_1rm(sub["Weight"].astype(float), sub["Reps"].astype(int))
        return sub.groupby("Exercise", as_index=False)["est_1rm"].max().rename(columns={"est_1rm":"e1RM"})
    last = df["Date"].max()
    cutoff = last - pd.Timedelta(weeks=int(weeks))
    e_recent = e1tbl(df[df["Date"]>=cutoff]).rename(columns={"e1RM":"e1RM_recent"})
    e_all    = e1tbl(df).rename(columns={"e1RM":"e1RM_all"})
    nz = df[df["Weight"]>0].sort_values("Date").groupby("Exercise", as_index=False).tail(1)[["Exercise","Weight"]]
    nz = nz.rename(columns={"Weight":"last_nz_weight"})
    return e_recent.merge(e_all, on="Exercise", how="outer").merge(nz, on="Exercise", how="outer")

def _e1rm_choice(row):
    er = _safe_float(row.get("e1RM_recent"))
    ea = _safe_float(row.get("e1RM_all"))
    return er if (er is not None and not np.isnan(er)) else ea

def _expand_advice_multivals(adf: pd.DataFrame, workouts_df: pd.DataFrame) -> pd.DataFrame:
    """
    advice_periodization (CSV): Workout,Exercise,Pcts,Reps,SetsOverride
      - Pcts: comma-separated percentages (bijv. '75,80,70')
      - Reps: comma-separated reps (bijv. '8,6,10')
      - SetsOverride: optioneel
    Expand naar per-set regels; lengte bepaald door SetsOverride of workouts.Sets of max(len(Pcts), len(Reps)).
    """
    rows = []
    sets_map: Dict[Tuple[str,str], int] = {}
    if workouts_df is not None and not workouts_df.empty:
        sets_map = workouts_df.groupby(["Workout","Exercise"])["Sets"].max().to_dict()

    for r in adf.itertuples(index=False):
        w = str(getattr(r, "Workout"))
        ex = str(getattr(r, "Exercise"))
        pcts = [p.strip() for p in str(getattr(r, "Pcts") or "").split(",") if str(p).strip()!=""]
        reps = [p.strip() for p in str(getattr(r, "Reps") or "").split(",") if str(p).strip()!=""]
        so   = _safe_int(getattr(r, "SetsOverride") if "SetsOverride" in adf.columns else None)
        n = so if so else _safe_int(sets_map.get((w,ex)))
        if n is None: n = max(len(pcts), len(reps)) or 3

        if not pcts: pcts = ["70"] * n
        if not reps: reps = ["8"] * n
        if len(pcts) < n: pcts += [pcts[-1]] * (n - len(pcts))
        if len(reps) < n: reps += [reps[-1]] * (n - len(reps))

        for i in range(n):
            rows.append({
                "Workout": w,
                "Exercise": ex,
                "SetNumber": i + 1,
                "%1RM": _coerce_pct(pcts[i]),
                "Reps": _safe_int(reps[i])
            })
    return pd.DataFrame(rows)

def _detect_table(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.strip())
    if {"ISO_Year","ISO_Week","Block","Workout"}.issubset(cols): return "schedule"
    if {"Workout","Variant","Exercise","Sets"}.issubset(cols): return "workouts"
    if {"Workout","Exercise","Pcts","Reps"}.issubset(cols): return "advice_periodization"
    # fallback: try minimal forms
    if {"Workout","Exercise","Sets"}.issubset(cols): return "workouts"
    return "unknown"

def _read_plan_csvs(files) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    sch, wko, adv = None, None, None
    for f in files or []:
        try:
            df = pd.read_csv(f)
        except Exception:
            st.warning(f"Kon CSV niet lezen: {getattr(f,'name', 'bestandsobject')}")
            continue
        kind = _detect_table(df)
        if kind == "schedule":
            sch = df.copy()
        elif kind == "workouts":
            wko = df.copy()
        elif kind == "advice_periodization":
            adv = df.copy()
        else:
            # heuristiek via bestandsnaam
            name = (getattr(f, "name", "") or "").lower()
            if "sched" in name and sch is None: sch = df.copy()
            elif "work" in name and wko is None: wko = df.copy()
            elif "advice" in name or "period" in name and adv is None: adv = df.copy()
    return sch, wko, adv

# ---------------- Pipeline ----------------
if strong_file is None:
    st.info("Upload eerst je **Strong CSV**.")
    st.stop()

df_strong = _read_strong_csv(strong_file)
ref_tbl = _compute_refs(df_strong, lookback_weeks)

schedule_df, workouts_df, advicep_df = _read_plan_csvs(plan_files)

# Validatie
if schedule_df is None or workouts_df is None or advicep_df is None:
    st.error(
        "Benodigde CSV’s niet gevonden. Upload tegelijk:\n"
        "- schedule.csv  (kolommen: ISO_Year, ISO_Week, Block, Workout)\n"
        "- workouts.csv  (kolommen: Workout, Variant, Exercise, Sets)\n"
        "- advice_periodization.csv  (kolommen: Workout, Exercise, Pcts, Reps, SetsOverride opt.)"
    )
    st.stop()

# Normaliseer types/kolommen
schedule_df["ISO_Year"] = schedule_df["ISO_Year"].map(_safe_int)
schedule_df["ISO_Week"] = schedule_df["ISO_Week"].map(_safe_int)
schedule_df["Block"]    = schedule_df["Block"].astype(str)
schedule_df["Workout"]  = schedule_df["Workout"].astype(str)
schedule_df["WeekStr"]  = schedule_df["ISO_Year"].astype(int).astype(str) + "-W" + schedule_df["ISO_Week"].astype(int).astype(str)

workouts_df["Workout"]  = workouts_df["Workout"].astype(str)
if "Variant" not in workouts_df.columns: workouts_df["Variant"] = ""
workouts_df["Variant"]  = workouts_df["Variant"].fillna("").astype(str)
workouts_df["Exercise"] = workouts_df["Exercise"].astype(str)
workouts_df["Sets"]     = workouts_df["Sets"].map(_safe_int).fillna(3).astype(int)

if "SetsOverride" not in advicep_df.columns: advicep_df["SetsOverride"] = np.nan
advicep_df["Workout"]   = advicep_df["Workout"].astype(str)
advicep_df["Exercise"]  = advicep_df["Exercise"].astype(str)

# ---------------- Filters (bovenaan) ----------------
fc = st.columns([1,1,2])
weeks_sorted = sorted(schedule_df["WeekStr"].dropna().unique().tolist())
sel_week = fc[0].selectbox("Week (ISO)", weeks_sorted, index=0 if weeks_sorted else None)

week_blocks = schedule_df.loc[schedule_df["WeekStr"]==sel_week, "Block"].unique().tolist()
week_blocks_sorted = sorted(week_blocks, key=lambda b: (len(b), b))
sel_block = fc[1].selectbox("Block", week_blocks_sorted, index=0 if week_blocks_sorted else None)

wk_row = schedule_df[(schedule_df["WeekStr"]==sel_week) & (schedule_df["Block"]==sel_block)]
wk_default = wk_row["Workout"].iloc[0] if not wk_row.empty else (workouts_df["Workout"].iloc[0] if not workouts_df.empty else "")
workout_options = sorted(schedule_df["Workout"].dropna().unique().tolist())
sel_workout = fc[2].selectbox("Workout", [wk_default] + [w for w in workout_options if w != wk_default], index=0)

# ---------------- Build sessie-template ----------------
adv_expanded = _expand_advice_multivals(advicep_df[advicep_df["Workout"]==sel_workout], workouts_df)
tmpl = workouts_df[workouts_df["Workout"]==sel_workout][["Workout","Variant","Exercise","Sets"]].copy()

rows = []
for r in tmpl.itertuples(index=False):
    w, v, ex, n = r.Workout, r.Variant, r.Exercise, int(r.Sets)
    sub = adv_expanded[(adv_expanded["Workout"]==w) & (adv_expanded["Exercise"]==ex)].sort_values("SetNumber")
    if sub.empty:
        for s in range(1, n+1):
            rows.append({"Workout":w,"Variant":v,"Exercise":ex,"SetNumber":s,"%1RM":70.0 if use_defaults else None,"Reps":8 if use_defaults else None})
    else:
        # ensure exactly n sets
        if len(sub) < n:
            last = sub.iloc[-1].to_dict()
            for s in range(len(sub)+1, n+1):
                last2 = last.copy(); last2["SetNumber"] = s
                sub = pd.concat([sub, pd.DataFrame([last2])], ignore_index=True)
        rows += sub.head(n).assign(Variant=v).to_dict(orient="records")

plan = pd.DataFrame(rows)
plan = plan.merge(ref_tbl, on="Exercise", how="left")
plan["e1RM"] = plan.apply(lambda r: _e1rm_choice(r), axis=1)

def _target(e1rm, pct, last_nz):
    if pct is None: return None
    base = None
    if e1rm is not None and not np.isnan(e1rm) and e1rm > 0: base = e1rm
    elif last_nz is not None and not np.isnan(last_nz) and last_nz > 0: base = last_nz
    if base is None: return None
    p = max(0.0, pct + float(safety_delta)) if safety_delta != 0 else pct
    return _round_weight(base * (p / 100.0), kg_step, round_mode)

plan["TargetKg"] = plan.apply(lambda r: _target(r["e1RM"], r["%1RM"], r.get("last_nz_weight")), axis=1)

# ---------------- Metrics ----------------
mc = st.columns(4)
with mc[0]: st.metric("Oefeningen", plan["Exercise"].nunique())
with mc[1]: st.metric("Set-blokken", len(plan))
with mc[2]: st.metric("Week", sel_week or "-")
with mc[3]: st.metric("Workout", sel_workout or "-")

# ---------------- Styling ----------------
CSS = """
<style>
.section-title { margin-top:.75rem; margin-bottom:.25rem; font-size:1.25rem; font-weight:700; }
.exercise-card { padding:1rem; margin:.25rem 0 1rem 0; border-radius:14px; border:1px solid rgba(120,120,120,.25);
  background:linear-gradient(180deg, rgba(250,250,250,.95), rgba(242,242,242,.95)); }
.set-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(320px,1fr)); gap:14px; }
.set-card { border:1px solid rgba(100,100,100,.25); border-radius:12px; padding:14px; background:#fff; box-shadow:0 1px 4px rgba(0,0,0,.06); }
.set-header { font-weight:700; font-size:1rem; margin-bottom:10px; }
.pill-row { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; }
.pill { border:1px solid rgba(0,0,0,.1); border-radius:10px; padding:10px 12px; background:#fafafa; }
.pill .label { font-size:.75rem; color:#666; margin-bottom:4px; }
.pill .value { font-size:1.1rem; font-weight:700; color:#111; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; font-weight:700; color:#124; background:#e9f3ff; border:1px solid #cfe4ff; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Render ----------------
for ex, grp in plan.groupby("Exercise", sort=False):
    grp = grp.sort_values("SetNumber")
    e1 = grp["e1RM"].dropna().max()
    badge = f"(e1RM: {e1:.1f} kg)" if isinstance(e1,(int,float)) and not np.isnan(e1) else "(e1RM: onbekend)"
    wk = sel_workout
    st.markdown(f"<div class='section-title'>{ex} <span class='badge'>{badge} · {wk}</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='exercise-card'>", unsafe_allow_html=True)
    st.markdown("<div class='set-grid'>", unsafe_allow_html=True)
    for _, rr in grp.iterrows():
        st.markdown("<div class='set-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='set-header'>Set {int(rr['SetNumber'])}</div>", unsafe_allow_html=True)
        st.markdown("<div class='pill-row'>", unsafe_allow_html=True)
        st.markdown(f"<div class='pill'><div class='label'>%1RM</div><div class='value'>{_fmt_pct(rr['%1RM'])}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pill'><div class='label'>Reps</div><div class='value'>{_fmt_int(rr['Reps'])}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='pill'><div class='label'>Doel-kg</div><div class='value'>{_fmt_kg(rr['TargetKg'])}</div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Export ----------------
session_export = plan[["Workout","Variant","Exercise","SetNumber","%1RM","Reps","TargetKg","e1RM"]].copy()
st.download_button(
    "Download geselecteerde sessie (CSV)",
    data=session_export.to_csv(index=False).encode("utf-8"),
    file_name=f"session_{sel_week}_{sel_block}_{sel_workout}.csv",
    mime="text/csv"
)

# Optioneel: volledige periode exporteren (alle weken/blocks) -> één CSV
def export_full(schedule_df, workouts_df, adv_df, ref_tbl) -> bytes:
    out_rows = []
    schedule_df = schedule_df.copy()
    schedule_df["WeekStr"] = schedule_df["ISO_Year"].astype(int).astype(str) + "-W" + schedule_df["ISO_Week"].astype(int).astype(str)
    for wkstr in sorted(schedule_df["WeekStr"].unique().tolist()):
        for _, rr in schedule_df[schedule_df["WeekStr"]==wkstr].iterrows():
            workout = rr["Workout"]
            # expand advice
            advice_exp = _expand_advice_multivals(adv_df[adv_df["Workout"]==workout], workouts_df)
            tmpl = workouts_df[workouts_df["Workout"]==workout][["Workout","Variant","Exercise","Sets"]].copy()
            rows = []
            for r in tmpl.itertuples(index=False):
                ex, var, n = r.Exercise, r.Variant, int(r.Sets)
                sub = advice_exp[(advice_exp["Workout"]==workout) & (advice_exp["Exercise"]==ex)].sort_values("SetNumber")
                if sub.empty:
                    for s in range(1, n+1):
                        rows.append({"Workout":workout,"Variant":var,"Exercise":ex,"SetNumber":s,"%1RM":70.0 if use_defaults else None,"Reps":8 if use_defaults else None})
                else:
                    if len(sub)<n:
                        last=sub.iloc[-1].to_dict()
                        for s in range(len(sub)+1, n+1):
                            last2 = last.copy(); last2["SetNumber"]=s
                            sub = pd.concat([sub, pd.DataFrame([last2])], ignore_index=True)
                    rows += sub.head(n).assign(Variant=var, Workout=workout).to_dict(orient="records")
            sess = pd.DataFrame(rows)
            if sess.empty: continue
            sess = sess.merge(ref_tbl, on="Exercise", how="left")
            sess["e1RM"] = sess.apply(lambda r: _e1rm_choice(r), axis=1)
            def _tg(r): return _target(r["e1RM"], r["%1RM"], r.get("last_nz_weight"))
            sess["TargetKg"] = sess.apply(_tg, axis=1)
            sess["Week"] = wkstr
            sess["Block"] = rr["Block"]
            out_rows.append(sess[["Week","Block","Workout","Variant","Exercise","SetNumber","%1RM","Reps","TargetKg","e1RM"]])
    big = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame()
    return big.to_csv(index=False).encode("utf-8")

full_bytes = export_full(schedule_df, workouts_df, advicep_df, ref_tbl)
st.download_button("Download volledige periode (CSV)", data=full_bytes, file_name="periodized_full_export.csv", mime="text/csv")

# ---------------- Debug ----------------
if show_debug:
    st.subheader("Diagnostiek")
    st.write("- schedule rows:", 0 if schedule_df is None else len(schedule_df))
    st.write("- workouts rows:", 0 if workouts_df is None else len(workouts_df))
    st.write("- advice rows:", 0 if advicep_df is None else len(advicep_df))
    st.write("- template sets (session):", len(plan))
    st.dataframe(plan.head(50))
