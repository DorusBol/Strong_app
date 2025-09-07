# app.py — Periodized planner with weekly schedule, multi-% advice, and per-session KG export
import io, math
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Next Workout — Periodized", layout="wide")
st.title("Next Workout — Periodized Coaching (Kaartenweergave)")
st.caption("Upload Strong CSV + Periodized Excel (tabs: schedule, workouts, advice_periodization). "
           "Filter op Week/Block/Workout; KG wordt uit je Strong-data berekend.")

# -------- Uploads & Settings --------
c1, c2 = st.columns(2)
with c1: strong_file = st.file_uploader("Strong CSV", type=["csv"])
with c2: plan_xlsx = st.file_uploader("Periodized Excel (schedule/workouts/advice_periodization)", type=["xlsx"])

with st.expander("Instellingen", expanded=True):
    s1, s2, s3, s4 = st.columns(4)
    with s1: kg_step = st.number_input("Rondingsstap (kg)", 0.25, 5.0, 1.0, step=0.25)
    with s2: round_mode = st.selectbox("Rondingsmodus", ["nearest","down","up"], index=0)
    with s3: safety_delta = st.number_input("Veiligheidsmarge op %1RM (±%)", value=0.0, step=1.0, format="%.1f")
    with s4: lookback_weeks = st.number_input("Lookback e1RM (weken)", 1, 52, 8, step=1)
    s5, s6 = st.columns(2)
    with s5: use_defaults = st.checkbox("Gebruik defaults als advies ontbreekt (70% × 8)", True)
    with s6: show_debug = st.checkbox("Toon diagnostiek", False)

# -------- Helpers --------
def _norm(s): 
    if pd.isna(s): return ""
    return " ".join(str(s).strip().lower().split())
def _safe_int(x):
    try:
        if x is None or (isinstance(x,float) and np.isnan(x)) or (isinstance(x,str) and x.strip()==""): return None
        return int(float(x))
    except: return None
def _safe_float(x):
    try:
        if x is None or (isinstance(x,float) and np.isnan(x)) or (isinstance(x,str) and x.strip()==""): return None
        return float(x)
    except: return None
def _coerce_pct(val):
    if val is None or (isinstance(val,float) and np.isnan(val)): return None
    try: return float(str(val).replace("%","").strip())
    except: return None
def _fmt_pct(x): return "" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.0f}%"
def _fmt_int(x): return "" if x is None else f"{int(x)}"
def _fmt_kg(x):  return "" if x is None or (isinstance(x,float) and np.isnan(x)) else f"{x:.1f} kg"
def _round_weight(kg, step, mode):
    if step<=0: return round(kg,1)
    q=kg/step
    if mode=="up":   return round(math.ceil(q)*step,3)
    if mode=="down": return round(math.floor(q)*step,3)
    return round(round(q)*step,3)
def _epley_1rm(w, r): return w*(1.0+r/30.0)

def _isocalendar_tuple(ts: pd.Timestamp)->Tuple[int,int]:
    iso = ts.isocalendar(); return int(iso.year), int(iso.week)

def _read_strong_csv(file)->pd.DataFrame:
    df = pd.read_csv(file)
    req={"Date","Exercise Name","Weight","Reps"}
    miss=req-set(df.columns)
    if miss: st.error(f"Strong CSV mist kolommen: {sorted(miss)}"); st.stop()
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    df["Exercise"]=df["Exercise Name"].astype(str)
    df["Weight"]=pd.to_numeric(df["Weight"],errors="coerce")
    df["Reps"]=pd.to_numeric(df["Reps"],errors="coerce")
    df["ISO_Year"], df["ISO_Week"] = zip(*df["Date"].map(_isocalendar_tuple))
    return df

def _read_sheet(bytes_, name)->Optional[pd.DataFrame]:
    if not bytes_: return None
    try: return pd.read_excel(io.BytesIO(bytes_), sheet_name=name, engine="openpyxl")
    except ValueError: return None

def _compute_refs(df: pd.DataFrame, weeks:int)->pd.DataFrame:
    def e1tbl(sub):
        sub=sub.dropna(subset=["Weight","Reps"]); sub=sub[(sub["Weight"]>0)&(sub["Reps"]>0)]
        if sub.empty: return pd.DataFrame(columns=["Exercise","e1RM"])
        sub=sub.copy(); sub["est_1rm"]=_epley_1rm(sub["Weight"].astype(float), sub["Reps"].astype(int))
        return sub.groupby("Exercise",as_index=False)["est_1rm"].max().rename(columns={"est_1rm":"e1RM"})
    last= df["Date"].max(); cutoff= last - pd.Timedelta(weeks=int(weeks))
    e_recent=e1tbl(df[df["Date"]>=cutoff]).rename(columns={"e1RM":"e1RM_recent"})
    e_all=e1tbl(df).rename(columns={"e1RM":"e1RM_all"})
    nz=df[df["Weight"]>0].sort_values("Date").groupby("Exercise",as_index=False).tail(1)[["Exercise","Weight"]]
    nz= nz.rename(columns={"Weight":"last_nz_weight"})
    return e_recent.merge(e_all, on="Exercise", how="outer").merge(nz, on="Exercise", how="outer")

def _e1rm_choice(row):
    er=_safe_float(row.get("e1RM_recent")); ea=_safe_float(row.get("e1RM_all"))
    return er if (er is not None and not np.isnan(er)) else ea

def _expand_advice_multivals(adf: pd.DataFrame, workouts_df: pd.DataFrame)->pd.DataFrame:
    """
    advice_periodization: columns Workout, Exercise, Pcts, Reps, SetsOverride
    Expand to per-set rows based on either SetsOverride or workouts_df.Sets.
    """
    rows=[]
    # fallback sets lookup
    sets_map = workouts_df.groupby(["Workout","Exercise"])["Sets"].max().to_dict()
    for r in adf.itertuples(index=False):
        w=str(getattr(r,"Workout")); ex=str(getattr(r,"Exercise"))
        pcts=[p.strip() for p in str(getattr(r,"Pcts") or "").split(",") if str(p).strip()!=""]
        reps=[p.strip() for p in str(getattr(r,"Reps") or "").split(",") if str(p).strip()!=""]
        so=_safe_int(getattr(r,"SetsOverride"))
        n = so if so else _safe_int(sets_map.get((w,ex)))
        if n is None: n = max(len(pcts), len(reps)) or 3
        # extend lists to length n
        if not pcts: pcts=["70"]*n
        if not reps: reps=["8"]*n
        if len(pcts)<n: pcts = pcts + [pcts[-1]]*(n-len(pcts))
        if len(reps)<n: reps = reps + [reps[-1]]*(n-len(reps))
        for i in range(n):
            rows.append({"Workout":w,"Exercise":ex,"SetNumber":i+1,"%1RM":_coerce_pct(pcts[i]),"Reps":_safe_int(reps[i])})
    return pd.DataFrame(rows)

# -------- Pipeline --------
if strong_file is None:
    st.info("Upload je Strong CSV."); st.stop()

df_strong = _read_strong_csv(strong_file)
ref_tbl = _compute_refs(df_strong, lookback_weeks)

plan_bytes = plan_xlsx.read() if plan_xlsx else None
schedule_df = _read_sheet(plan_bytes, "schedule")
workouts_df = _read_sheet(plan_bytes, "workouts")
advicep_df  = _read_sheet(plan_bytes, "advice_periodization")

if schedule_df is None or workouts_df is None or advicep_df is None:
    st.error("Excel moet sheets bevatten: 'schedule', 'workouts', 'advice_periodization'."); st.stop()

# Filters
schedule_df["WeekStr"] = schedule_df["ISO_Year"].astype(int).astype(str)+"-W"+schedule_df["ISO_Week"].astype(int).astype(str)
weeks_sorted = sorted(schedule_df["WeekStr"].unique().tolist())
fc = st.columns([1,1,1])
sel_week = fc[0].selectbox("Week (ISO)", weeks_sorted, index=0)
blocks = schedule_df.loc[schedule_df["WeekStr"]==sel_week,"Block"].tolist()
blocks = [b for b in [f"{i}A" for i in range(1,9)]] + [b for b in [f"{i}B" for i in range(1,9)]]  # order hint
blocks = [b for b in blocks if b in schedule_df.loc[schedule_df["WeekStr"]==sel_week,"Block"].unique().tolist()] or sorted(schedule_df.loc[schedule_df["WeekStr"]==sel_week,"Block"].unique().tolist())
sel_block = fc[1].selectbox("Block", blocks, index=0)
wk = schedule_df[(schedule_df["WeekStr"]==sel_week)&(schedule_df["Block"]==sel_block)]["Workout"].iloc[0]
sel_workout = fc[2].selectbox("Workout", [wk] + sorted(schedule_df["Workout"].unique().tolist()), index=0)

# Expand advice for this workout
advice_expanded = _expand_advice_multivals(advicep_df[advicep_df["Workout"]==sel_workout], workouts_df)
# Template for this session (only exercises belonging to the workout library)
template = workouts_df[workouts_df["Workout"]==sel_workout][["Workout","Exercise","Sets"]].copy()
# Merge to get set numbers and parameters
rows=[]
for r in template.itertuples(index=False):
    w, ex, n = r.Workout, r.Exercise, int(r.Sets)
    sub = advice_expanded[(advice_expanded["Workout"]==w)&(advice_expanded["Exercise"]==ex)].copy()
    if sub.empty:
        # build defaults
        for s in range(1, n+1):
            rows.append({"Workout":w,"Exercise":ex,"SetNumber":s,"%1RM":70.0 if use_defaults else None,"Reps":8 if use_defaults else None})
    else:
        # ensure n rows
        sub=sub.sort_values("SetNumber")
        if len(sub)<n:
            last=sub.iloc[-1].to_dict()
            for s in range(len(sub)+1, n+1):
                last2=last.copy(); last2["SetNumber"]=s; sub=pd.concat([sub, pd.DataFrame([last2])], ignore_index=True)
        rows += sub.head(n).to_dict(orient="records")

plan = pd.DataFrame(rows)
plan = plan.merge(ref_tbl, on="Exercise", how="left")
# compute KG
def _target(e1rm, pct, last_nz):
    if pct is None: return None
    if e1rm is not None and not np.isnan(e1rm) and e1rm>0: base=e1rm
    elif last_nz is not None and not np.isnan(last_nz) and last_nz>0: base=last_nz
    else: return None
    p = max(0.0, pct + float(safety_delta)) if safety_delta!=0 else pct
    return _round_weight(base*(p/100.0), kg_step, round_mode)

plan["e1RM"] = plan.apply(lambda r: _e1rm_choice(r), axis=1)
plan["TargetKg"] = plan.apply(lambda r: _target(r["e1RM"], r["%1RM"], r.get("last_nz_weight")), axis=1)

# -------- Header metrics --------
mc = st.columns(4)
with mc[0]: st.metric("Oefeningen", plan["Exercise"].nunique())
with mc[1]: st.metric("Set-blokken", len(plan))
with mc[2]: st.metric("Week", sel_week)
with mc[3]: st.metric("Workout", sel_workout)

# -------- Styling --------
CSS="""
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

# -------- Render --------
for ex, grp in plan.groupby("Exercise", sort=False):
    grp=grp.sort_values("SetNumber")
    e1=grp["e1RM"].dropna().max()
    badge=f"(e1RM: {e1:.1f} kg)" if isinstance(e1,(int,float)) and not np.isnan(e1) else "(e1RM: onbekend)"
    st.markdown(f"<div class='section-title'>{ex} <span class='badge'>{badge} · {sel_workout}</span></div>", unsafe_allow_html=True)
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

# -------- Export: full Excel with per-session tabs --------
def export_workbook(plan_by_session: pd.DataFrame, schedule_df: pd.DataFrame, sel_week: str, workout: str) -> bytes:
    # Build a workbook: Overview + one sheet for the selected session
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Weekly overview (what workouts performed this week)
        wk_overview = schedule_df[schedule_df["WeekStr"]==sel_week][["ISO_Year","ISO_Week","Block","Workout"]].copy()
        wk_overview.to_excel(writer, sheet_name="weekly_overview", index=False)
        # Session sheet
        sheet_name = f"{sel_week} {workout}".replace(":","-")[:31]
        plan_by_session[["Exercise","SetNumber","%1RM","Reps","TargetKg","e1RM"]].to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

dl_bytes = export_workbook(plan, schedule_df.assign(WeekStr=schedule_df["WeekStr"]), sel_week, sel_workout)
st.download_button("Download geselecteerde sessie (Excel)", data=dl_bytes, file_name=f"Session_{sel_week}_{sel_workout}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Optional full export (all sessions & weeks)
def export_full(schedule_df, workouts_df, advicep_df, ref_tbl):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Overview per week
        schedule_df[["ISO_Year","ISO_Week","Block","Workout"]].to_excel(writer, sheet_name="schedule", index=False)
        # For each week/block/workout, produce a session sheet
        for wkstr in sorted(schedule_df["WeekStr"].unique().tolist()):
            for _, rr in schedule_df[schedule_df["WeekStr"]==wkstr].iterrows():
                workout = rr["Workout"]
                # expand advice
                advice_exp = _expand_advice_multivals(advicep_df[advicep_df["Workout"]==workout], workouts_df)
                tmpl = workouts_df[workouts_df["Workout"]==workout][["Workout","Exercise","Sets"]].copy()
                rows=[]
                for r in tmpl.itertuples(index=False):
                    ex, n = r.Exercise, int(r.Sets)
                    sub = advice_exp[(advice_exp["Workout"]==workout)&(advice_exp["Exercise"]==ex)].sort_values("SetNumber")
                    if sub.empty:
                        for s in range(1, n+1): rows.append({"Exercise":ex,"SetNumber":s,"%1RM":70.0 if use_defaults else None,"Reps":8 if use_defaults else None})
                    else:
                        if len(sub)<n:
                            last=sub.iloc[-1].to_dict()
                            for s in range(len(sub)+1, n+1):
                                last2=last.copy(); last2["SetNumber"]=s; sub=pd.concat([sub, pd.DataFrame([last2])], ignore_index=True)
                        rows += sub.head(n)[["Exercise","SetNumber","%1RM","Reps"]].to_dict(orient="records")
                sess = pd.DataFrame(rows)
                sess = sess.merge(ref_tbl, on="Exercise", how="left")
                sess["e1RM"] = sess.apply(lambda r: _e1rm_choice(r), axis=1)
                def _t(r): return _target(r["e1RM"], r["%1RM"], r.get("last_nz_weight"))
                sess["TargetKg"] = sess.apply(_t, axis=1)
                sname = (wkstr+" "+workout).replace(":","-")[:31]
                sess[["Exercise","SetNumber","%1RM","Reps","TargetKg","e1RM"]].to_excel(writer, sheet_name=sname, index=False)
    return output.getvalue()

full_bytes = export_full(schedule_df, workouts_df, advicep_df, ref_tbl)
st.download_button("Download volledige periode (Excel)", data=full_bytes, file_name="Periodized_Full_Export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
