import json, io, numpy as np, pandas as pd, streamlit as st, xgboost as xgb
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import fraud_engine as fe
st.set_page_config(page_title="Fraud Detection Operations Hub", page_icon="🛡️", layout="wide")
R = json.load(open('data/results.json')); FEATS = R['features']; TSYN = R['thresholds']
MCCS = ['Grocery','Restaurants','Fuel','Retail','Travel','Electronics','Utilities','Jewelry','Crypto','Online_Services']; CTRY = ['US','GB','DE','IN','BR','SG']
NAVY, RED, GREY = '#1f3a5f', '#c0392b', '#8a94a6'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def dfshow(df, **kw):
    """Display a dataframe; works on old and new Streamlit versions."""
    for extra in ({'hide_index': True, 'width': 'stretch'}, {'hide_index': True, 'use_container_width': True}, {'use_container_width': True}, {}):
        try:
            st.dataframe(df, **extra); return
        except TypeError:
            continue
def do_rerun():
    (getattr(st, 'rerun', None) or st.experimental_rerun)()

@st.cache_resource
def pretrained():
    m = xgb.XGBClassifier(); m.load_model('data/fraud_model_xgb.json'); return m

PAGES = ["0 · Upload & Setup","1 · Executive Overview","2 · Segment Breakdown","3 · Detailed EDA","4 · ML Diagnostics & Live Simulator","5 · Action Plan & Investigator Queue"]
page = st.sidebar.radio("Page", PAGES)
res = st.session_state.get('res')
st.sidebar.markdown("---")
if res: st.sidebar.success(f"Analysis ready: {len(res['df']):,} transactions · {res['mode_label']}")
else: st.sidebar.info("Upload a CSV on page 0 to begin.")

def need_data():
    st.info("No analysis yet. Go to **0 · Upload & Setup**, upload your CSV and press **Run analysis**."); st.stop()
def band_of(s, t1, t2): return np.where(s > t2, 'Decline', np.where(s >= t1, 'Step-up', 'Approve'))
def show(fig): st.pyplot(fig); plt.close(fig)

# ---------------------------------------------------------------- 0 · SETUP
if page == PAGES[0]:
    st.title("Upload & Setup")
    st.write("Upload your own transaction file (CSV). The app cleans it, builds leakage-safe card-history features, scores every transaction and produces all five dashboard pages from **your** data.")
    up = st.file_uploader("Transactions CSV", type=['csv'])
    demo = st.checkbox("No file handy? Use the small demo file (synthetic data)", value=False)
    raw = None
    if up is not None: raw = pd.read_csv(up)
    elif demo: raw = pd.read_csv('data/sample_transactions.csv')
    if raw is None:
        st.markdown("**Expected columns** (names can differ; you will map them in the next step). Only the first two are essential:")
        st.table(pd.DataFrame([['Timestamp','Yes','Date-time (or a numeric time offset)'],['Transaction_Amount','Yes','Purchase amount'],['Card_ID','Strongly recommended','Needed for card-history features (velocity, new device, spend ratio)'],['Fraud_Flag','For evaluation / training','0 = legitimate, 1 = fraud. Without it you get scores and queues but no accuracy metrics'],['Merchant_Category, Merchant_Country, Card_Country','Optional','Improves detection'],['Distance_From_Home, Device_Fingerprint_Hash, Is_Proxy_IP','Optional','Improves detection']], columns=['Column','Needed?','Notes']))
        tmpl = pd.read_csv('data/sample_transactions.csv', nrows=200).to_csv(index=False)
        st.download_button("Download a template CSV (first 200 demo rows)", tmpl, "template_transactions.csv", "text/csv")
        st.stop()
    st.success(f"File loaded: {len(raw):,} rows × {raw.shape[1]} columns"); dfshow(raw.head(5))
    if len(raw) > 600_000: st.warning("Large file: processing may take a few minutes.")
    g = fe.guess_mapping(raw.columns); opts = ['(not in my file)'] + list(raw.columns)
    st.subheader("Map your columns")
    mapping = {}; cols = st.columns(3)
    for i, k in enumerate(fe.STD):
        default = opts.index(g[k]) if k in g and g[k] in opts else 0
        mapping[k] = cols[i % 3].selectbox(k + (' *' if k in fe.REQUIRED else ''), opts, index=default, key=f"map_{k}")
    mapping = {k: v for k, v in mapping.items() if v != '(not in my file)'}
    missing = [k for k in fe.REQUIRED if k not in mapping]
    if missing: st.error(f"Map the required column(s): {', '.join(missing)}"); st.stop()
    tcol = raw[mapping['Timestamp']]
    auto = 0 if not pd.api.types.is_numeric_dtype(tcol) else 1
    tm = st.radio("How is the time column stored?", ["Date-time text (e.g. 2026-03-01 14:05:00)", "Numeric seconds since start (e.g. TransactionDT)", "Numeric hours since start (e.g. PaySim step)"], index=auto, horizontal=True)
    tmode = {0: 'datetime', 1: 'seconds', 2: 'hours'}[["Date-time", "Numeric s", "Numeric h"].index(next(x for x in ["Date-time", "Numeric s", "Numeric h"] if tm.startswith(x)))] if False else ('datetime' if tm.startswith("Date") else 'seconds' if "seconds" in tm else 'hours')
    has_label = 'Fraud_Flag' in mapping
    st.subheader("Choose how to score")
    mode = st.radio("Model", ["Use the pre-trained model (trained on synthetic data)", "Train a new model on my data (needs a fraud label column)"], index=0 if not has_label else 1)
    own = mode.startswith("Train")
    if own and not has_label: st.error("Training needs a Fraud_Flag column. Map it above or choose the pre-trained model."); st.stop()
    with st.expander("Cost assumptions and targets (edit to match your business)"):
        c = st.columns(4); cb = c[0].number_input("Chargeback fee ($)", 0.0, 500.0, 25.0); fd = c[1].number_input("False decline cost ($)", 0.0, 500.0, 12.0)
        sc_ = c[2].slider("Step-up catches fraud (%)", 0, 100, 78); rt = c[3].slider("Validation recall target (%)", 50, 99, 88)
    prm = dict(chargeback=cb, false_decline=fd, step_catch=sc_/100, recall_target=rt/100)
    if not own: st.warning("The pre-trained model learned patterns from synthetic data. On real data it may perform poorly; training on your own labelled data is more reliable.")
    if st.button("▶ Run analysis", type="primary"):
        with st.spinner("Cleaning, building features and scoring…"):
            try:
                d, log, hl, n0 = fe.standardize(raw, mapping, tmode)
                if len(d) < 100: raise ValueError("Fewer than 100 usable rows after cleaning.")
                info = {}
                if own:
                    dd, X = fe.build_features_ordered(d, FEATS, 0.70)
                    model, spw, srch, mtr, mva, mte = fe.train_own(dd, X, prm)
                    s = model.predict_proba(X)[:, 1]; y = dd.Fraud_Flag.values; a = dd.Transaction_Amount.values
                    t1, t2 = fe.choose_thresholds(s[mva], y[mva], a[mva], prm); eval_mask = mte
                    info = dict(spw=spw, search=srch, split=f"Train first 70% of time ({mtr.sum():,} rows) · Validate next 15% ({mva.sum():,}) · Test last 15% ({mte.sum():,})")
                    label = "Model trained on your data"; scope = "Metrics are computed on the held-out final 15% of your timeline only."
                else:
                    dd, X = fe.build_features_ordered(d, FEATS); model = pretrained()
                    s = model.predict_proba(X)[:, 1]; t1, t2 = TSYN['step_up'], TSYN['decline']; eval_mask = np.ones(len(dd), bool)
                    label = "Pre-trained model"; scope = "Metrics use all rows (the model was not trained on your file)."
                dd['score'] = s; dd['band'] = band_of(s, t1, t2)
                lat = fe.latency_ms(model, X)
                st.session_state['res'] = dict(df=dd, X=X, model=model, t1=t1, t2=t2, has_label=hl, eval_mask=np.asarray(eval_mask), prm=prm, log=log, mode_label=label, scope=scope, own=own, info=info, latency=lat, n_raw=n0)
                do_rerun()
            except Exception as e:
                st.error(f"Could not complete the analysis: {e}")

# ---------------------------------------------------------------- guards
else:
    if not res: need_data()
    d = res['df']; hl = res['has_label']; em = res['eval_mask']; t1, t2 = res['t1'], res['t2']; prm = res['prm']; ev = d[em]
    if hl: m = fe.evaluate(ev.Fraud_Flag.values, ev.score.values, ev.Transaction_Amount.values, t1, t2, prm)

if page == PAGES[1]:
    st.title("Executive Overview"); st.caption("What is our aggregate status?  " + res['scope'])
    c = st.columns(5); c[0].metric("Transactions analysed", f"{len(d):,}"); c[1].metric("Flagged (step-up + decline)", f"{(d.band!='Approve').sum():,}")
    c[2].metric("Hard declines", f"{(d.band=='Decline').sum():,}"); c[3].metric("Step-ups", f"{(d.band=='Step-up').sum():,}"); c[4].metric("Scoring latency", f"{res['latency']:.1f} ms")
    if hl:
        c = st.columns(5); c[0].metric("Fraud transactions", f"{int(d.Fraud_Flag.sum()):,}"); c[1].metric("Fraud rate", f"{d.Fraud_Flag.mean()*100:.3f}%")
        c[2].metric("Fraud flagged (target 85%)", f"{m['recall']*100:.1f}%", f"{(m['recall']-.85)*100:+.1f} pts"); c[3].metric("False alerts per fraud (limit 5)", f"{m['fp_to_tp']:.2f}"); c[4].metric("Fraud $ saved", f"{m['pct_saved']:.0f}%")
        st.metric("Expected loss: no model → with model", f"${m['no_model_cost']:,.0f} → ${m['net_cost']:,.0f}")
        daily = d.groupby('Day').Fraud_Flag.mean()*1e4; st.subheader("Daily fraud rate (basis points)"); st.bar_chart(daily)
    else:
        st.info("No fraud label column was mapped, so accuracy metrics cannot be computed. You still get scores, triage bands and queues.")
        st.subheader("Flagged transactions per day"); st.bar_chart((d.band != 'Approve').groupby(d.Day).sum())
    st.subheader("Data cleaning log")
    dfshow(pd.DataFrame(res['log'], columns=['Anomaly','Rows','Action','Why']) if res['log'] else pd.DataFrame({'Result': ['No cleaning actions were needed']}))

elif page == PAGES[2]:
    st.title("Segment Breakdown"); st.caption("Where are the acute risks concentrated?")
    agg = dict(txns=('score','size'), flagged=('band', lambda s: int((s!='Approve').sum())), avg_score=('score','mean'))
    if hl: agg.update(frauds=('Fraud_Flag','sum'))
    seg = d.groupby('Merchant_Category').agg(**agg).reset_index()
    seg['flagged_%'] = (100*seg.flagged/seg.txns).round(2)
    if hl:
        seg['fraud_rate_%'] = (100*seg.frauds/seg.txns).round(3); seg['fraud_$'] = [d[(d.Merchant_Category==k)&(d.Fraud_Flag==1)].Transaction_Amount.sum().round(0) for k in seg.Merchant_Category]
    key = 'fraud_rate_%' if hl else 'flagged_%'; seg = seg.sort_values(key, ascending=False)
    a, b = st.columns(2); a.subheader(("Fraud rate" if hl else "Flagged rate") + " by merchant category (%)"); a.bar_chart(seg.set_index('Merchant_Category')[key])
    if hl: b.subheader("Fraud dollars by category ($)"); b.bar_chart(seg.set_index('Merchant_Category')['fraud_$'])
    else: b.subheader("Average score by category"); b.bar_chart(seg.set_index('Merchant_Category')['avg_score'])
    dfshow(seg)
    st.subheader("Risk flags")
    rows = {}
    for nm, col in [('New device','Device_New'),('Cross-border','Cross_Border'),('Proxy IP','Is_Proxy_IP')]:
        g = d.groupby(col)[('Fraud_Flag' if hl else 'score')].mean()*(100 if hl else 1); rows[nm] = [f"{g.get(0, float('nan')):.3f}", f"{g.get(1, float('nan')):.3f}"]
    st.caption("Values are fraud rate (%)" if hl else "Values are average score")
    st.table(pd.DataFrame(rows, index=['Flag = 0','Flag = 1']))
    st.subheader("Data browser")
    c = st.columns(3); cats = sorted(d.Merchant_Category.unique()); sel = c[0].multiselect("Merchant category", cats, default=cats); bands = c[1].multiselect("Triage band", ['Approve','Step-up','Decline'], default=['Step-up','Decline'])
    mn = c[2].slider("Minimum amount", 0.0, float(d.Transaction_Amount.quantile(.999)), 0.0)
    f = d[d.Merchant_Category.isin(sel) & d.band.isin(bands) & (d.Transaction_Amount >= mn)]
    show_cols = [c_ for c_ in ['Transaction_ID','Timestamp','Card_ID','Merchant_Category','Transaction_Amount','Distance_From_Home','Device_New','Tx_Velocity_1H','score','band','Fraud_Flag'] if c_ in f.columns]
    st.write(f"{len(f):,} transactions match"); dfshow(f[show_cols].head(500))
    st.download_button("Download filtered transactions (CSV)", f[show_cols].to_csv(index=False), "filtered_transactions.csv", "text/csv")

elif page == PAGES[3]:
    st.title("Detailed EDA"); st.caption("Why are these patterns occurring?")
    val = 'Fraud_Flag' if hl else 'score'; lab = "Fraud rate (%)" if hl else "Average score"; mult = 100 if hl else 1
    a, b = st.columns(2)
    v = d.groupby(d.Tx_Velocity_1H.clip(0, 5))[val].mean()*mult; f1, ax = plt.subplots(figsize=(5, 3)); ax.bar(v.index.astype(int).astype(str), v.values, color=NAVY); ax.set_title(f"{lab} by card transactions in prior hour"); ax.set_xlabel("Prior 1H transactions"); a.pyplot(f1); plt.close(f1)
    hh = d.groupby('Hour_Of_Day')[val].mean()*mult; f2, ax = plt.subplots(figsize=(5, 3)); ax.bar(hh.index, hh.values, color=NAVY); ax.set_title(f"{lab} by hour of day"); ax.set_xlabel("Hour"); b.pyplot(f2); plt.close(f2)
    a, b = st.columns(2)
    f3, ax = plt.subplots(figsize=(5, 3)); la = np.log10(d.Transaction_Amount.clip(lower=.5))
    if hl:
        ax.hist(la[d.Fraud_Flag == 0], bins=50, density=True, alpha=.55, color=NAVY, label='Legitimate'); ax.hist(la[d.Fraud_Flag == 1], bins=50, density=True, alpha=.65, color=RED, label='Fraud')
    else:
        ax.hist(la, bins=50, density=True, alpha=.6, color=NAVY, label='All'); ax.hist(la[d.band != 'Approve'], bins=50, density=True, alpha=.6, color=RED, label='Flagged')
    ax.set_xticks(range(0, 6)); ax.set_xticklabels(['$1','$10','$100','$1K','$10K','$100K']); ax.legend(frameon=False); ax.set_title("Amount density (log scale)"); a.pyplot(f3); plt.close(f3)
    cols = ['Log_Amount','Amount_Ratio','Log_Dist','Tx_Velocity_1H','Spend_Last_1H','Cross_Border','Device_New','Is_Proxy_IP','Log_Secs_Since_Last', val]
    cc = d[cols].corr().fillna(0); f4, ax = plt.subplots(figsize=(5, 3.6)); ax.imshow(cc, cmap='RdBu_r', vmin=-1, vmax=1); ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([x.replace('_', ' ') for x in cols], rotation=60, ha='right', fontsize=6); ax.set_yticklabels([x.replace('_', ' ') for x in cols], fontsize=6); ax.set_title("Correlation heatmap"); b.pyplot(f4); plt.close(f4)
    st.subheader("Drift over time")
    dm = d.groupby('Day').agg(avg_score=('score', 'mean'), proxy_share=('Is_Proxy_IP', 'mean'), new_device_share=('Device_New', 'mean'))
    st.line_chart(dm); st.caption("Shifts in these shares across days indicate changing behaviour (drift); retrain when they move materially.")

elif page == PAGES[4]:
    st.title("ML Diagnostics & Live Simulator"); st.caption(f"{res['mode_label']} · {res['scope']}")
    if res['own']:
        st.write(res['info']['split']); st.write(f"XGBoost class weight chosen on validation PR-AUC: **{res['info']['spw']}** (search: {res['info']['search']})")
    if hl:
        rows = [['PR-AUC', f"{m['pr_auc']:.4f}"], ['ROC-AUC', f"{m['roc_auc']:.4f}" if m['roc_auc'] else 'n/a'], ['Fraud flagged', f"{m['recall']*100:.1f}%"], ['False alerts per fraud', f"{m['fp_to_tp']:.2f}"], ['Precision of alerts', f"{m['precision']*100:.1f}%"], ['False decline rate', f"{m['false_decline_rate']:.3f}%"], ['Fraud $ saved', f"${m['dollars_saved']:,.0f} ({m['pct_saved']:.0f}%)"], ['Net expected cost', f"${m['net_cost']:,.0f} (no model ${m['no_model_cost']:,.0f})"]]
        st.table(pd.DataFrame(rows, columns=['Metric', 'Value (evaluation slice)']))
        st.subheader("Threshold explorer")
        c = st.columns(3); s1 = c[0].number_input("Step-up threshold", 0.0005, 0.95, float(max(t1, .0005)), 0.001, format="%.4f"); s2 = c[1].slider("Decline threshold", 0.02, 0.99, float(min(max(t2, .02), .99)), 0.01); fdc = c[2].number_input("False decline cost ($)", 1.0, 100.0, float(prm['false_decline']))
        if s2 <= s1: st.error("Decline threshold must be above the step-up threshold.")
        else:
            p2 = dict(prm, false_decline=fdc); mm = fe.evaluate(ev.Fraud_Flag.values, ev.score.values, ev.Transaction_Amount.values, s1, s2, p2); k = st.columns(5)
            k[0].metric("Fraud flagged", f"{mm['recall']*100:.1f}%"); k[1].metric("False alerts per fraud", f"{mm['fp_to_tp']:.2f}"); k[2].metric("Fraud $ saved", f"${mm['dollars_saved']:,.0f}"); k[3].metric("Flagged", f"{mm['flagged']:,}"); k[4].metric("Net cost", f"${mm['net_cost']:,.0f}")
        a, b, c3 = st.columns(3)
        P, Rc, _ = precision_recall_curve(ev.Fraud_Flag.values, ev.score.values); f1, ax = plt.subplots(figsize=(4, 3)); ax.plot(Rc, P, color=NAVY); ax.scatter([m['recall']], [m['precision']], c='k', zorder=5, label='Operating point'); ax.axhline(1/6, c=GREY, ls=':'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision-recall'); ax.legend(frameon=False, fontsize=7); a.pyplot(f1); plt.close(f1)
        imp = pd.Series(res['model'].feature_importances_, index=FEATS).sort_values().tail(10); f2, ax = plt.subplots(figsize=(4, 3)); ax.barh([i.replace('_', ' ') for i in imp.index], imp.values*100, color=NAVY); ax.set_title('Top features (gain %)'); b.pyplot(f2); plt.close(f2)
        q = pd.qcut(ev.score, 10, duplicates='drop'); cal = ev.groupby(q, observed=True).agg(p=('score', 'mean'), o=('Fraud_Flag', 'mean')); f3, ax = plt.subplots(figsize=(4, 3)); mx = max(cal.p.max(), cal.o.max()); ax.plot([0, mx], [0, mx], '--', c=GREY); ax.plot(cal.p, cal.o, marker='o', color=NAVY); ax.set_xlabel('Predicted'); ax.set_ylabel('Observed'); ax.set_title('Calibration (deciles)'); c3.pyplot(f3); plt.close(f3)
    else:
        st.info("No fraud labels: showing score distribution only."); f1, ax = plt.subplots(figsize=(6, 3)); ax.hist(d.score, bins=60, color=NAVY); ax.set_yscale('log'); ax.axvline(t1, c=GREY, ls=':'); ax.axvline(t2, c=RED, ls=':'); ax.set_title('Score distribution (dotted = thresholds)'); st.pyplot(f1); plt.close(f1)
    st.subheader("Transaction simulator")
    c1, c2, c3 = st.columns(3)
    amt = c1.number_input("Amount ($)", 0.5, 100000.0, 250.0); mcc = c1.selectbox("Merchant category", MCCS, index=8); cc = c1.selectbox("Card country", CTRY); mct = c1.selectbox("Merchant country", CTRY, index=1)
    hist = c2.number_input("Card's average past spend ($)", 0.0, 100000.0, 40.0); dist = c2.number_input("Distance from home (km)", 0.0, 20000.0, 600.0); hour = c2.slider("Hour of day", 0, 23, 3); secs = c2.number_input("Seconds since card's last transaction (0 = none)", 0, 10_000_000, 300)
    v1 = c3.number_input("Card transactions in last 1 hour", 0, 50, 3); v24 = c3.number_input("Card transactions in last 24 hours", 0, 200, 4); s1h = c3.number_input("Card spend in last 1 hour ($)", 0.0, 100000.0, 400.0)
    newdev = c3.checkbox("New device for this card", True); proxy = c3.checkbox("Proxy / TOR IP", False); first = c3.checkbox("First transaction on this card", False)
    row = {f: 0.0 for f in FEATS}
    row.update(Log_Amount=np.log1p(amt), Amount_Ratio=amt/(hist+1), Log_Dist=np.log1p(dist), Dist_Missing=0, Tx_Velocity_1H=v1, Tx_Velocity_24H=v24, Spend_Last_1H=s1h, Cross_Border=int(cc != mct), Device_New=int(newdev and not first), Is_First_Tx=int(first), Is_Proxy_IP=int(proxy), Hour_Sin=np.sin(2*np.pi*hour/24), Hour_Cos=np.cos(2*np.pi*hour/24), Log_Secs_Since_Last=np.log1p(secs if secs > 0 else 7*86400))
    row[f'MCC_{mcc}'] = 1; row[f'CC_{cc}'] = 1
    score = float(res['model'].predict_proba(pd.DataFrame([row])[FEATS])[0, 1]); action = "🚫 DECLINE & freeze" if score > t2 else ("🔐 STEP-UP (OTP / 3-D Secure)" if score >= t1 else "✅ APPROVE")
    st.divider(); a, b = st.columns(2); a.metric("Fraud score", f"{score:.3f}"); b.subheader(action); st.caption(f"Using the {res['mode_label'].lower()}; bands: approve < {t1:.4f} · step-up {t1:.4f}–{t2:.2f} · decline > {t2:.2f}.")

elif page == PAGES[5]:
    st.title("Action Plan & Investigator Queue"); st.caption("What specific actions must we execute tomorrow?")
    days = ['All days'] + sorted(d.Day.unique().tolist()); dsel = st.selectbox("Day", days, index=len(days)-1); band = st.multiselect("Band", ['Decline', 'Step-up'], default=['Decline', 'Step-up'])
    day = d if dsel == 'All days' else d[d.Day == dsel]; qd = day[day.band.isin(band)].sort_values('score', ascending=False)
    c = st.columns(3); c[0].metric("Flagged", f"{(day.band!='Approve').sum():,}"); c[1].metric("Tier 1: Decline", f"{(day.band=='Decline').sum():,}"); c[2].metric("Tier 2: Step-up", f"{(day.band=='Step-up').sum():,}")
    n = st.slider("Rows to show", 10, 200, 50); cols = [c_ for c_ in ['Transaction_ID','Timestamp','Card_ID','Merchant_Category','Transaction_Amount','Distance_From_Home','Device_New','Tx_Velocity_1H','score','band','Fraud_Flag'] if c_ in qd.columns]
    dfshow(qd[cols].head(n))
    out = d[[c_ for c_ in cols if c_ in d.columns]].copy(); st.download_button("Download ALL scored transactions (CSV)", out.to_csv(index=False), "scored_transactions.csv", "text/csv")
    st.subheader("Strategic action plan")
    st.table(pd.DataFrame([[f'Deploy 3-band triage: approve < {t1:.4f}, step-up {t1:.4f}–{t2:.2f}, decline > {t2:.2f}','Risk Engine Lead','≥85% fraud flagged; false alerts per fraud < 5'],['Use step-up (OTP / 3-D Secure) for the middle band instead of manual review','Risk Engine Lead','Auth completion > 92%'],['Throttle velocity (≤2 attempts/hour) on the highest-risk categories shown on page 2','Fraud Ops','Chargeback rate < 0.9%'],['Weekly drift review; retrain when recall < 80% or score distribution shifts','ML Lead + Fraud Ops','Recall ≥ 80% weekly'],['Cap the manual queue at investigator capacity','Fraud Ops Manager','Queue age < 4 h']], columns=['Intervention','Owner','KPI']))
    with st.expander("Method, assumptions & limits"):
        st.markdown(f"""
**Pipeline:** clean → leakage-safe card-history features (only earlier transactions) → XGBoost with class weights → three-band triage (approve / step-up / decline).
**Mode used:** {res['mode_label']}. {res['scope']}
**Cost assumptions:** missed fraud = amount + ${prm['chargeback']:.0f}; step-up catches {prm['step_catch']*100:.0f}% of fraud and costs about $0.40; false decline = ${prm['false_decline']:.0f}; $0.50 review per decline.
**Limits:** metrics depend on label quality and on how many fraud rows your file contains; the pre-trained model was learned from synthetic data and may not transfer; scores are decision support, not proof of fraud.
**Prohibited uses:** automatic account closure, credit decisions, populations unlike the training data.""")
