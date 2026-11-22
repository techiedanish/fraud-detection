"""Processing engine for the Streamlit app: clean -> leakage-safe features -> score/train -> evaluate. No synthetic data is used unless the user asks for the demo file."""
import numpy as np, pandas as pd, time
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import xgboost as xgb

STD = ['Timestamp','Card_ID','Transaction_Amount','Merchant_Category','Merchant_Country','Card_Country','Distance_From_Home','Device_Fingerprint_Hash','Is_Proxy_IP','Fraud_Flag','Transaction_ID']
REQUIRED = ['Timestamp','Transaction_Amount']
HINTS = {'Timestamp':['timestamp','transactiondt','datetime','date','time','step','trans_date_trans_time'],
 'Card_ID':['card_id','card1','cc_num','nameorig','account','customer','card','client_id','user_id'],
 'Transaction_Amount':['transaction_amount','transactionamt','amount','amt','value'],
 'Merchant_Category':['merchant_category','category','mcc','productcd','type','merchant_type'],
 'Merchant_Country':['merchant_country','addr2','country'],'Card_Country':['card_country','issuer_country'],
 'Distance_From_Home':['distance_from_home','dist1','distance','distance_from_last_transaction'],
 'Device_Fingerprint_Hash':['device_fingerprint_hash','deviceinfo','device_id','device'],
 'Is_Proxy_IP':['is_proxy_ip','proxy','is_proxy'],
 'Fraud_Flag':['fraud_flag','isfraud','is_fraud','class','fraud','label','target'],
 'Transaction_ID':['transaction_id','transactionid','trans_num','id']}
def guess_mapping(cols):
    low={c.lower().replace(' ','_'):c for c in cols}; m={}
    for k,hs in HINTS.items():
        for h in hs:
            if h in low: m[k]=low[h]; break
    return m

def standardize(raw, mapping, time_mode='datetime'):
    log=[]; d=pd.DataFrame(index=raw.index)
    for k,c in mapping.items():
        if c and c in raw.columns: d[k]=raw[c]
    n0=len(d)
    if time_mode=='datetime': d['Timestamp']=pd.to_datetime(d['Timestamp'],errors='coerce',format='mixed')
    else:
        mult=3600 if time_mode=='hours' else 1; d['Timestamp']=pd.Timestamp('2026-01-01')+pd.to_timedelta(pd.to_numeric(d['Timestamp'],errors='coerce')*mult,unit='s')
    bad=d.Timestamp.isna().sum()
    if bad: log.append(('Unparseable timestamps',int(bad),'Dropped','Cannot order or split without time'))
    d=d[d.Timestamp.notna()]
    d['Transaction_Amount']=pd.to_numeric(d['Transaction_Amount'],errors='coerce'); bad=d.Transaction_Amount.isna().sum()
    if bad: log.append(('Non-numeric amounts',int(bad),'Dropped','Amount required')); 
    d=d[d.Transaction_Amount.notna()]
    neg=d.Transaction_Amount<=0
    if neg.sum(): log.append(('Zero / negative amounts',int(neg.sum()),'Quarantined and dropped','Refunds or reversals are not purchase authorisations'))
    d=d[~neg]
    if 'Transaction_ID' in d: 
        dup=d.Transaction_ID.duplicated().sum(); d=d.drop_duplicates('Transaction_ID')
    else:
        dup=d.duplicated().sum(); d=d.drop_duplicates()
    if dup: log.append(('Duplicate records',int(dup),'Dropped (kept first)','Batch replays inflate counts and velocity'))
    if 'Card_ID' not in d: d['Card_ID']=np.arange(len(d)); log.append(('No Card_ID column',len(d),'Each row treated as its own card','History/velocity features will be zero; weaker detection'))
    d['Card_ID']=d.Card_ID.astype(str)
    if 'Merchant_Category' not in d: d['Merchant_Category']='Unknown'
    d['Merchant_Category']=d.Merchant_Category.fillna('Unknown').astype(str)
    for c in ['Merchant_Country','Card_Country']:
        if c not in d: d[c]='NA'
        d[c]=d[c].fillna('NA').astype(str)
    if 'Distance_From_Home' not in d: d['Distance_From_Home']=np.nan
    d['Distance_From_Home']=pd.to_numeric(d.Distance_From_Home,errors='coerce')
    if 'Device_Fingerprint_Hash' not in d: d['Device_Fingerprint_Hash']='MISSING'
    d['Device_Fingerprint_Hash']=d.Device_Fingerprint_Hash.fillna('MISSING').astype(str)
    if 'Is_Proxy_IP' not in d: d['Is_Proxy_IP']=0
    d['Is_Proxy_IP']=pd.to_numeric(d.Is_Proxy_IP,errors='coerce').fillna(0).clip(0,1).astype(int)
    has_label='Fraud_Flag' in d
    if has_label:
        f=d.Fraud_Flag
        if f.dtype==object: f=f.astype(str).str.strip().str.lower().map({'1':1,'0':0,'true':1,'false':0,'yes':1,'no':0,'fraud':1,'legit':0,'legitimate':0})
        d['Fraud_Flag']=pd.to_numeric(f,errors='coerce'); nl=d.Fraud_Flag.isna().sum()
        if nl: log.append(('Unreadable fraud labels',int(nl),'Dropped','Label must be 0/1'))
        d=d[d.Fraud_Flag.notna()]; d['Fraud_Flag']=d.Fraud_Flag.astype(int)
    if 'Transaction_ID' not in d: d['Transaction_ID']=['R%08d'%i for i in range(len(d))]
    d=d.sort_values('Timestamp',kind='stable').reset_index(drop=True)
    d['Day']=((d.Timestamp-d.Timestamp.min().normalize()).dt.days+1)
    return d,log,has_label,n0

def build_features(d, feats, fit_mask=None):
    d=d.copy(); d['ts']=d.Timestamp.astype('int64')//10**9
    d=d.sort_values(['Card_ID','ts','Transaction_ID']).reset_index(drop=True)
    g=d.groupby('Card_ID'); d['Prior_Tx_Count']=g.cumcount()
    d['Hist_Avg_Spend']=(g.Transaction_Amount.cumsum()-d.Transaction_Amount)/d.Prior_Tx_Count.replace(0,np.nan)
    d['Device_New']=((d.groupby(['Card_ID','Device_Fingerprint_Hash']).cumcount()==0)&(d.Prior_Tx_Count>0)&(d.Device_Fingerprint_Hash!='MISSING')).astype(int)
    d['Secs_Since_Last']=g.ts.diff()
    ts=d.ts.values; amt=d.Transaction_Amount.values; v1=np.zeros(len(d)); v24=np.zeros(len(d)); s1=np.zeros(len(d))
    for c,ix in g.indices.items():
        if len(ix)==1: continue
        t=ts[ix]; i0=np.searchsorted(t,t-3600,'left'); i24=np.searchsorted(t,t-86400,'left'); pos=np.arange(len(t)); a=np.concatenate([[0],np.cumsum(amt[ix])])
        v1[ix]=pos-i0; v24[ix]=pos-i24; s1[ix]=a[pos]-a[i0]
    d['Tx_Velocity_1H']=v1; d['Tx_Velocity_24H']=v24; d['Spend_Last_1H']=s1
    d['Cross_Border']=(d.Merchant_Country!=d.Card_Country).astype(int); d['Hour_Of_Day']=d.Timestamp.dt.hour
    d['Hour_Sin']=np.sin(2*np.pi*d.Hour_Of_Day/24); d['Hour_Cos']=np.cos(2*np.pi*d.Hour_Of_Day/24); d['Log_Amount']=np.log1p(d.Transaction_Amount)
    d['Dist_Missing']=d.Distance_From_Home.isna().astype(int); d['Is_First_Tx']=(d.Prior_Tx_Count==0).astype(int)
    fm=np.ones(len(d),bool) if fit_mask is None else fit_mask.reindex(d.Transaction_ID).values if hasattr(fit_mask,'reindex') else fit_mask
    dm=d.loc[fm,'Distance_From_Home'].median(); dm=0.0 if pd.isna(dm) else dm; d['Distance_From_Home']=d.Distance_From_Home.fillna(dm)
    mm=d[fm].groupby('Merchant_Category').Transaction_Amount.median(); d['Hist_Avg_Spend']=d.Hist_Avg_Spend.fillna(d.Merchant_Category.map(mm)).fillna(d.Transaction_Amount.median())
    d['Amount_Ratio']=d.Transaction_Amount/(d.Hist_Avg_Spend+1); d['Log_Dist']=np.log1p(d.Distance_From_Home); d['Log_Secs_Since_Last']=np.log1p(d.Secs_Since_Last.fillna(7*86400))
    X=pd.concat([d,pd.get_dummies(d.Merchant_Category,prefix='MCC').astype(int),pd.get_dummies(d.Card_Country,prefix='CC').astype(int)],axis=1).reindex(columns=feats,fill_value=0)
    d=d.sort_values(['Timestamp','Transaction_ID']).reset_index(drop=True); X=X.loc[d.index] if False else X
    return d,X
def build_features_ordered(d,feats,fit_frac=None):
    """Returns (d_sorted_by_time, X aligned to d). fit_frac: learn imputers on the earliest fraction of rows."""
    fm=None
    if fit_frac: 
        cut=d.Timestamp.quantile(fit_frac); fm=pd.Series((d.Timestamp<=cut).values,index=d.Transaction_ID.values)
    dd,X=build_features(d,feats,fm)
    # build_features sorted X by card order; realign by recomputing with same sort
    d2=d.copy(); d2['ts']=d2.Timestamp.astype('int64')//10**9; d2=d2.sort_values(['Card_ID','ts','Transaction_ID']).reset_index(drop=True)
    X.index=d2.Transaction_ID.values; X=X.loc[dd.Transaction_ID.values].reset_index(drop=True)
    return dd,X

def rec_at(yt,p,prec):
    P,Rc,_=precision_recall_curve(yt,p); ok=Rc[P>=prec]; return float(ok.max()) if len(ok) else 0.0
def evaluate(y,s,a,t1,t2,prm):
    y=np.asarray(y); s=np.asarray(s); a=np.asarray(a); dec=s>t2; stp=(s>=t1)&(s<=t2); fl=dec|stp
    tp=int((fl&(y==1)).sum()); fp=int((fl&(y==0)).sum()); nf=int(y.sum()); CB=prm['chargeback']; C=prm['step_catch']
    saved=float(a[dec&(y==1)].sum()+C*a[stp&(y==1)].sum()); tot=float(a[y==1].sum())
    miss=((~fl)&(y==1))@(a+CB)+(stp&(y==1))@((1-C)*(a+CB)); fd=(dec&(y==0)).sum()*prm['false_decline']
    cost=float(miss+stp.sum()*.4+((stp&(y==0))@a)*.004+fd+dec.sum()*.5)
    out=dict(recall=tp/max(nf,1),fp_to_tp=fp/max(tp,1),precision=tp/max(tp+fp,1),flagged=int(fl.sum()),declined=int(dec.sum()),stepped=int(stp.sum()),tp=tp,fp=fp,fn=nf-tp,tn=int(((~fl)&(y==0)).sum()),
        fraud_dollars=tot,dollars_saved=saved,pct_saved=100*saved/max(tot,1e-9),false_decline_rate=100*(dec&(y==0)).sum()/max((y==0).sum(),1),net_cost=cost,no_model_cost=float((a[y==1]+CB).sum()))
    out['pr_auc']=float(average_precision_score(y,s)) if nf>0 else None; out['roc_auc']=float(roc_auc_score(y,s)) if 0<nf<len(y) else None
    return out
def choose_thresholds(sv,yv,av,prm):
    P,Rc,T=precision_recall_curve(yv,sv); ok=np.where(Rc[:-1]>=prm['recall_target'])[0]; t1=float(T[ok.max()]) if len(ok) else float(T[0])
    grid=np.unique(np.round(np.r_[np.arange(max(t1,0)+.02,.99,.02)],2)); 
    if len(grid)==0: return t1,min(.99,t1+.05)
    best=min(grid,key=lambda t2:evaluate(yv,sv,av,t1,t2,prm)['net_cost']); return t1,float(best)

def train_own(d,X,prm,progress=None):
    n=len(d); i1,i2=int(.70*n),int(.85*n); mtr=np.zeros(n,bool); mva=np.zeros(n,bool); mte=np.zeros(n,bool); mtr[:i1]=True; mva[i1:i2]=True; mte[i2:]=True
    y=d.Fraud_Flag.values
    for nm,m,mn in [('train',mtr,30),('validation',mva,5),('test',mte,5)]:
        if y[m].sum()<mn: raise ValueError(f'Only {int(y[m].sum())} fraud rows in the {nm} period (need at least {mn}). Upload more fraud examples or use the pre-trained model.')
    res={}
    for spw in [5,20,60]:
        m=xgb.XGBClassifier(n_estimators=200,learning_rate=.08,max_depth=5,min_child_weight=5,subsample=.8,colsample_bytree=.8,reg_lambda=5,scale_pos_weight=spw,tree_method='hist',n_jobs=2,random_state=0).fit(X[mtr],y[mtr])
        res[spw]=(average_precision_score(y[mva],m.predict_proba(X[mva])[:,1]),m)
        if progress: progress(spw)
    spw=max(res,key=lambda k:res[k][0]); return res[spw][1],spw,{k:round(v[0],4) for k,v in res.items()},mtr,mva,mte

def latency_ms(model,X):
    one=X.iloc[[0]]; t0=time.time(); [model.predict_proba(one) for _ in range(50)]; return (time.time()-t0)/50*1000
