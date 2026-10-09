"""FlyRank reproducible, client-disjoint March-April benchmark.
Run: python run_capstone.py --csv path/to/flyrank_march_april_model_data.csv
Never publish or commit the input CSV or scored individual records.
"""
import argparse,json,pathlib
import numpy as np,pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES=['impressions','clicks','ctr','avg_position','active_days']

def precision_k(y,s,k):
    ids=np.argsort(-np.asarray(s),kind='stable')[:min(k,len(s))]
    return float(np.asarray(y)[ids].mean())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--csv',required=True);ap.add_argument('--out',default='outputs');a=ap.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.csv)
    need=['client_hash_id','content_hash_id','april_impressions','is_future_decline']+FEATURES
    assert set(need).issubset(d),'Missing columns'
    assert not d[need].isna().any().any(),'Input contains missing values'
    assert d[['client_hash_id','content_hash_id']].duplicated().sum()==0,'Duplicate client/page keys'
    assert d.is_future_decline.isin([0,1]).all()
    assert (d.is_future_decline.values==(d.april_impressions.values<0.8*d.impressions.values).astype(int)).all(),'Invalid labels'
    assert (d.impressions>=0).all() and (d.clicks>=0).all()
    split=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=42)
    tr,te=next(split.split(d,groups=d.client_hash_id))
    train,test=d.iloc[tr].copy(),d.iloc[te].copy()
    assert not (set(train.client_hash_id)&set(test.client_hash_id))
    # A deliberately transparent visibility heuristic: higher March impressions = higher exposure to decline.
    baseline=np.log1p(test.impressions.clip(lower=0).to_numpy())
    model=Pipeline([('impute',SimpleImputer(strategy='median')),('rf',RandomForestClassifier(n_estimators=150,min_samples_leaf=10,max_features='sqrt',random_state=42,n_jobs=-1))])
    model.fit(train[FEATURES],train.is_future_decline)
    score=model.predict_proba(test[FEATURES])[:,1]
    y=test.is_future_decline.to_numpy()
    def metrics(s):
        return {'roc_auc':round(float(roc_auc_score(y,s)),6),'average_precision':round(float(average_precision_score(y,s)),6),'precision_at_20':round(precision_k(y,s,20),6),'precision_at_50':round(precision_k(y,s,50),6),'precision_at_100':round(precision_k(y,s,100),6),'precision_at_500':round(precision_k(y,s,500),6)}
    baseline_metrics=metrics(baseline);model_metrics=metrics(score)
    per_client=[]
    for client,g in test.assign(model_score=score,baseline_score=baseline).groupby('client_hash_id'):
        if len(g)>=50:
            per_client.append({'n_pages':len(g),'base_rate':float(g.is_future_decline.mean()),'model_p50':precision_k(g.is_future_decline,g.model_score,50),'baseline_p50':precision_k(g.is_future_decline,g.baseline_score,50)})
    importance=dict(zip(FEATURES,model.named_steps['rf'].feature_importances_.round(5).tolist()))
    report={'dataset':{'n_rows':len(d),'n_clients':int(d.client_hash_id.nunique()),'base_rate':float(d.is_future_decline.mean()),'train_pages':len(train),'test_pages':len(test),'train_clients':int(train.client_hash_id.nunique()),'test_clients':int(test.client_hash_id.nunique()),'test_base_rate':float(y.mean()),'feature_window':'2026-03-01 to 2026-03-31','outcome_window':'2026-04-01 to 2026-04-30'},'baseline_definition':'Rank descending by March log(1+impressions) as a simple visibility-only heuristic','model_definition':'RandomForestClassifier(150 trees, min_samples_leaf=10, max_features=sqrt, random_state=42)','evaluation':'One seeded 75/25 group-disjoint client split; global top-k precision pooled over held-out clients','baseline':baseline_metrics,'model':model_metrics,'per_client_evaluation':{'clients_with_at_least_50_pages':len(per_client),'macro_p50_model':float(np.mean([r['model_p50'] for r in per_client])) if per_client else None,'macro_p50_baseline':float(np.mean([r['baseline_p50'] for r in per_client])) if per_client else None},'feature_importance':importance,'caveats':['Observational proxy label does not establish refresh lift.','April INNER JOIN excluded pages absent from April observed data, risking survivorship bias.','Single random group split and one month pair do not establish temporal or client generalization.','Global precision@50 can be dominated by large clients; macro client metrics also reported.','Simple visibility baseline is not the earlier Week 4 refresh heuristic and not the previous starter dataset baseline.']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report
if __name__=='__main__':main()
