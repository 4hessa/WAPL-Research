"""Post-hoc tuning audit: fixed final fit set, grouped folds within fit only."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

GRID=[.1,1.,10.,100.]

def xy(rows,arm):
    return np.array([r['baseline']+(r[arm] if arm!='baseline' else []) for r in rows]),np.array([r['utility'] for r in rows])

def fit_candidate(rows,arm,rule):
    fit=[r for r in rows if r['split']=='fit'];x,y=xy(fit,arm);groups=np.array([r['seed'] for r in fit]);scores=[];folds=[]
    if rule=='grouped':
        for seed in sorted(set(groups)):
            tr=groups!=seed;va=groups==seed;sc=StandardScaler().fit(x[tr]);values=[]
            for alpha in GRID:
                m=Ridge(alpha=alpha).fit(sc.transform(x[tr]),y[tr]);values.append(float(np.mean(abs(m.predict(sc.transform(x[va]))-y[va]))))
            folds.append({'held_out_seed':int(seed),'train_seeds':sorted(set(int(v) for v in groups[tr])),'train_rows':int(tr.sum()),'held_out_rows':int(va.sum()),'mae_by_alpha':values})
        scores=list(zip(np.mean([f['mae_by_alpha'] for f in folds],axis=0).tolist(),GRID))
    elif rule=='holdout':
        tune=[r for r in rows if r['split']=='tune'];xd,yd=xy(tune,arm);sc=StandardScaler().fit(x)
        for alpha in GRID:
            m=Ridge(alpha=alpha).fit(sc.transform(x),y);scores.append((float(np.mean(abs(m.predict(sc.transform(xd))-yd))),alpha))
    else:raise ValueError('Unknown tuning rule')
    alpha=min(scores)[1];sc=StandardScaler().fit(x);m=Ridge(alpha=alpha).fit(sc.transform(x),y)
    return sc,m,{'alpha':alpha,'selection_scores':scores,'folds':folds,'final_fit_rows':len(fit)}

def metric(rows,p):
    per=[]
    for seed in range(25200,25208):
        states=[]
        for cp in [16,48]:
            ix=[i for i,r in enumerate(rows) if r['seed']==seed and r['checkpoint']==cp];y=np.array([rows[i]['utility'] for i in ix]);pp=p[ix]
            states.append([float(np.mean(abs(y-pp))),float(spearmanr(y,pp).statistic),float(np.sort(y)[-2:].mean()-y[np.argsort(pp,kind='stable')[-2:]].mean())])
        per.append(np.mean(states,axis=0).tolist())
    return {'mean':np.mean(per,axis=0).tolist(),'per_seed':per}

def main():
    path=Path('results/corrected-feature-records.json');source=json.loads(path.read_text());results={}
    for language in ['ar','en']:
        rows=[r.copy() for r in source if r['language']==language];validation=[r for r in rows if r['split']=='validate'];runs=[]
        configurations=[('baseline','baseline',None),('corrected','corrected',None),('raw','raw',None)]+[(f'noise_{s}','noise',s) for s in range(51000,51016)]
        for name,arm,seed in configurations:
            if seed is not None:
                for r,z in zip(rows,np.random.default_rng(seed).normal(size=(len(rows),3))):r['noise']=z.tolist()
            for rule in ['holdout','grouped']:
                sc,m,selection=fit_candidate(rows,arm,rule);v=[r for r in rows if r['split']=='validate'];x,_=xy(v,arm);p=m.predict(sc.transform(x))
                runs.append({'name':name,'arm':arm,'noise_seed':seed,'rule':rule,'selection':selection,'metrics':metric(v,p),'predictions':p.tolist()})
        comparisons={}
        for rule in ['holdout','grouped']:
            a=next(r for r in runs if r['name']=='corrected' and r['rule']==rule);b=next(r for r in runs if r['name']=='baseline' and r['rule']==rule);d=np.array(a['metrics']['per_seed'])[:,0]-np.array(b['metrics']['per_seed'])[:,0];bs=np.random.default_rng(5101).choice(d,(10000,8),replace=True).mean(axis=1)
            comparisons[rule]={'corrected_minus_baseline_mae':float(d.mean()),'exploratory_ci95':np.quantile(bs,[.025,.975]).tolist()}
        results[language]={'runs':runs,'comparisons':comparisons,'noise_summary':{rule:{'median_mae':float(np.median([r['metrics']['mean'][0] for r in runs if r['arm']=='noise' and r['rule']==rule])),'p05_p95':np.quantile([r['metrics']['mean'][0] for r in runs if r['arm']=='noise' and r['rule']==rule],[.05,.95]).tolist()} for rule in ['holdout','grouped']}}
    files=[path,Path(__file__)];out={'status':'post_hoc_tuning_audit_exposed_labels','metric_order':['mae','spearman','top2_regret'],'noise_seeds':16,'results':results,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('results/grouped-tuning.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({l:{'comparisons':r['comparisons'],'models':[{k:v for k,v in q.items() if k in ['name','rule']}|{'alpha':q['selection']['alpha'],'metrics':q['metrics']['mean']} for q in r['runs'] if q['arm']!='noise'],'noise_summary':r['noise_summary']} for l,r in results.items()},indent=2))
if __name__=='__main__':main()
