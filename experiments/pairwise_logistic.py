"""Exploratory direct pairwise logistic ranking on exposed utility records."""
import hashlib,json,warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from experiments.grouped_tuning import xy,metric
from experiments.regret_tuning import state_regret

C_GRID=[.01,.1,1.,10.]

def pairs(rows,x):
    dx=[];labels=[];ties=0
    for key in sorted({(r['seed'],r['checkpoint']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if (r['seed'],r['checkpoint'])==key]
        for pos,i in enumerate(ix):
            for j in ix[pos+1:]:
                difference=rows[i]['utility']-rows[j]['utility']
                if difference==0:ties+=1;continue
                dx.extend([x[i]-x[j],x[j]-x[i]]);label=int(difference>0);labels.extend([label,1-label])
    if not dx:raise ValueError('No non-tied within-state pairs')
    return np.array(dx),np.array(labels),ties

def logistic(x,y,c):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        return LogisticRegression(C=c,fit_intercept=False,solver='lbfgs',max_iter=2000,tol=1e-8).fit(x,y)

def fit_ranker(rows,arm):
    fit=[r for r in rows if r['split']=='fit'];x,_=xy(fit,arm);groups=np.array([r['seed'] for r in fit]);folds=[]
    for seed in sorted(set(groups)):
        tr=groups!=seed;va=groups==seed;sc=StandardScaler().fit(x[tr]);training=[r for r in fit if r['seed']!=seed];held=[r for r in fit if r['seed']==seed];px,py,ties=pairs(training,sc.transform(x[tr]));scores=[]
        for c in C_GRID:
            m=logistic(px,py,c);scores.append(state_regret(held,m.decision_function(sc.transform(x[va]))))
        folds.append({'held_out_seed':int(seed),'train_seeds':sorted(set(int(s) for s in groups[tr])),'directed_pairs':len(py),'omitted_tied_pairs':ties,'regret_by_c':scores})
    selection=list(zip(np.mean([f['regret_by_c'] for f in folds],axis=0).tolist(),C_GRID));c=min(selection)[1];sc=StandardScaler().fit(x);px,py,ties=pairs(fit,sc.transform(x));m=logistic(px,py,c)
    return sc,m,{'c':c,'selection_scores':selection,'folds':folds,'final_fit_rows':len(fit),'directed_pairs':len(py),'omitted_tied_pairs':ties,'iterations':int(m.n_iter_[0])}

def main():
    source=Path('results/corrected-feature-records.json');rows_all=json.loads(source.read_text());previous=json.loads(Path('results/regret-tuning.json').read_text());results={}
    for lang in ['ar','en']:
        rows=[r.copy() for r in rows_all if r['language']==lang];runs=[]
        configs=[(a,a,None) for a in ['baseline','corrected','raw']]+[(f'noise_{s}','noise',s) for s in range(51000,51016)]
        for name,arm,seed in configs:
            if seed is not None:
                for r,z in zip(rows,np.random.default_rng(seed).normal(size=(len(rows),3))):r['noise']=z.tolist()
            sc,m,selection=fit_ranker(rows,arm);val=[r for r in rows if r['split']=='validate'];x,_=xy(val,arm);p=m.decision_function(sc.transform(x));metrics=metric(val,p)
            # Logistic margins are scores, not calibrated utility estimates.
            metrics={'mean':metrics['mean'][1:],'per_seed':[v[1:] for v in metrics['per_seed']]}
            runs.append({'name':name,'arm':arm,'selection':selection,'predictions':p.tolist(),'metrics':metrics})
        a=next(r for r in runs if r['name']=='corrected');b=next(r for r in runs if r['name']=='baseline');old=previous['results'][lang];ridge=next(v for v in old['runs'] if v['name']=='corrected' and v['criterion']=='regret');comparisons={}
        for label,values in [('rank_baseline',np.array(b['metrics']['per_seed'])[:,1]),('ridge_corrected',np.array(ridge['metrics']['per_seed'])[:,2]),('alignment',np.array(old['alignment']['per_seed'])[:,2])]:
            d=np.array(a['metrics']['per_seed'])[:,1]-values;bs=np.random.default_rng(5301).choice(d,(10000,8),replace=True).mean(axis=1);comparisons['corrected_ranker minus '+label]={'regret_difference':float(d.mean()),'exploratory_ci95':np.quantile(bs,[.025,.975]).tolist()}
        noise=[v['metrics']['mean'][1] for v in runs if v['arm']=='noise'];results[lang]={'runs':runs,'comparisons':comparisons,'noise_summary':{'median_regret':float(np.median(noise)),'p05_p95':np.quantile(noise,[.05,.95]).tolist()}}
    files=[source,Path(__file__),Path('experiments/grouped_tuning.py'),Path('experiments/regret_tuning.py'),Path('results/regret-tuning.json')]
    result={'status':'post_hoc_direct_ranking_exposed_labels','metric_order':['spearman','top2_regret'],'results':results,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('results/pairwise-logistic.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({l:{'comparisons':r['comparisons'],'models':[{'name':v['name'],'c':v['selection']['c'],'directed_pairs':v['selection']['directed_pairs'],'mean':v['metrics']['mean']} for v in r['runs'] if v['arm']!='noise'],'noise':r['noise_summary']} for l,r in results.items()},indent=2))
if __name__=='__main__':main()
