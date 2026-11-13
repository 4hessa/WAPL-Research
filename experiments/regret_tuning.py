"""Post-hoc grouped tuning by selection regret instead of value MAE."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from experiments.grouped_tuning import GRID,xy,fit_candidate,metric

def state_regret(rows,p):
    values=[]
    for key in sorted({(r['seed'],r['checkpoint']) for r in rows}):
        ix=[i for i,r in enumerate(rows) if (r['seed'],r['checkpoint'])==key];y=np.array([rows[i]['utility'] for i in ix]);pp=p[ix]
        values.append(float(np.sort(y)[-2:].mean()-y[np.argsort(pp,kind='stable')[-2:]].mean()))
    return float(np.mean(values))

def fit_regret(rows,arm):
    fit=[r for r in rows if r['split']=='fit'];x,y=xy(fit,arm);groups=np.array([r['seed'] for r in fit]);folds=[]
    for seed in sorted(set(groups)):
        tr=groups!=seed;va=groups==seed;sc=StandardScaler().fit(x[tr]);held=[r for r in fit if r['seed']==seed];scores=[]
        for alpha in GRID:
            m=Ridge(alpha=alpha).fit(sc.transform(x[tr]),y[tr]);scores.append(state_regret(held,m.predict(sc.transform(x[va]))))
        folds.append({'held_out_seed':int(seed),'train_seeds':sorted(set(int(s) for s in groups[tr])),'regret_by_alpha':scores})
    scores=list(zip(np.mean([f['regret_by_alpha'] for f in folds],axis=0).tolist(),GRID));alpha=min(scores)[1];sc=StandardScaler().fit(x);m=Ridge(alpha=alpha).fit(sc.transform(x),y)
    return sc,m,{'alpha':alpha,'selection_scores':scores,'folds':folds,'final_fit_rows':len(fit)}

def main():
    source=Path('results/corrected-feature-records.json');records=json.loads(source.read_text());results={}
    for lang in ['ar','en']:
        rows=[r.copy() for r in records if r['language']==lang];runs=[]
        configs=[(a,a,None) for a in ['baseline','corrected','raw']]+[(f'noise_{s}','noise',s) for s in range(51000,51016)]
        for name,arm,seed in configs:
            if seed is not None:
                for r,z in zip(rows,np.random.default_rng(seed).normal(size=(len(rows),3))):r['noise']=z.tolist()
            for criterion in ['mae','regret']:
                sc,m,selection=fit_candidate(rows,arm,'grouped') if criterion=='mae' else fit_regret(rows,arm)
                val=[r for r in rows if r['split']=='validate'];x,_=xy(val,arm);p=m.predict(sc.transform(x));runs.append({'name':name,'arm':arm,'criterion':criterion,'selection':selection,'predictions':p.tolist(),'metrics':metric(val,p)})
        val=[r for r in rows if r['split']=='validate'];alignment=metric(val,np.array([r['baseline'][3] for r in val]));comparisons={}
        for label,left,right in [('regret_corrected_minus_baseline',('corrected','regret'),('baseline','regret')),('regret_corrected_minus_mae_corrected',('corrected','regret'),('corrected','mae')),('regret_corrected_minus_alignment',('corrected','regret'),None)]:
            a=next(r['metrics'] for r in runs if (r['name'],r['criterion'])==left);b=alignment if right is None else next(r['metrics'] for r in runs if (r['name'],r['criterion'])==right);d=np.array(a['per_seed'])[:,2]-np.array(b['per_seed'])[:,2];bs=np.random.default_rng(5201).choice(d,(10000,8),replace=True).mean(axis=1)
            comparisons[label]={'regret_difference':float(d.mean()),'exploratory_ci95':np.quantile(bs,[.025,.975]).tolist()}
        noise={criterion:{'median_regret':float(np.median([r['metrics']['mean'][2] for r in runs if r['arm']=='noise' and r['criterion']==criterion])),'p05_p95':np.quantile([r['metrics']['mean'][2] for r in runs if r['arm']=='noise' and r['criterion']==criterion],[.05,.95]).tolist()} for criterion in ['mae','regret']}
        results[lang]={'runs':runs,'alignment':alignment,'comparisons':comparisons,'noise_summary':noise}
    files=[source,Path(__file__),Path('experiments/grouped_tuning.py')];out={'status':'post_hoc_exposed_labels_not_independent_confirmation','metric_order':['mae','spearman','top2_regret'],'results':results,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('results/regret-tuning.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({l:{'comparisons':r['comparisons'],'models':[{'name':v['name'],'criterion':v['criterion'],'alpha':v['selection']['alpha'],'mean':v['metrics']['mean']} for v in r['runs'] if v['arm']!='noise'],'alignment':r['alignment']['mean'],'noise_summary':r['noise_summary']} for l,r in results.items()},indent=2))
if __name__=='__main__':main()
