"""Post-hoc noise/regularization audit on fixed, exposed stage-49 records."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

GRID=[.1,1.,10.,100.]

def fit_noise(rows,seed,baseline_alpha):
    # Fresh independent noise, preserving record and split ordering.
    z=np.random.default_rng(seed).normal(size=(len(rows),3))
    ids={s:[i for i,r in enumerate(rows) if r['split']==s] for s in ['fit','tune','validate']}
    x=np.column_stack([np.array([r['baseline'] for r in rows]),z]);y=np.array([r['utility'] for r in rows]);sc=StandardScaler().fit(x[ids['fit']]);x=sc.transform(x);models={};tuning=[]
    for alpha in GRID:
        m=Ridge(alpha=alpha).fit(x[ids['fit']],y[ids['fit']]);models[alpha]=m;tuning.append((float(np.mean(abs(m.predict(x[ids['tune']])-y[ids['tune']]))),alpha))
    chosen=min(tuning)[1]
    return {arm:{'alpha':alpha,'predictions':models[alpha].predict(x[ids['validate']]).tolist()} for arm,alpha in [('tuned',chosen),('fixed',baseline_alpha)]}

def main():
    source=Path('results/corrected-feature-records.json');records=json.loads(source.read_text());old=json.loads(Path('results/corrected-features.json').read_text());results={}
    for language in ['ar','en']:
        rows=[r for r in records if r['language']==language];val=[r for r in rows if r['split']=='validate'];y=np.array([r['utility'] for r in val]);alpha=old['selection'][language]['baseline']['alpha'];runs=[]
        for seed in range(50000,50064):
            arms=fit_noise(rows,seed,alpha)
            for name,v in arms.items():
                p=np.array(v['predictions']);per=[float(np.mean(abs(y[[i for i,r in enumerate(val) if r['seed']==s]]-p[[i for i,r in enumerate(val) if r['seed']==s]]))) for s in range(25200,25208)]
                runs.append({'noise_seed':seed,'arm':name,'alpha':v['alpha'],'mae':float(np.mean(per)),'per_model_seed_mae':per})
        summaries={}
        for name in ['tuned','fixed']:
            rr=[r for r in runs if r['arm']==name];a=np.array([r['mae'] for r in rr]);summaries[name]={'mean':float(a.mean()),'median':float(np.median(a)),'p05_p95':np.quantile(a,[.05,.95]).tolist(),'alpha_counts':dict(Counter(r['alpha'] for r in rr)),'fraction_better_than_baseline':float(np.mean(a<old['outputs'][language]['baseline']['mean'][0])),'fraction_better_than_corrected':float(np.mean(a<old['outputs'][language]['corrected']['mean'][0]))}
        # A diagnostic sweep, not a new selected model: expose every alpha equally.
        tr=[r for r in rows if r['split']=='fit'];dv=[r for r in rows if r['split']=='tune'];sc=StandardScaler().fit([r['baseline'] for r in tr]);grid=[]
        for aa in GRID:
            m=Ridge(alpha=aa).fit(sc.transform([r['baseline'] for r in tr]),[r['utility'] for r in tr]);grid.append({'alpha':aa,'tune_mae':float(np.mean(abs(m.predict(sc.transform([r['baseline'] for r in dv]))-np.array([r['utility'] for r in dv])))),'validation_mae':float(np.mean(abs(m.predict(sc.transform([r['baseline'] for r in val]))-y)))})
        results[language]={'baseline_alpha':alpha,'original_mae':{k:v['mean'][0] for k,v in old['outputs'][language].items()},'noise_summary':summaries,'baseline_grid_diagnostic':grid,'runs':runs}
    files=[source,Path('results/corrected-features.json'),Path(__file__)]
    out={'status':'post_hoc_exploratory_same_labels_no_new_model_training','noise_seeds':64,'model_validation_seeds':8,'percentiles_are_not_confidence_intervals':True,'results':results,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('results/noise-sensitivity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({l:{k:v for k,v in r.items() if k!='runs'} for l,r in results.items()},indent=2))
if __name__=='__main__':main()
