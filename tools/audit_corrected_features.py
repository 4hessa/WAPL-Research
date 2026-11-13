"""Audit feature fitting and validation statistics without retraining trajectories."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from experiments.corrected_features import fit_models,design
from experiments.multilingual_readiness import byte_chunks

def main():
    result=json.loads(Path('results/corrected-features.json').read_text());rows=json.loads(Path('results/corrected-feature-records.json').read_text());pred=json.loads(Path('results/corrected-feature-predictions.json').read_text());checks={'320_rows':len(rows)==320,'hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in result['sha256'].items())}
    for language in ['ar','en']:
        rr=[r for r in rows if r['language']==language];checks[language+'_splits']=Counter(r['split'] for r in rr)=={'fit':64,'tune':32,'validate':64}
        checks[language+'_groups']=set(Counter((r['split'],r['seed'],r['checkpoint']) for r in rr).values())=={4}
        dev=byte_chunks(Path(f'data/ud_{language}_dev.txt').read_bytes());expected=np.random.default_rng(4800).choice(len(dev),128,replace=False)
        checks[language+'_dev_indices']=np.array_equal(expected,result['dev_indices'][language])
        models,selection=fit_models(rr);checks[language+'_selection']=json.loads(json.dumps(selection))==result['selection'][language];vr=[r for r in rr if r['split']=='validate'];y=np.array([r['utility'] for r in vr])
        for arm in ['baseline','corrected','raw','noise','alignment']:
            if arm=='alignment':p=np.array([r['baseline'][3] for r in vr])
            else:sc,m=models[arm];p=m.predict(sc.transform(design(vr,arm)))
            checks[f'{language}_{arm}_predictions']=np.array_equal(p,pred[language][arm]);per=[]
            for seed in range(25200,25208):
                states=[]
                for cp in [16,48]:
                    ix=[i for i,r in enumerate(vr) if r['seed']==seed and r['checkpoint']==cp];yy=y[ix];pp=p[ix]
                    states.append([float(np.mean(abs(yy-pp))),float(spearmanr(yy,pp).statistic),float(np.sort(yy)[-2:].mean()-yy[np.argsort(pp,kind='stable')[-2:]].mean())])
                per.append(np.mean(states,axis=0).tolist())
            out=result['outputs'][language][arm];checks[f'{language}_{arm}_metrics']=np.array_equal(per,out['per_seed']) and np.array_equal(np.mean(per,axis=0),out['mean'])
        for control in ['baseline','raw','noise']:
            out=result['outputs'][language];d=np.array(out['corrected']['per_seed'])[:,0]-np.array(out[control]['per_seed'])[:,0];bs=np.random.default_rng(4802).choice(d,(10000,8),replace=True).mean(axis=1);c=result['comparisons'][language]['corrected minus '+control]
            checks[f'{language}_{control}_comparison']=float(d.mean())==c['mae_difference'] and np.array_equal(np.quantile(bs,[.025,.975]),c['exploratory_ci95'])
    checks['no_test_files']=not any('_test.txt' in p for p in result['sha256'])
    assert all(checks.values()),checks
    Path('results/corrected-features-audit.json').write_text(json.dumps({'checks':{k:bool(v) for k,v in checks.items()},'tests_passed':43,'full_trajectory_rerun':False},indent=2)+'\n');print('Corrected-feature fit and metric checks passed')
if __name__=='__main__':main()
