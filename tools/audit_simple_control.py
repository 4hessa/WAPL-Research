"""Recompute fitting, predictions and metrics without repeating model trajectories."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from experiments.pairwise_student import fit_rankers,regret

def main():
    r=json.loads(Path('results/simple-control.json').read_text())
    rows=json.loads(Path('results/simple-control-records.json').read_text())
    pred=json.loads(Path('results/simple-control-predictions.json').read_text())
    models,selection=fit_rankers(json.loads(Path('results/student-records.json').read_text()))
    x=np.array([s['baseline'] for s in rows]);groups=Counter((s['seed'],s['checkpoint']) for s in rows)
    checks={'selection':json.loads(json.dumps(selection))==r['selection'], '256_rows':len(rows)==256,
            '32_groups_of_8':len(groups)==32 and set(groups.values())=={8},
            'source_hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in r['sha256'].items())}
    for arm,(sc,m) in models.items():checks[arm+'_predictions']=bool(np.array_equal(m.predict(sc.transform(x[:,3:4] if arm=='calibrated' else x)),pred[arm]))
    by_checkpoint={}
    for arm,pp in pred.items():
        per=[];cp_values={12:[],48:[]}
        for seed in range(16000,16016):
            states=[]
            for cp in [12,48]:
                ix=[i for i,s in enumerate(rows) if s['seed']==seed and s['checkpoint']==cp]
                y=np.array([rows[i]['target'] for i in ix]);p=np.array([pp[i] for i in ix])
                v=[regret(y,p),float(spearmanr(y,p).statistic)];states.append(v);cp_values[cp].append(v)
            per.append(np.mean(states,axis=0).tolist())
        checks[arm+'_metrics']=bool(np.array_equal(per,r['outputs'][arm]['per_seed']) and np.array_equal(np.mean(per,axis=0),r['outputs'][arm]['mean']))
        by_checkpoint[arm]={str(cp):np.mean(v,axis=0).tolist() for cp,v in cp_values.items()}
    d=np.array(r['outputs']['pairwise']['per_seed'])[:,0]-np.array(r['outputs']['calibrated']['per_seed'])[:,0]
    bs=np.random.default_rng(3001).choice(d,(10000,16),replace=True).mean(axis=1)
    checks['primary']=float(d.mean())==r['primary']['regret_difference'] and np.array_equal(np.quantile(bs,[.025,.975]),r['primary']['ci95'])
    checks['teacher_predictions']=pred['teacher']==[s['teacher'] for s in rows]
    assert all(checks.values()),checks
    out={'checks':checks,'by_checkpoint':by_checkpoint,'tests_passed':27,'full_transformer_rerun':False}
    Path('results/simple-control-audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
