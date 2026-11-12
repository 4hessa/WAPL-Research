"""Exploratory, post-hoc ranking audit of the exposed Brown evaluation set."""
import hashlib
import json
from pathlib import Path
import numpy as np


def metrics(y, p):
    """Eight candidates per state; offsets do not affect within-state rankings."""
    error = p - y
    centered = error - error.mean()
    target_order = np.argsort(y, kind='stable')[-2:]
    predicted_order = np.argsort(p, kind='stable')[-2:]
    i, j = np.triu_indices(len(y), 1)
    truth = np.sign(y[i] - y[j])
    predicted = np.sign(p[i] - p[j])
    non_tie = truth != 0
    concordance = np.where(predicted == truth, 1., np.where(predicted == 0, .5, 0.))
    return dict(mse=float(np.mean(error**2)), offset_mse=float(error.mean()**2),
                within_mse=float(np.mean(centered**2)),
                pair_accuracy=float(concordance[non_tie].mean()),
                top2_overlap=float(len(set(target_order) & set(predicted_order))/2),
                regret=float(y[target_order].mean()-y[predicted_order].mean()),
                prediction_std=float(p.std()), target_std=float(y.std()))


def main():
    record_path=Path('results/student-records.json')
    prediction_path=Path('results/student-predictions.json')
    rows=[r for r in json.loads(record_path.read_text()) if r['split']=='test']
    predictions=json.loads(prediction_path.read_text())
    groups=sorted({(r['seed'],r['checkpoint']) for r in rows})
    results={}
    for arm, predictions_arm in predictions.items():
        states=[]
        for seed, cp in groups:
            ix=[i for i,r in enumerate(rows) if (r['seed'],r['checkpoint'])==(seed,cp)]
            assert len(ix)==8
            y=np.array([rows[i]['target'] for i in ix]);p=np.array([predictions_arm[i] for i in ix])
            m=metrics(y,p)
            assert np.isclose(m['mse'],m['offset_mse']+m['within_mse'],rtol=1e-12,atol=1e-18)
            states.append(dict(seed=seed,checkpoint=cp,**m))
        keys=list(m)
        results[arm]={'states':states,'mean':{k:float(np.mean([s[k] for s in states])) for k in keys},
                      'by_checkpoint':{str(cp):{k:float(np.mean([s[k] for s in states if s['checkpoint']==cp])) for k in keys} for cp in [12,48]}}
    seeds=sorted({r['seed'] for r in rows});comparisons={}
    for metric in ['regret','pair_accuracy','within_mse']:
        delta=[]
        for seed in seeds:
            a=np.mean([r[metric] for r in results['baseline']['states'] if r['seed']==seed])
            b=np.mean([r[metric] for r in results['calibrated']['states'] if r['seed']==seed])
            delta.append(a-b)
        bs=np.random.default_rng(2701).choice(delta,(10000,len(seeds)),replace=True).mean(axis=1)
        comparisons[metric]={'baseline_minus_calibrated':float(np.mean(delta)), 'exploratory_ci95':np.quantile(bs,[.025,.975]).tolist()}
    files=[record_path,prediction_path,Path(__file__)]
    out={'status':'exploratory_post_hoc_exposed_Brown','groups':len(groups),'seeds':len(seeds),
         'arms':results,'comparisons':comparisons,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('results/ranking-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'means':{k:v['mean'] for k,v in results.items()},'comparisons':comparisons},indent=2))

if __name__=='__main__':main()
