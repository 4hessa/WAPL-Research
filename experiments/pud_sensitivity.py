"""Post-hoc descriptive fold/seed sensitivity; does not replace primary inference."""
import hashlib,json
from pathlib import Path
import numpy as np


def metric_matrix(states,seeds,folds):
    lookup={}
    for row in states:
        key=(row['seed'],row['fold'])
        if key in lookup:raise ValueError('Duplicate state key')
        lookup[key]=row['top2_regret']
    if set(lookup)!={(s,f) for s in seeds for f in folds}:raise ValueError('Incomplete state grid')
    return np.array([[lookup[(s,f)] for f in folds] for s in seeds])


def sensitivity(delta):
    return {'mean_difference':float(delta.mean()),'per_seed_mean':delta.mean(axis=1).tolist(),'per_fold_mean':delta.mean(axis=0).tolist(),'leave_one_seed_out':[float(np.delete(delta,i,axis=0).mean()) for i in range(delta.shape[0])],'leave_one_fold_out':[float(np.delete(delta,i,axis=1).mean()) for i in range(delta.shape[1])],'state_counts':{'corrected_better':int((delta<0).sum()),'corrected_worse':int((delta>0).sum()),'exact_tie':int((delta==0).sum())}}


def main():
    paths=[Path('results/pud-transfer-state-metrics.json'),Path('results/pud-transfer-records.json'),Path('results/pud-transfer-predictions.json'),Path('results/pud-transfer.json')];states,records,pred,original=[json.loads(p.read_text()) for p in paths];seeds=list(range(27000,27008));folds=list(range(10));results={}
    for language in ['ar','en']:
        delta=metric_matrix(states[language]['corrected'],seeds,folds)-metric_matrix(states[language]['baseline'],seeds,folds);result=sensitivity(delta)
        assert np.isclose(result['mean_difference'],original['comparisons'][language]['corrected minus baseline']['regret_difference'],rtol=0,atol=1e-18)
        rr=[r for r in records if r['language']==language];choices=[]
        for seed in seeds:
            for fold in folds:
                ix=[i for i,r in enumerate(rr) if r['seed']==seed and r['fold']==fold];assert len(ix)==4
                selected={arm:tuple(sorted(np.argsort(np.array(pred[language][arm])[ix],kind='stable')[-2:].tolist())) for arm in ['baseline','corrected','raw','alignment']}
                choices.append({'seed':seed,'fold':fold,'selected_positions':selected})
        result['choice_comparisons']={arm:{'identical_top2_states':sum(c['selected_positions']['corrected']==c['selected_positions'][arm] for c in choices),'mean_top2_overlap_fraction':float(np.mean([len(set(c['selected_positions']['corrected'])&set(c['selected_positions'][arm]))/2 for c in choices]))} for arm in ['baseline','raw','alignment']}
        result['choices']=choices;result['delta_by_seed_fold']=delta.tolist();results[language]=result
    files=paths+[Path(__file__)];out={'status':'post_hoc_descriptive_sensitivity_no_new_hypothesis_test','seed_order':seeds,'fold_order':folds,'positive_difference_means_corrected_worse':True,'results':results,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('results/pud-sensitivity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({l:{k:v for k,v in r.items() if k not in ['choices','delta_by_seed_fold']} for l,r in results.items()},indent=2))
if __name__=='__main__':main()
