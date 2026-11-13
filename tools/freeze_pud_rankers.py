"""Freeze existing rankers before any PUD outcome is computed."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.pairwise_logistic import fit_ranker
from experiments.grouped_tuning import xy

def main():
    source=Path('results/corrected-feature-records.json');rows=json.loads(source.read_text());old=json.loads(Path('results/pairwise-logistic.json').read_text());models={};checks={}
    for language in ['ar','en']:
        data=[r for r in rows if r['language']==language];models[language]={}
        for arm in ['baseline','corrected','raw']:
            sc,m,selection=fit_ranker(data,arm);expected=next(r for r in old['results'][language]['runs'] if r['name']==arm);validation=[r for r in data if r['split']=='validate'];x,_=xy(validation,arm)
            checks[language+'_'+arm]=selection['c']==expected['selection']['c'] and np.allclose(m.decision_function(sc.transform(x)),expected['predictions'],rtol=0,atol=1e-12)
            models[language][arm]={'feature_mean':sc.mean_.tolist(),'feature_scale':sc.scale_.tolist(),'coefficient':m.coef_[0].tolist(),'intercept':float(m.intercept_[0]),'c':selection['c'],'feature_order':'baseline five then corrected/raw weight three if present'}
    assert all(checks.values()),checks
    files=[source,Path('results/pairwise-logistic.json'),Path('experiments/pairwise_logistic.py'),Path('experiments/grouped_tuning.py'),Path('experiments/regret_tuning.py'),Path('research/pud-transfer-protocol.json')]
    result={'status':'frozen_without_PUD_model_evaluation','models':models,'historical_prediction_checks':checks,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    Path('research/pud-frozen-rankers.json').write_text(json.dumps(result,indent=2)+'\n');print('Six rankers frozen; predictions match stage 53')
if __name__=='__main__':main()
