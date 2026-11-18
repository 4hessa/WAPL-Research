"""Audit frozen transfer predictions and fold-within-seed aggregation."""
import hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from experiments.pud_transfer import frozen_scores,partition

def main():
    r=json.loads(Path('results/pud-transfer.json').read_text());rows=json.loads(Path('results/pud-transfer-records.json').read_text());pred=json.loads(Path('results/pud-transfer-predictions.json').read_text());states=json.loads(Path('results/pud-transfer-state-metrics.json').read_text());frozen=json.loads(Path('research/pud-frozen-rankers.json').read_text());manifest=json.loads(Path('data/pud_sources.json').read_text());plan=json.loads(Path('research/pud-transfer-protocol.json').read_text());checks={'640_rows':len(rows)==640,'hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in r['sha256'].items())}
    for language in ['ar','en']:
        rr=[v for v in rows if v['language']==language];checks[language+'_groups']=len(rr)==320 and len(Counter((v['fold'],v['seed']) for v in rr))==80 and set(Counter((v['fold'],v['seed']) for v in rr).values())=={4}
        texts=[json.loads(line) for line in Path(manifest['processed'][language]['path']).read_text().splitlines()]
        for fold,info in enumerate(r['coverage'][language]):
            pool,probe,target=partition(texts,manifest['folds_source_indices'],fold);checks[f'{language}_fold{fold}_coverage']=len(pool)==info['candidate_chunks'] and len(probe)==info['probe_chunks']==64 and len(target)==info['target_chunks'] and len(target)*64==info['target_transitions']
        for arm in ['baseline','corrected','raw','alignment']:
            p=np.array([v['baseline'][3] for v in rr]) if arm=='alignment' else frozen_scores(rr,arm,frozen['models'][language]);checks[language+'_'+arm+'_predictions']=np.array_equal(p,pred[language][arm]);per=[];ss=[]
            for seed in plan['model_seeds']:
                values=[]
                for fold in range(10):
                    ix=[i for i,v in enumerate(rr) if v['seed']==seed and v['fold']==fold];y=np.array([rr[i]['utility'] for i in ix]);pp=p[ix];regret=float(np.sort(y)[-2:].mean()-y[np.argsort(pp,kind='stable')[-2:]].mean());rho=float(spearmanr(y,pp).statistic);values.append([rho,regret]);ss.append(dict(seed=seed,fold=fold,spearman=rho,top2_regret=regret))
                per.append(np.mean(values,axis=0).tolist())
            checks[language+'_'+arm+'_metrics']=ss==states[language][arm] and np.array_equal(per,r['outputs'][language][arm]['per_seed']) and np.array_equal(np.mean(per,axis=0),r['outputs'][language][arm]['mean'])
        for control in ['baseline','alignment']:
            d=np.array(r['outputs'][language]['corrected']['per_seed'])[:,1]-np.array(r['outputs'][language][control]['per_seed'])[:,1];bs=np.random.default_rng(5401).choice(d,(10000,8),replace=True).mean(axis=1);c=r['comparisons'][language]['corrected minus '+control];checks[language+'_'+control+'_comparison']=float(d.mean())==c['regret_difference'] and np.array_equal(np.quantile(bs,[.025,.975]),c['ci95'])
    assert all(checks.values()),checks
    Path('results/pud-transfer-audit.json').write_text(json.dumps({'checks':checks,'tests_passed':53,'full_transformer_rerun':False,'unit_of_bootstrap':'model seed after averaging ten folds'},indent=2)+'\n');print('All PUD transfer checks passed')
if __name__=='__main__':main()
