"""Audit schedules and independently aggregate bilingual policy outcomes."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.multilingual_readiness import byte_chunks

def main():
    result=json.loads(Path('results/multilingual-policy.json').read_text());rows=json.loads(Path('results/multilingual-policy-records.json').read_text());checks={'96_rows':len(rows)==96,'hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in result['sha256'].items())}
    for language in ['ar','en']:
        pool=byte_chunks(Path(f'data/ud_{language}_train.txt').read_bytes());test_blob=Path(f'data/ud_{language}_test.txt').read_bytes();target=byte_chunks(test_blob);info=result['data_info'][language]
        checks[language+'_coverage']=info=={'train_chunks':len(pool),'test_chunks':len(target),'test_transitions':len(target)*64,'discarded_test_tail_bytes':len(test_blob)%65}
        for seed in range(23000,23008):
            rr=[r for r in rows if r['language']==language and r['seed']==seed];rng=np.random.default_rng(seed)
            for _ in range(32):rng.integers(len(pool),size=4)
            ids=rng.integers(len(pool),size=(20000,8));choices=rng.integers(8,size=20000);h=hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest()
            checks[f'{language}_{seed}_shared']=len(rr)==6 and len({r['initial_loss'] for r in rr})==1 and all(r['schedule_sha256']==h for r in rr)
            for r in rr:
                u=r['updates'];a=r['attempted'];key=f'{language}_{seed}_{r["mode"]}_{r["arm"]}_{r["repeat"]}'
                checks[key]=u==len(r['choices']) and u<=a<=u+1 and not r['schedule_exhausted'] and all(0<=v<8 for v in r['choices']) and all(v==choices[i] for i,v in enumerate(r['choices']) if r['arm']=='random' or i%16!=0) and (r['last_accepted_seconds']<=3. if r['mode']=='time' else u==256)
        for mode in ['updates','time']:
            per={}
            for arm in ['random','sparse_loss']:
                rr=[r for r in rows if (r['language'],r['mode'],r['arm'])==(language,mode,arm)];means=[]
                for seed in range(23000,23008):
                    ss=sorted([r for r in rr if r['seed']==seed],key=lambda r:r['repeat']);assert [r['repeat'] for r in ss]==([0,1] if mode=='time' else [0]);means.append(sum(r['final_loss'] for r in ss)/len(ss))
                per[arm]=means;out=result['outputs'][language][mode][arm]
                checks[f'{language}_{mode}_{arm}_summary']=means==out['per_seed_mean_loss'] and all(float(np.mean([r[k] for r in rr]))==out[k] for k in ['final_loss','updates','actual_seconds','overrun_seconds'])
            d=np.array(per['sparse_loss'])-np.array(per['random']);bs=np.random.default_rng(4401).choice(d,(10000,8),replace=True).mean(axis=1);c=result['comparisons'][language][mode]
            checks[f'{language}_{mode}_comparison']=float(d.mean())==c['difference'] and np.array_equal(np.quantile(bs,[.025,.975]),c['ci95'])
    assert all(checks.values()),checks
    Path('results/multilingual-policy-audit.json').write_text(json.dumps({'checks':{k:bool(v) for k,v in checks.items()},'tests_passed':40,'planned_time_repeats':2,'additional_full_rerun':False},indent=2)+'\n');print('All bilingual policy checks passed')
if __name__=='__main__':main()
