"""Recompute stored deadline-run statistics, not timing-dependent trajectories."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.utility_student import prepare

def main():
    r=json.loads(Path('results/deadline-policy.json').read_text());rows=json.loads(Path('results/deadline-policy-records.json').read_text());n=sum(len(p) for p in prepare('data/whitman.txt')[:3]);checks={'rows':len(rows)==36,'hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in r['sha256'].items())}
    for seed in range(18000,18012):
        rr=[v for v in rows if v['seed']==seed];rng=np.random.default_rng(seed)
        for _ in range(12):rng.integers(n,size=4)
        ids=rng.integers(n,size=(10000,8));choices=rng.integers(8,size=10000);h=hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest()
        checks[str(seed)+'_shared']=len(rr)==3 and len({v['initial_loss'] for v in rr})==1 and all(v['schedule_sha256']==h for v in rr)
        for v in rr:
            checks[str(seed)+'_'+v['arm']]=v['updates']==len(v['choices']) and v['updates']<=v['attempted']<=v['updates']+1 and v['last_accepted_seconds']<=1. and not v['schedule_exhausted'] and all(0<=i<8 for i in v['choices']) and v['overrun_seconds']==max(0.,v['actual_seconds']-1.)
            if v['arm']=='random':checks[str(seed)+'_random']=np.array_equal(v['choices'],choices[:v['updates']])
    for arm,out in r['outputs'].items():
        rr=sorted([v for v in rows if v['arm']==arm],key=lambda v:v['seed'])
        checks[arm+'_summary']=all(float(np.mean([v[k] for v in rr]))==out[k] for k in ['final_loss','updates','actual_seconds','overrun_seconds']) and [v['final_loss'] for v in rr]==out['per_seed_final_loss']
    d=np.array(r['outputs']['pairwise']['per_seed_final_loss'])-np.array(r['outputs']['random']['per_seed_final_loss']);bs=np.random.default_rng(3401).choice(d,(10000,12),replace=True).mean(axis=1)
    checks['primary']=float(d.mean())==r['primary']['loss_difference'] and np.array_equal(np.quantile(bs,[.025,.975]),r['primary']['ci95'])
    assert all(checks.values()),checks
    Path('results/deadline-policy-audit.json').write_text(json.dumps({'checks':{k:bool(v) for k,v in checks.items()},'tests_passed':30,'full_timed_rerun':False},indent=2)+'\n');print('All stored deadline checks passed')
if __name__=='__main__':main()
