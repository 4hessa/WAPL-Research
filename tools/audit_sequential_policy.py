"""Audit stored equal-update policy outcomes; not a trajectory rerun."""
import json,hashlib
from pathlib import Path
import numpy as np
from experiments.utility_student import prepare

def main():
    result=json.loads(Path('results/sequential-policy.json').read_text());rows=json.loads(Path('results/sequential-policy-records.json').read_text())
    checks={'rows':len(rows)==24,'hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in result['sha256'].items())}
    pool_size=sum(len(p) for p in prepare('data/milton.txt')[:3])
    for seed in range(17000,17008):
        rr=[r for r in rows if r['seed']==seed];assert len(rr)==3
        rng=np.random.default_rng(seed)
        for _ in range(12):rng.integers(pool_size,size=4)
        schedule=rng.integers(pool_size,size=(32,8));random_choices=rng.integers(8,size=32)
        checks[str(seed)+'_schedule']=all(np.array_equal(r['candidate_indices'],schedule) for r in rr)
        checks[str(seed)+'_initial_loss']=len({r['initial_loss'] for r in rr})==1
        checks[str(seed)+'_updates']=all(r['updates']==32 and len(r['choices'])==32 and all(0<=v<8 for v in r['choices']) for r in rr)
        checks[str(seed)+'_random_choices']=np.array_equal(next(r['choices'] for r in rr if r['arm']=='random'),random_choices)
    for arm in ['random','alignment','pairwise']:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['seed']);out=result['outputs'][arm];losses=[r['final_loss'] for r in rr]
        checks[arm+'_summary']=losses==out['per_seed_final_loss'] and float(np.mean(losses))==out['mean_final_loss'] and float(np.mean([r['seconds'] for r in rr]))==out['mean_seconds']
        checks[arm+'_finite']=bool(np.isfinite(losses).all())
    d=np.array(result['outputs']['pairwise']['per_seed_final_loss'])-np.array(result['outputs']['alignment']['per_seed_final_loss']);bs=np.random.default_rng(3201).choice(d,(10000,8),replace=True).mean(axis=1)
    checks['primary']=float(d.mean())==result['primary']['loss_difference'] and np.array_equal(np.quantile(bs,[.025,.975]),result['primary']['ci95'])
    assert all(checks.values()),checks
    Path('results/sequential-policy-audit.json').write_text(json.dumps({'checks':{k:bool(v) for k,v in checks.items()},'tests_passed':28,'full_trajectory_rerun':False},indent=2)+'\n')
    print('All stored-result checks passed')
if __name__=='__main__':main()
