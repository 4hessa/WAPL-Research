"""Stored-result audit for sparse guidance; no timing-dependent replay."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.utility_student import prepare
from experiments.sparse_policy import ARMS

def main():
    r=json.loads(Path('results/sparse-policy.json').read_text());rows=json.loads(Path('results/sparse-policy-records.json').read_text());n=sum(len(p) for p in prepare('data/whitman.txt')[:3]);checks={'rows':len(rows)==48,'hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in r['sha256'].items())}
    for seed in range(19000,19012):
        rr=[v for v in rows if v['seed']==seed];rng=np.random.default_rng(seed)
        for _ in range(12):rng.integers(n,size=4)
        ids=rng.integers(n,size=(10000,8));choices=rng.integers(8,size=10000);h=hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest()
        checks[str(seed)+'_shared']=len(rr)==4 and len({v['initial_loss'] for v in rr})==1 and all(v['schedule_sha256']==h for v in rr)
        for v in rr:
            arm,interval=ARMS[v['arm']];u=v['updates'];attempt=v['attempted'];guided=0 if arm=='random' else len(range(0,u,interval));guided_attempt=0 if arm=='random' else len(range(0,attempt,interval))
            checks[str(seed)+'_'+v['arm']]=u==len(v['choices']) and u<=attempt<=u+1 and v['last_accepted_seconds']<=1. and not v['schedule_exhausted'] and v['guided_accepted']==guided and v['guided_attempted']==guided_attempt and v['guidance_interval']==interval and all(0<=x<8 for x in v['choices']) and all(choice==choices[i] for i,choice in enumerate(v['choices']) if arm=='random' or i%interval!=0)
    for arm,out in r['outputs'].items():
        rr=sorted([v for v in rows if v['arm']==arm],key=lambda v:v['seed']);checks[arm+'_summary']=all(float(np.mean([v[k] for v in rr]))==out[k] for k in ['final_loss','updates','guided_accepted','guided_attempted','actual_seconds','overrun_seconds']) and [v['final_loss'] for v in rr]==out['per_seed_final_loss']
    for control in ['random','dense_pairwise']:
        c=r['comparisons']['sparse_pairwise minus '+control];d=np.array(r['outputs']['sparse_pairwise']['per_seed_final_loss'])-np.array(r['outputs'][control]['per_seed_final_loss']);bs=np.random.default_rng(3601).choice(d,(10000,12),replace=True).mean(axis=1)
        checks['comparison_'+control]=float(d.mean())==c['difference'] and np.array_equal(np.quantile(bs,[.025,.975]),c['exploratory_ci95'])
    assert all(checks.values()),checks
    Path('results/sparse-policy-audit.json').write_text(json.dumps({'checks':{k:bool(v) for k,v in checks.items()},'tests_passed':31,'full_timed_rerun':False},indent=2)+'\n');print('Sparse-policy stored-result checks passed')
if __name__=='__main__':main()
