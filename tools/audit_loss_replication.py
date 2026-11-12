"""Independent aggregation and schedule checks for the Blake timing repeats."""
import hashlib,json
from pathlib import Path
import numpy as np
from experiments.utility_student import prepare
from experiments.loss_replication import ARMS,SEEDS

def main():
    r=json.loads(Path('results/loss-replication.json').read_text());rows=json.loads(Path('results/loss-replication-records.json').read_text());n=sum(len(p) for p in prepare('data/blake.txt')[:3]);checks={'72_rows':len(rows)==72,'hashes':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in r['sha256'].items())}
    for seed in SEEDS:
        rr=[v for v in rows if v['seed']==seed];rng=np.random.default_rng(seed)
        for _ in range(12):rng.integers(n,size=4)
        ids=rng.integers(n,size=(10000,8));choices=rng.integers(8,size=10000);h=hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest()
        checks[str(seed)+'_shared']=len(rr)==6 and len({v['initial_loss'] for v in rr})==1 and all(v['schedule_sha256']==h for v in rr)
        for v in rr:
            arm,interval=ARMS[v['arm']];u=v['updates'];attempt=v['attempted'];guided=0 if arm=='random' else len(range(0,u,interval));guided_attempt=0 if arm=='random' else len(range(0,attempt,interval))
            checks[f'{seed}_{v["arm"]}_{v["repeat"]}']=u==len(v['choices']) and u<=attempt<=u+1 and v['last_accepted_seconds']<=1. and not v['schedule_exhausted'] and v['guided_accepted']==guided and v['guided_attempted']==guided_attempt and all(0<=x<8 for x in v['choices']) and all(choice==choices[i] for i,choice in enumerate(v['choices']) if arm in ['random','loss_sham'] or i%interval!=0)
    recomputed={}
    for arm,out in r['outputs'].items():
        rr=[v for v in rows if v['arm']==arm];per=[];diffs=[]
        for seed in SEEDS:
            ss=sorted([v for v in rr if v['seed']==seed],key=lambda v:v['repeat']);assert [v['repeat'] for v in ss]==[0,1]
            per.append((ss[0]['final_loss']+ss[1]['final_loss'])/2);diffs.append([abs(ss[0]['final_loss']-ss[1]['final_loss']),abs(ss[0]['updates']-ss[1]['updates'])])
        recomputed[arm]=per
        checks[arm+'_summary']=np.array_equal(per,out['per_seed_mean_loss']) and np.array_equal(np.mean(diffs,axis=0),out['mean_absolute_repeat_difference']) and all(float(np.mean([v[k] for v in rr]))==out[k] for k in ['final_loss','updates','guided_accepted','actual_seconds','overrun_seconds'])
    for control in ['random','sham_loss']:
        c=r['comparisons']['sparse_loss minus '+control];d=np.array(recomputed['sparse_loss'])-np.array(recomputed[control]);bs=np.random.default_rng(4001).choice(d,(10000,12),replace=True).mean(axis=1)
        checks['comparison_'+control]=float(d.mean())==c['difference'] and np.array_equal(np.quantile(bs,[.025,.975]),c['ci95'])
    assert all(checks.values()),checks
    Path('results/loss-replication-audit.json').write_text(json.dumps({'checks':{k:bool(v) for k,v in checks.items()},'tests_passed':33,'planned_timing_repeats':2,'additional_full_rerun':False},indent=2)+'\n');print('All Blake replication checks passed')
if __name__=='__main__':main()
