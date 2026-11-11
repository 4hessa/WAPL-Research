"""Exact shared-softmax implementation for the bigram alignment baseline."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from experiments.text_pilot import V,LR,prepare,counts,loss_grad,step

def reference_scores(w,v,cs,probe):
    gp=loss_grad(w,probe)[1]
    return np.array([LR*np.sum(gp*(.9*v+loss_grad(w,c)[1])) for c in cs])

def shared_scores(w,v,cs,probe):
    p=np.exp(w-logsumexp(w,axis=1,keepdims=True))
    gp=(probe.sum(axis=1,keepdims=True)*p-probe)/probe.sum()
    row_inner=(p*gp).sum(axis=1);momentum=np.sum(gp*.9*v)
    scores=np.array([LR*(momentum+(np.dot(c.sum(axis=1),row_inner)-np.sum(c*gp))/c.sum()) for c in cs])
    return scores,p

def run(w,v,pool,probe,ids,ri,arm):
    w,v=w.copy(),v.copy();selected=[];start=time.perf_counter()
    for candidates,random_index in zip(ids,ri):
        if arm=='random':
            k=int(random_index);w,v=step(w,v,counts(pool[candidates[k]]))
        else:
            cs=[counts(pool[i]) for i in candidates]
            if arm=='reference':
                k=int(np.argmax(reference_scores(w,v,cs,probe)));w,v=step(w,v,cs[k])
            else:
                scores,p=shared_scores(w,v,cs,probe);k=int(np.argmax(scores));c=cs[k]
                g=(c.sum(axis=1,keepdims=True)*p-c)/c.sum();v=.9*v+g;w=w-LR*v
        selected.append(int(candidates[k]))
    return w,v,selected,time.perf_counter()-start

def main():
    pools,_=prepare(Path('data/emma.txt').read_text());pool=pools[0]+pools[1]+pools[2]
    probe=sum((counts(s) for s in pools[3]),np.zeros((V,V)))
    records=[];ratios=[];overheads=[]
    for seed in range(8000,8008):
        rng=np.random.default_rng(seed);w=rng.normal(0,.01,(V,V));v=np.zeros_like(w)
        for _ in range(24):w,v=step(w,v,counts(pool[rng.integers(len(pool))]))
        ids=rng.integers(len(pool),size=(96,8));ri=rng.integers(8,size=96);arms=['random','reference','shared'];times={a:[] for a in arms};errors=[];selections=None
        for rep in range(3):
            order=(seed+rep)%3;out={}
            for arm in arms[order:]+arms[:order]:
                wa,va,selected,t=run(w,v,pool,probe,ids,ri,arm);out[arm]=(wa,va,selected);times[arm].append(t)
            assert out['reference'][2]==out['shared'][2]
            err=max(np.max(abs(out['reference'][0]-out['shared'][0])),np.max(abs(out['reference'][1]-out['shared'][1])))
            assert err<=1e-10;errors.append(float(err));selections=out['shared'][2]
        med={a:float(np.median(t)) for a,t in times.items()};ratios.append(med['reference']/med['shared']);overheads.append(med['shared']/med['random'])
        records.append({'seed':seed,'seconds':times,'median_seconds':med,'max_state_errors':errors,'selected_ids':selections,'selected_ids_match':True})
    files=['experiments/reuse.py','experiments/text_pilot.py','reports/13_reuse_protocol_ar.md','data/emma.txt']
    r={'runs':records,'mean_reference_over_shared':float(np.mean(ratios)),'mean_shared_over_random':float(np.mean(overheads)),'max_state_error':max(max(r['max_state_errors']) for r in records),'all_choices_match':True,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},'final_holdout_unused':True}
    Path('results/reuse.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k not in ['runs','sha256']})
if __name__=='__main__':main()
