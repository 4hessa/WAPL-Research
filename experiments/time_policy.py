"""Fixed-deadline deployment diagnostic on new text; see protocol 08."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from experiments.text_pilot import V,LR,prepare,counts,loss_grad,step
from experiments.policy import choose,fit_predictors
BUDGET=.5

def timed_run(w,v,pool,probe,arm,models,ids,random_choices,budget=BUDGET,clock=time.perf_counter):
    w,v=w.copy(),v.copy();selected=[];accepted_time=0.;attempted=0
    start=clock()
    for ix,ri in zip(ids,random_choices):
        if clock()-start>=budget:break
        attempted+=1
        if arm=='random':k=int(ri);c=counts(pool[ix[k]])
        else:
            cs=[counts(pool[i]) for i in ix]
            if arm=='alignment':
                pg=loss_grad(w,probe)[1]
                scores=[float(np.sum(pg*(LR*(.9*v+loss_grad(w,c)[1])))) for c in cs]
                k=int(np.argmax(scores))
            else:k=choose(w,v,cs,probe,arm,models,int(ri))
            c=cs[k]
        wn,vn=step(w,v,c);completed=clock()-start
        if completed>budget:break
        w,v=wn,vn;selected.append(int(ix[k]));accepted_time=completed
    elapsed=clock()-start
    return w,{'updates':len(selected),'transitions':len(selected)*64,'selected_ids':selected,'attempted':attempted,'last_completed_seconds':accepted_time,'actual_seconds':elapsed,'overrun_seconds':max(0.,elapsed-budget),'schedule_exhausted':len(selected)==len(ids)}

def main():
    started=time.perf_counter();pools,_=prepare(Path('data/moby_dick.txt').read_text())
    pool=pools[0]+pools[1]+pools[2];probe=sum((counts(s) for s in pools[3]),np.zeros((V,V)));final=sum((counts(s) for s in pools[5]),np.zeros((V,V)))
    t=time.perf_counter();models,alphas=fit_predictors(json.loads(Path('results/centered-records.json').read_text()));fit_seconds=time.perf_counter()-t
    # Warm up numerical routines outside the measured deployment windows.
    z=np.zeros((V,V));c=counts(pool[0]);step(z,z,c)
    for a in ['baseline','centered']:choose(z,z,[c]*8,probe,a,models,0)
    arms=['random','alignment','baseline','centered'];runs=[]
    for seed in range(5000,5012):
        rng=np.random.default_rng(seed);w=rng.normal(0,.01,(V,V));v=np.zeros_like(w)
        for _ in range(24):w,v=step(w,v,counts(pool[rng.integers(len(pool))]))
        ids=rng.integers(len(pool),size=(20000,8));choices=rng.integers(8,size=20000)
        schedule_hash=hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest();initial=loss_grad(w,final)[0]
        for arm in arms[seed%4:]+arms[:seed%4]:
            wn,record=timed_run(w,v,pool,probe,arm,models,ids,choices)
            record.update({'seed':seed,'arm':arm,'initial_loss':initial,'final_loss':loss_grad(wn,final)[0],'schedule_sha256':schedule_hash});runs.append(record)
    summary={a:{'final_loss':float(np.mean([r['final_loss'] for r in runs if r['arm']==a])),'mean_updates':float(np.mean([r['updates'] for r in runs if r['arm']==a])),'mean_actual_seconds':float(np.mean([r['actual_seconds'] for r in runs if r['arm']==a])),'max_overrun_seconds':max(r['overrun_seconds'] for r in runs if r['arm']==a)} for a in arms}
    comparisons={}
    for a,b in [('centered','random'),('centered','baseline'),('alignment','random'),('baseline','random')]:
        d=np.array([next(r['final_loss'] for r in runs if r['arm']==a and r['seed']==s)-next(r['final_loss'] for r in runs if r['arm']==b and r['seed']==s) for s in range(5000,5012)])
        bs=np.random.default_rng(912).choice(d,(10000,len(d)),replace=True).mean(axis=1)
        comparisons[a+' minus '+b]={'difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
    paths=['experiments/time_policy.py','experiments/policy.py','experiments/text_pilot.py','experiments/invariance.py','reports/08_time_protocol_ar.md','data/moby_dick.txt','results/centered-records.json']
    result={'summary':summary,'comparisons':comparisons,'runs':runs,'budget_seconds':BUDGET,'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-started,'alphas':alphas,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'moby_final_partition_consumed':True}
    assert all(r['last_completed_seconds']<=BUDGET and not r['schedule_exhausted'] for r in runs)
    Path('results/time-policy.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
