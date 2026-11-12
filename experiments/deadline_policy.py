"""One-second deployment windows with transactional update acceptance."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.sequential_policy import score_candidates
from experiments.pairwise_student import fit_rankers
from experiments.utility_student import prepare
from experiments.transformer_pilot import TinyTransformer,optimizer,clone,update,loss


def timed_policy(model,opt,pool,probe,arm,ranker,ids,random_choices,budget=1.,clock=time.perf_counter):
    # Initial independent deployment copy is outside the measured window.
    model,opt=clone(model,opt);accepted=[];attempted=0;last=0.;start=clock()
    for step,indices in enumerate(ids):
        if clock()-start>=budget:break
        attempted+=1
        trial,to=clone(model,opt);candidates=pool[indices]
        choice=int(random_choices[step]) if arm=='random' else int(np.argmax(score_candidates(trial,to,candidates,probe,arm,ranker)))
        update(trial,to,candidates[choice:choice+1]);completed=clock()-start
        if completed>budget:break
        model,opt=trial,to;last=completed;accepted.append(choice)
    actual=clock()-start
    return model,{'updates':len(accepted),'attempted':attempted,'choices':accepted,'last_accepted_seconds':last,'actual_seconds':actual,'overrun_seconds':max(0.,actual-budget),'schedule_exhausted':len(accepted)==len(ids)}


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    t=time.perf_counter();models,selection=fit_rankers(json.loads(Path('results/student-records.json').read_text()));fit=time.perf_counter()-t
    pools=prepare('data/whitman.txt');pool=torch.cat(pools[:3]);probe=pools[3][:32];target=pools[5][:32];arms=['random','alignment','pairwise'];rows=[]
    for seed in range(18000,18012):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);base=TinyTransformer();bo=optimizer(base)
        for _ in range(12):update(base,bo,pool[rng.integers(len(pool),size=4)])
        ids=rng.integers(len(pool),size=(10000,8));random_choices=rng.integers(8,size=10000);schedule_hash=hashlib.sha256(ids.tobytes()+random_choices.tobytes()).hexdigest()
        with torch.no_grad():initial=float(loss(base,target))
        shift=(seed-18000)%3
        for arm in arms[shift:]+arms[:shift]:
            model,record=timed_policy(base,bo,pool,probe,arm,models['pairwise'],ids,random_choices)
            with torch.no_grad():final=float(loss(model,target))
            record.update(seed=seed,arm=arm,initial_loss=initial,final_loss=final,schedule_sha256=schedule_hash);rows.append(record)
    outputs={}
    for arm in arms:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['seed']);outputs[arm]={k:float(np.mean([r[k] for r in rr])) for k in ['final_loss','updates','actual_seconds','overrun_seconds']};outputs[arm]['per_seed_final_loss']=[r['final_loss'] for r in rr]
    d=np.array(outputs['pairwise']['per_seed_final_loss'])-np.array(outputs['random']['per_seed_final_loss']);bs=np.random.default_rng(3401).choice(d,(10000,12),replace=True).mean(axis=1)
    assert all(r['last_accepted_seconds']<=1. and not r['schedule_exhausted'] for r in rows)
    files=['experiments/deadline_policy.py','experiments/sequential_policy.py','experiments/pairwise_student.py','experiments/utility_student.py','experiments/transformer_pilot.py','reports/34_deadline_protocol_ar.md','results/student-records.json','data/whitman.txt']
    result={'outputs':outputs,'primary':{'loss_difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()},'budget_seconds':1.,'fit_seconds':fit,'selection':selection,'whitman_target_consumed':True,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    Path('results/deadline-policy.json').write_text(json.dumps(result,indent=2)+'\n');Path('results/deadline-policy-records.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
