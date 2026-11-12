"""Exploratory batched-loss guidance on the already exposed Whitman target."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.deadline_policy import timed_policy
from experiments.pairwise_student import fit_rankers
from experiments.utility_student import prepare
from experiments.transformer_pilot import TinyTransformer,optimizer,update,loss

ARMS={'random':('random',1),'dense_loss':('loss',1),'sparse_loss':('loss',16),'sparse_pairwise':('pairwise',16)}

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models,selection=fit_rankers(json.loads(Path('results/student-records.json').read_text()))
    pools=prepare('data/whitman.txt');pool=torch.cat(pools[:3]);probe=pools[3][:32];target=pools[5][:32];arms=list(ARMS);rows=[]
    for seed in range(20000,20012):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);base=TinyTransformer();bo=optimizer(base)
        for _ in range(12):update(base,bo,pool[rng.integers(len(pool),size=4)])
        ids=rng.integers(len(pool),size=(10000,8));choices=rng.integers(8,size=10000);schedule_hash=hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest()
        with torch.no_grad():initial=float(loss(base,target))
        shift=(seed-20000)%4
        for name in arms[shift:]+arms[:shift]:
            arm,interval=ARMS[name];model,record=timed_policy(base,bo,pool,probe,arm,models['pairwise'],ids,choices,guidance_interval=interval)
            with torch.no_grad():final=float(loss(model,target))
            record.update(seed=seed,arm=name,initial_loss=initial,final_loss=final,schedule_sha256=schedule_hash);rows.append(record)
    outputs={}
    for arm in arms:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['seed'])
        outputs[arm]={k:float(np.mean([r[k] for r in rr])) for k in ['final_loss','updates','guided_accepted','guided_attempted','actual_seconds','overrun_seconds']};outputs[arm]['per_seed_final_loss']=[r['final_loss'] for r in rr]
    comparisons={}
    for control in ['random','sparse_pairwise']:
        d=np.array(outputs['sparse_loss']['per_seed_final_loss'])-np.array(outputs[control]['per_seed_final_loss']);bs=np.random.default_rng(3801).choice(d,(10000,12),replace=True).mean(axis=1)
        comparisons['sparse_loss minus '+control]={'difference':float(d.mean()),'exploratory_ci95':np.quantile(bs,[.025,.975]).tolist()}
    assert all(r['last_accepted_seconds']<=1. and not r['schedule_exhausted'] for r in rows)
    files=['experiments/loss_proxy_policy.py','experiments/deadline_policy.py','experiments/sequential_policy.py','experiments/pairwise_student.py','experiments/utility_student.py','experiments/transformer_pilot.py','reports/38_loss_proxy_protocol_ar.md','results/student-records.json','data/whitman.txt']
    result={'status':'exploratory_exposed_Whitman','outputs':outputs,'comparisons':comparisons,'budget_seconds':1.,'selection':selection,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    Path('results/loss-proxy-policy.json').write_text(json.dumps(result,indent=2)+'\n');Path('results/loss-proxy-policy-records.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
