"""Independent Blake evaluation; aggregate timing repeats within seed."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from experiments.deadline_policy import timed_policy
from experiments.utility_student import prepare
from experiments.transformer_pilot import TinyTransformer,optimizer,update,loss

ARMS={'random':('random',1),'sparse_loss':('loss',16),'sham_loss':('loss_sham',16)}
SEEDS=range(21000,21012)

def summarize(rows):
    outputs={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm];per=[];repeat_diffs=[]
        for seed in SEEDS:
            ss=sorted([r for r in rr if r['seed']==seed],key=lambda r:r['repeat'])
            assert [r['repeat'] for r in ss]==[0,1]
            per.append(float(np.mean([r['final_loss'] for r in ss])))
            repeat_diffs.append([abs(ss[0]['final_loss']-ss[1]['final_loss']),abs(ss[0]['updates']-ss[1]['updates'])])
        outputs[arm]={k:float(np.mean([r[k] for r in rr])) for k in ['final_loss','updates','guided_accepted','actual_seconds','overrun_seconds']}
        outputs[arm].update(per_seed_mean_loss=per,mean_absolute_repeat_difference=np.mean(repeat_diffs,axis=0).tolist())
    comparisons={}
    for control in ['random','sham_loss']:
        d=np.array(outputs['sparse_loss']['per_seed_mean_loss'])-np.array(outputs[control]['per_seed_mean_loss']);bs=np.random.default_rng(4001).choice(d,(10000,12),replace=True).mean(axis=1)
        comparisons['sparse_loss minus '+control]={'difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'role':'primary' if control=='random' else 'secondary'}
    return outputs,comparisons


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    pools=prepare('data/blake.txt');pool=torch.cat(pools[:3]);probe=pools[3][:32];target=pools[5][:32]
    assert len(target)==32 and len(probe)==32
    rows=[];arms=list(ARMS)
    for seed in SEEDS:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);base=TinyTransformer();bo=optimizer(base)
        for _ in range(12):update(base,bo,pool[rng.integers(len(pool),size=4)])
        ids=rng.integers(len(pool),size=(10000,8));choices=rng.integers(8,size=10000);h=hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest()
        with torch.no_grad():initial=float(loss(base,target))
        for repeat in range(2):
            shift=(seed-21000+repeat)%3
            for name in arms[shift:]+arms[:shift]:
                arm,interval=ARMS[name];model,record=timed_policy(base,bo,pool,probe,arm,None,ids,choices,guidance_interval=interval)
                with torch.no_grad():final=float(loss(model,target))
                record.update(seed=seed,repeat=repeat,arm=name,initial_loss=initial,final_loss=final,schedule_sha256=h);rows.append(record)
    outputs,comparisons=summarize(rows)
    assert all(r['last_accepted_seconds']<=1. and not r['schedule_exhausted'] for r in rows)
    files=['experiments/loss_replication.py','experiments/deadline_policy.py','experiments/sequential_policy.py','experiments/utility_student.py','experiments/transformer_pilot.py','reports/40_loss_replication_protocol_ar.md','data/blake.txt']
    result={'outputs':outputs,'comparisons':comparisons,'budget_seconds':1.,'repeats_per_seed':2,'split_unique_chunks':[len(p) for p in pools],'blake_target_consumed':True,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    Path('results/loss-replication.json').write_text(json.dumps(result,indent=2)+'\n');Path('results/loss-replication-records.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
