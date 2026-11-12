"""Equal-update sequential policy test; deliberately not equal-compute."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from experiments.pairwise_student import fit_rankers
from experiments.utility_student import prepare
from experiments.transformer_pilot import TinyTransformer,optimizer,clone,update,vec,loss


def score_candidates(model,opt,candidates,probe,arm,ranker=None):
    # Compute probe gradient on a clone, preserving all original state and gradients.
    pm,po=clone(model,opt);po.zero_grad();loss(pm,probe).backward();pg=vec(pm,True);w=vec(model)
    features=[]
    for candidate in candidates:
        cm,co=clone(model,opt);co.zero_grad();l=loss(cm,candidate[None]);l.backward();g=vec(cm,True);co.step();delta=vec(cm)-w
        if arm=='alignment':features.append([-delta.dot(pg).item()])
        else:features.append([l.item(),g.norm().item(),g.dot(pg).item(),-delta.dot(pg).item(),delta.norm().item()])
    x=np.array(features)
    if arm=='alignment':return x[:,0]
    scaler,regressor=ranker
    return regressor.predict(scaler.transform(x))


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    models,selection=fit_rankers(json.loads(Path('results/student-records.json').read_text()))
    slope=float(models['calibrated'][1].coef_[0]);assert slope>0
    pools=prepare('data/milton.txt');pool=torch.cat(pools[:3]);probe=pools[3][:32];target=pools[5][:32]
    rows=[];arms=['random','alignment','pairwise']
    for seed in range(17000,17008):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);base=TinyTransformer();bo=optimizer(base)
        for _ in range(12):update(base,bo,pool[rng.integers(len(pool),size=4)])
        schedule=rng.integers(len(pool),size=(32,8));random_choices=rng.integers(8,size=32)
        with torch.no_grad():initial=float(loss(base,target))
        shift=(seed-17000)%3
        for arm in arms[shift:]+arms[:shift]:
            model,opt=clone(base,bo);choices=[];start=time.perf_counter()
            for step,indices in enumerate(schedule):
                candidates=pool[indices]
                choice=int(random_choices[step]) if arm=='random' else int(np.argmax(score_candidates(model,opt,candidates,probe,arm,models['pairwise'])))
                choices.append(choice);update(model,opt,candidates[choice:choice+1])
            seconds=time.perf_counter()-start
            with torch.no_grad():final=float(loss(model,target))
            rows.append(dict(seed=seed,arm=arm,initial_loss=initial,final_loss=final,seconds=seconds,choices=choices,candidate_indices=schedule.tolist(),updates=32))
    outputs={}
    for arm in arms:
        rr=sorted([r for r in rows if r['arm']==arm],key=lambda r:r['seed'])
        outputs[arm]={'mean_final_loss':float(np.mean([r['final_loss'] for r in rr])),'mean_seconds':float(np.mean([r['seconds'] for r in rr])),'per_seed_final_loss':[r['final_loss'] for r in rr]}
    delta=np.array(outputs['pairwise']['per_seed_final_loss'])-np.array(outputs['alignment']['per_seed_final_loss'])
    bs=np.random.default_rng(3201).choice(delta,(10000,8),replace=True).mean(axis=1)
    files=['experiments/sequential_policy.py','experiments/pairwise_student.py','experiments/utility_student.py','experiments/transformer_pilot.py','reports/32_sequential_protocol_ar.md','results/student-records.json','data/milton.txt']
    result={'outputs':outputs,'primary':{'loss_difference':float(delta.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()},'selection':selection,'calibration_slope':slope,'milton_target_consumed':True,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    Path('results/sequential-policy.json').write_text(json.dumps(result,indent=2)+'\n');Path('results/sequential-policy-records.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
