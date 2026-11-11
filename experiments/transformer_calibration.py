import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from experiments.transformer_pilot import TinyTransformer,optimizer,clone,update,vec,loss,prepare

def signals(m,o,c,probe):
    mi,oi=clone(m,o)
    oi.zero_grad();initial=loss(mi,probe);initial.backward();pg=vec(mi,True);w=vec(mi)
    update(mi,oi,c);delta=vec(mi)-w
    t=time.perf_counter()
    with torch.no_grad():direct=(initial.detach()-loss(mi,probe)).item()
    return -delta.dot(pg).item(),direct,mi,oi,time.perf_counter()-t

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();pools=prepare();pool=torch.cat(pools[:3]);probe=pools[3][:32];target=pools[5][:32];rows=[];extra_seconds=0.
    for seed in range(11000,11012):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);m=TinyTransformer();o=optimizer(m);done=0
        def draw(n):return pool[rng.integers(len(pool),size=n)]
        for cp in [12,48]:
            while done<cp:update(m,o,draw(4));done+=1
            follow=[draw(4) for _ in range(3)]
            for i in range(8):
                c=draw(1);a,p,mi,oi,t=signals(m,o,c,probe);extra_seconds+=t;ms,os=clone(m,o)
                with torch.no_grad():t1=(loss(ms,target)-loss(mi,target)).item()
                for b in follow:update(mi,oi,b);update(ms,os,b)
                with torch.no_grad():t4=(loss(ms,target)-loss(mi,target)).item()
                rows.append({'seed':seed,'checkpoint':cp,'candidate':i,'alignment':a,'direct':p,'target1':t1,'target4':t4,'components':[a-p,p-t1,t1-t4]})
    out={};comparisons={}
    for h in [1,4]:
        out[str(h)]={}
        for arm in ['alignment','direct']:
            per=[]
            for seed in range(11000,11012):
                metrics=[]
                for cp in [12,48]:
                    rr=[r for r in rows if r['seed']==seed and r['checkpoint']==cp];y=np.array([r['target'+str(h)] for r in rr]);p=np.array([r[arm] for r in rr]);metrics.append([np.mean(abs(p-y)),spearmanr(y,p).statistic,np.sort(y)[-2:].mean()-y[np.argsort(p)[-2:]].mean()])
                per.append(np.mean(metrics,axis=0).tolist())
            out[str(h)][arm]={'mean':np.mean(per,axis=0).tolist(),'per_seed':per}
        d=np.array(out[str(h)]['direct']['per_seed'])[:,0]-np.array(out[str(h)]['alignment']['per_seed'])[:,0];bs=np.random.default_rng(1312).choice(d,(10000,12),replace=True).mean(axis=1)
        comparisons[str(h)]={'direct_minus_alignment_mae':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
    files=['experiments/transformer_calibration.py','experiments/transformer_pilot.py','reports/19_calibration_protocol_ar.md','data/persuasion.txt']
    result={'outputs':out,'comparisons':comparisons,'mean_absolute_components':np.mean(np.abs([r['components'] for r in rows]),axis=0).tolist(),'component_order':['local_approximation','probe_target_difference','horizon_drift'],'rows':len(rows),'negative_target1':sum(r['target1']<0 for r in rows),'negative_target4':sum(r['target4']<0 for r in rows),'extra_probe_forward_seconds':extra_seconds,'seconds':time.perf_counter()-start,'metric_order':['mae','spearman','top2_regret'],'persuasion_final_consumed':True,'torch':torch.__version__,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    Path('results/transformer-calibration.json').write_text(json.dumps(result,indent=2));Path('results/calibration-records.json').write_text(json.dumps(rows));print(json.dumps({k:v for k,v in result.items() if k not in ['outputs','sha256']},indent=2))
if __name__=='__main__':main()
