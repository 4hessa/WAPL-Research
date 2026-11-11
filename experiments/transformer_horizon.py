import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from experiments.transformer_pilot import TinyTransformer,optimizer,clone,update,vec,loss

def prepare():
 a=np.frombuffer(Path('data/thursday.txt').read_bytes(),dtype=np.uint8).astype(np.int64);edges=[0,.3,.4,.55,.65,.8,1.];seen=set();pools=[]
 for lo,hi in zip(edges,edges[1:]):
  part=a[int(len(a)*lo):int(len(a)*hi)];rows=[]
  for i in range(0,len(part)-32,33):
   b=part[i:i+33];key=b.tobytes()
   if key not in seen:seen.add(key);rows.append(b)
  pools.append(torch.tensor(np.array(rows),dtype=torch.long))
 return pools

def rollout(m,o,c,follow,evals):
 mi,oi=clone(m,o);ms,os=clone(m,o);update(mi,oi,c);out={}
 for h in range(1,len(follow)+2):
  if h in [1,4,16]:
   with torch.no_grad():out[str(h)]={name:(loss(ms,b)-loss(mi,b)).item() for name,b in evals.items()}
  if h<=len(follow):update(mi,oi,follow[h-1]);update(ms,os,follow[h-1])
 return out

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();pools=prepare();pool=torch.cat(pools[:3]);probe=pools[3][:32];target=pools[5][:32];rows=[];alternative_seconds=0.
 for seed in range(12000,12012):
  torch.manual_seed(seed);rng=np.random.default_rng(seed);m=TinyTransformer();o=optimizer(m);done=0
  def draw(n):return pool[rng.integers(len(pool),size=n)]
  for cp in [12,48]:
   while done<cp:update(m,o,draw(4));done+=1
   follow=[draw(4) for _ in range(15)];other=[draw(4) for _ in range(15)]
   for i in range(8):
    c=draw(1);truth=rollout(m,o,c,follow,{'probe':probe,'target':target})
    t=time.perf_counter();alternative=rollout(m,o,c,other,{'probe':probe});alternative_seconds+=time.perf_counter()-t
    assert truth['1']['probe']==alternative['1']['probe']
    rows.append({'seed':seed,'checkpoint':cp,'candidate':i,'shared':truth,'alternative':alternative})
 outputs={};comparisons={}
 for h in ['1','4','16']:
  outputs[h]={}
  for arm in ['one_step','shared','alternative']:
   per=[]
   for seed in range(12000,12012):
    groups=[]
    for cp in [12,48]:
     rr=[r for r in rows if r['seed']==seed and r['checkpoint']==cp];y=np.array([r['shared'][h]['target'] for r in rr]);p=np.array([r['shared']['1']['probe'] if arm=='one_step' else r[arm][h]['probe'] for r in rr]);groups.append([np.mean(abs(p-y)),spearmanr(y,p).statistic,np.sort(y)[-2:].mean()-y[np.argsort(p)[-2:]].mean()])
    per.append(np.mean(groups,axis=0).tolist())
   outputs[h][arm]={'mean':np.mean(per,axis=0).tolist(),'per_seed':per}
  if h!='1':
   for arm in ['shared','alternative']:
    d=np.array(outputs[h][arm]['per_seed'])[:,0]-np.array(outputs[h]['one_step']['per_seed'])[:,0];bs=np.random.default_rng(1412).choice(d,(10000,12),replace=True).mean(axis=1)
    comparisons[h+' '+arm]={'mae_difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
 paths=['experiments/transformer_horizon.py','experiments/transformer_pilot.py','reports/21_horizon_protocol_ar.md','data/thursday.txt']
 result={'outputs':outputs,'comparisons':comparisons,'rows':len(rows),'negative_targets':{h:sum(r['shared'][h]['target']<0 for r in rows) for h in ['1','4','16']},'alternative_seconds':alternative_seconds,'seconds':time.perf_counter()-start,'metric_order':['mae','spearman','top2_regret'],'target_now_consumed':True,'torch':torch.__version__,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
 Path('results/transformer-horizon.json').write_text(json.dumps(result,indent=2));Path('results/horizon-records.json').write_text(json.dumps(rows));print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
