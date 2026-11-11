import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from experiments.transformer_pilot import TinyTransformer,optimizer,update
from experiments.transformer_horizon import rollout

def prepare():
 a=np.frombuffer(Path('data/alice.txt').read_bytes(),dtype=np.uint8).astype(np.int64);edges=[0,.3,.4,.55,.65,.8,1.];seen=set();pools=[]
 for lo,hi in zip(edges,edges[1:]):
  part=a[int(len(a)*lo):int(len(a)*hi)];rows=[]
  for i in range(0,len(part)-32,33):
   b=part[i:i+33];key=b.tobytes()
   if key not in seen:seen.add(key);rows.append(b)
  pools.append(torch.tensor(np.array(rows),dtype=torch.long))
 return pools

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();pools=prepare();pool=torch.cat(pools[:3])
 ids=np.random.default_rng(1500).choice(len(pools[3]),64,replace=False);assert not set(ids[:32])&set(ids[32:]);probes={'A':pools[3][ids[:32]],'B':pools[3][ids[32:]]}
 ti=np.random.default_rng(1501).choice(len(pools[5]),32,replace=False);target=pools[5][ti];rows=[];alternative_seconds=0.
 for seed in range(13000,13012):
  torch.manual_seed(seed);rng=np.random.default_rng(seed);m=TinyTransformer();o=optimizer(m);done=0
  def draw(n):return pool[rng.integers(len(pool),size=n)]
  for cp in [12,48]:
   while done<cp:update(m,o,draw(4));done+=1
   follow=[draw(4) for _ in range(15)];others=[[draw(4) for _ in range(15)] for _ in range(2)]
   for i in range(8):
    c=draw(1);truth=rollout(m,o,c,follow,{'target':target});t=time.perf_counter();alt=[rollout(m,o,c,f,probes) for f in others];alternative_seconds+=time.perf_counter()-t
    assert all(alt[0]['1'][p]==alt[1]['1'][p] for p in probes)
    rows.append({'seed':seed,'checkpoint':cp,'candidate':i,'truth':truth,'alternatives':alt})
 out={};comparisons={}
 for panel in probes:
  out[panel]={}
  for arm in ['one_step','matched']:
   per=[]
   for seed in range(13000,13012):
    mm=[]
    for cp in [12,48]:
     rr=[r for r in rows if r['seed']==seed and r['checkpoint']==cp];y=np.array([r['truth']['16']['target'] for r in rr]);p=np.array([r['alternatives'][0]['1'][panel] if arm=='one_step' else np.mean([x['16'][panel] for x in r['alternatives']]) for r in rr]);mm.append([np.mean(abs(y-p)),spearmanr(y,p).statistic,np.sort(y)[-2:].mean()-y[np.argsort(p)[-2:]].mean()])
    per.append(np.mean(mm,axis=0).tolist())
   out[panel][arm]={'mean':np.mean(per,axis=0).tolist(),'per_seed':per}
  d=np.array(out[panel]['matched']['per_seed'])[:,0]-np.array(out[panel]['one_step']['per_seed'])[:,0];bs=np.random.default_rng(1512).choice(d,(10000,12),replace=True).mean(axis=1);comparisons[panel]={'mae_difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
 files=['experiments/horizon_replication.py','experiments/transformer_horizon.py','experiments/transformer_pilot.py','reports/23_replication_protocol_ar.md','data/alice.txt']
 result={'outputs':out,'comparisons':comparisons,'rows':len(rows),'negative_target16':sum(r['truth']['16']['target']<0 for r in rows),'mean_schedule_disagreement':{p:float(np.mean([abs(r['alternatives'][0]['16'][p]-r['alternatives'][1]['16'][p]) for r in rows])) for p in probes},'probe_indices':ids.tolist(),'target_indices':ti.tolist(),'alternative_seconds':alternative_seconds,'seconds':time.perf_counter()-start,'metric_order':['mae','spearman','top2_regret'],'alice_target_consumed':True,'torch':torch.__version__,'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
 Path('results/horizon-replication.json').write_text(json.dumps(result,indent=2));Path('results/replication-records.json').write_text(json.dumps(rows));print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
