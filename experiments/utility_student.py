import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,LinearRegression
from experiments.transformer_pilot import TinyTransformer,optimizer,clone,update,vec,loss
from experiments.transformer_horizon import rollout

def prepare(path):
 a=np.frombuffer(Path(path).read_bytes(),dtype=np.uint8).astype(np.int64);edges=[0,.3,.4,.55,.65,.8,1.];seen=set();out=[]
 for lo,hi in zip(edges,edges[1:]):
  part=a[int(len(a)*lo):int(len(a)*hi)];rows=[]
  for i in range(0,len(part)-32,33):
   b=part[i:i+33];k=b.tobytes()
   if k not in seen:seen.add(k);rows.append(b)
  out.append(torch.tensor(np.array(rows),dtype=torch.long))
 return out

def fit_models(rows):
 def xy(split,arm):
  rr=[r for r in rows if r['split']==split];x=np.array([r['baseline']+(r['weight'] if arm=='weight' else []) for r in rr]);return (x[:,3:4] if arm=='calibrated' else x),np.array([r['teacher'] for r in rr])
 models={};alphas={}
 for arm in ['calibrated','baseline','weight']:
  x,y=xy('train',arm);xd,yd=xy('dev',arm);sc=StandardScaler().fit(x)
  if arm=='calibrated':m=LinearRegression().fit(sc.transform(x),y);a=None
  else:
   scores=[]
   for a in [.1,1.,10.,100.]:
    m=Ridge(alpha=a).fit(sc.transform(x),y);scores.append((np.mean(abs(m.predict(sc.transform(xd))-yd)),a))
   a=min(scores)[1];m=Ridge(alpha=a).fit(sc.transform(x),y)
  models[arm]=(sc,m);alphas[arm]=a
 return models,alphas

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();alice=prepare('data/alice.txt');brown=prepare('data/brown.txt');rows=[];timing={'features':0.,'teacher':0.,'target':0.}
 for split,seeds in [('train',range(14000,14008)),('dev',range(14100,14104)),('test',range(14200,14208))]:
  pools=brown if split=='test' else alice;pool=torch.cat(pools[:3]) if split=='test' else pools[0 if split=='train' else 1];probe=pools[3][:32];target=pools[5][:32] if split=='test' else None
  for seed in seeds:
   torch.manual_seed(seed);rng=np.random.default_rng(seed);m=TinyTransformer();o=optimizer(m);done=0
   def draw(n):return pool[rng.integers(len(pool),size=n)]
   for cp in [12,48]:
    while done<cp:update(m,o,draw(4));done+=1
    teach_follow=[draw(4) for _ in range(15)];true_follow=[draw(4) for _ in range(15)] if split=='test' else None
    t=time.perf_counter();o.zero_grad();loss(m,probe).backward();pg=vec(m,True);w=vec(m);timing['features']+=time.perf_counter()-t
    for i in range(8):
     c=draw(1);t=time.perf_counter();mi,oi=clone(m,o);oi.zero_grad();l=loss(mi,c);l.backward();g=vec(mi,True);oi.step();d=vec(mi)-w
     b=[l.item(),g.norm().item(),g.dot(pg).item(),-d.dot(pg).item(),d.norm().item()];e=[w.dot(g).item(),(w*g).norm().item(),(d.norm()/(w.norm()+1e-12)).item()];timing['features']+=time.perf_counter()-t
     t=time.perf_counter();teacher=rollout(m,o,c,teach_follow,{'probe':probe})['16']['probe'];timing['teacher']+=time.perf_counter()-t
     y=None
     if split=='test':
      t=time.perf_counter();y=rollout(m,o,c,true_follow,{'target':target})['16']['target'];timing['target']+=time.perf_counter()-t
     rows.append({'split':split,'seed':seed,'checkpoint':cp,'candidate':i,'baseline':b,'weight':e,'teacher':teacher,'target':y})
 t=time.perf_counter();models,alphas=fit_models(rows);timing['fit']=time.perf_counter()-t;test=[r for r in rows if r['split']=='test'];y=np.array([r['target'] for r in test]);predictions={};out={}
 t=time.perf_counter()
 for arm in ['alignment','calibrated','baseline','weight','teacher']:
  if arm=='alignment':p=np.array([r['baseline'][3] for r in test])
  elif arm=='teacher':p=np.array([r['teacher'] for r in test])
  else:
   x=np.array([r['baseline']+(r['weight'] if arm=='weight' else []) for r in test]);x=x[:,3:4] if arm=='calibrated' else x;sc,m=models[arm];p=m.predict(sc.transform(x))
  predictions[arm]=p.tolist();per=[]
  for s in range(8):
   mm=[]
   for j in [s*16,s*16+8]:
    yy=y[j:j+8];pp=p[j:j+8];mm.append([np.mean(abs(yy-pp)),spearmanr(yy,pp).statistic,np.sort(yy)[-2:].mean()-yy[np.argsort(pp)[-2:]].mean()])
   per.append(np.mean(mm,axis=0).tolist())
  out[arm]={'mean':np.mean(per,axis=0).tolist(),'per_seed':per}
 timing['prediction_and_metrics']=time.perf_counter()-t;comparisons={}
 for a,b in [('baseline','calibrated'),('weight','baseline')]:
  d=np.array(out[a]['per_seed'])[:,0]-np.array(out[b]['per_seed'])[:,0];bs=np.random.default_rng(1612).choice(d,(10000,8),replace=True).mean(axis=1);comparisons[a+' minus '+b]={'mae_difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
 files=['experiments/utility_student.py','experiments/transformer_horizon.py','experiments/transformer_pilot.py','reports/25_student_protocol_ar.md','data/alice.txt','data/brown.txt']
 result={'outputs':out,'comparisons':comparisons,'alphas':alphas,'timing':timing,'seconds':time.perf_counter()-start,'rows':len(rows),'metric_order':['mae','spearman','top2_regret'],'brown_target_consumed':True,'torch':torch.__version__,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
 Path('results/utility-student.json').write_text(json.dumps(result,indent=2));Path('results/student-records.json').write_text(json.dumps(rows));Path('results/student-predictions.json').write_text(json.dumps(predictions));print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
