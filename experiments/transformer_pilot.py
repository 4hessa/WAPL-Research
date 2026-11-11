import copy,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class TinyTransformer(nn.Module):
 def __init__(self):
  super().__init__();self.token=nn.Embedding(128,16);self.pos=nn.Embedding(32,16)
  self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(16,2,32,dropout=0.,batch_first=True) for _ in range(2)])
  self.head=nn.Linear(16,128)
 def forward(self,x):
  n=x.shape[1];h=self.token(x)+self.pos(torch.arange(n));mask=torch.triu(torch.ones(n,n,dtype=torch.bool),1)
  for block in self.blocks:h=block(h,src_mask=mask)
  return self.head(h)

def loss(m,b):return nn.functional.cross_entropy(m(b[:,:-1]).reshape(-1,128),b[:,1:].reshape(-1))
def vec(m,grad=False):return torch.cat([(p.grad if grad else p).detach().flatten() for p in m.parameters()])
def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
def update(m,o,b):
 o.zero_grad();l=loss(m,b);l.backward();o.step()
def clone(m,o):
 c=copy.deepcopy(m);co=optimizer(c);co.load_state_dict(copy.deepcopy(o.state_dict()));return c,co

def prepare():
 a=np.frombuffer(Path('data/persuasion.txt').read_bytes(),dtype=np.uint8).astype(np.int64);edges=[0,.3,.4,.55,.65,.8,1.];seen=set();pools=[]
 for lo,hi in zip(edges,edges[1:]):
  part=a[int(len(a)*lo):int(len(a)*hi)];rows=[]
  for i in range(0,len(part)-32,33):
   b=part[i:i+33];key=b.tobytes()
   if key not in seen:seen.add(key);rows.append(b)
  pools.append(torch.tensor(np.array(rows),dtype=torch.long))
 return pools

def main():
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();pools=prepare();probe=pools[3][:32];target=pools[4][:32];rows=[]
 for part,seeds in enumerate([range(10000,10008),range(10100,10104),range(10200,10208)]):
  for seed in seeds:
   torch.manual_seed(seed);rng=np.random.default_rng(seed);m=TinyTransformer();o=optimizer(m);done=0
   def draw(n):return pools[part][rng.integers(len(pools[part]),size=n)]
   for checkpoint in [12,48]:
    while done<checkpoint:update(m,o,draw(4));done+=1
    o.zero_grad();loss(m,probe).backward();pg=vec(m,True);w=vec(m);follow=[draw(4) for _ in range(3)]
    for i in range(8):
     c=draw(1);mi,oi=clone(m,o);ms,os=clone(m,o)
     oi.zero_grad();l=loss(mi,c);l.backward();g=vec(mi,True);oi.step();delta=vec(mi)-w
     b=[l.item(),g.norm().item(),g.dot(pg).item(),-delta.dot(pg).item(),delta.norm().item()]
     e=[w.dot(g).item(),(w*g).norm().item(),(delta.norm()/(w.norm()+1e-12)).item()]
     for batch in follow:update(mi,oi,batch);update(ms,os,batch)
     with torch.no_grad():y=(loss(ms,target)-loss(mi,target)).item()
     rows.append({'seed':seed,'checkpoint':checkpoint,'candidate':i,'split':['train','dev','test'][part],'baseline':b,'weight':e,'utility':y})
 noise=np.random.default_rng(112).normal(size=(len(rows),3));outputs={};predictions={}
 def xy(split,arm):
  ids=[i for i,r in enumerate(rows) if r['split']==split];return np.array([rows[i]['baseline']+(rows[i]['weight'] if arm=='weight' else noise[i].tolist() if arm=='random' else []) for i in ids]),np.array([rows[i]['utility'] for i in ids])
 for arm in ['baseline','weight','random','alignment']:
  x,y=xy('train',arm);xd,yd=xy('dev',arm);xt,yt=xy('test',arm);alpha=None
  if arm=='alignment':p=xt[:,3]
  else:
   sc=StandardScaler().fit(x);opts=[]
   for a in [.1,1.,10.,100.]:
    model=Ridge(alpha=a).fit(sc.transform(x),y);opts.append((np.mean(abs(model.predict(sc.transform(xd))-yd)),a))
   alpha=min(opts)[1];model=Ridge(alpha=alpha).fit(sc.transform(x),y);p=model.predict(sc.transform(xt))
  per=[]
  for k in range(8):
   group=[]
   for j in [k*16,k*16+8]:
    yy=yt[j:j+8];pp=p[j:j+8];group.append([spearmanr(yy,pp).statistic,np.mean(abs(yy-pp)),np.sort(yy)[-2:].mean()-yy[np.argsort(pp)[-2:]].mean()])
   per.append(np.mean(group,axis=0).tolist())
  outputs[arm]={'alpha':alpha,'mean':np.mean(per,axis=0).tolist(),'per_seed':per};predictions[arm]=p.tolist()
 d=np.array(outputs['weight']['per_seed'])[:,0]-np.array(outputs['baseline']['per_seed'])[:,0];bs=np.random.default_rng(1212).choice(d,(10000,8),replace=True).mean(axis=1)
 files=['experiments/transformer_pilot.py','reports/17_transformer_protocol_ar.md','data/persuasion.txt']
 result={'outputs':outputs,'comparison':{'difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()},'parameters':sum(p.numel() for p in m.parameters()),'rows':len(rows),'negative_labels':sum(r['utility']<0 for r in rows),'seconds':time.perf_counter()-start,'torch':torch.__version__,'metric_order':['spearman','mae','top2_regret'],'final_holdout_unused':True,'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
 Path('results/transformer-pilot.json').write_text(json.dumps(result,indent=2));Path('results/transformer-records.json').write_text(json.dumps(rows));Path('results/transformer-predictions.json').write_text(json.dumps(predictions));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
