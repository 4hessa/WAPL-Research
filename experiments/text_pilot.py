"""Frozen-protocol character-bigram diagnostic. See reports/02_protocol_ar.md."""
import hashlib,json,time,platform
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
V=128
LR=.2

def counts(seq):
    seq=np.asarray(seq)
    return np.bincount(seq[:-1]*V+seq[1:],minlength=V*V).reshape(V,V).astype(float)

def loss_grad(w,c):
    z=w-logsumexp(w,axis=1,keepdims=True)
    total=c.sum()
    loss=-np.sum(c*z)/total
    g=(c.sum(axis=1,keepdims=True)*np.exp(z)-c)/total
    return float(loss),g

def step(w,v,c):
    g=loss_grad(w,c)[1]
    nv=.9*v+g
    return w-LR*nv,nv

def effects(w,v,c,follow,target):
    wi,vi=step(w.copy(),v.copy(),c);ws,vs=w.copy(),v.copy()
    out=[loss_grad(ws,target)[0]-loss_grad(wi,target)[0]]
    for j,b in enumerate(follow,2):
        wi,vi=step(wi,vi,b);ws,vs=step(ws,vs,b)
        if j in (4,16):out.append(loss_grad(ws,target)[0]-loss_grad(wi,target)[0])
    return out

def prepare(text):
    a=np.frombuffer(text.encode('ascii'),dtype=np.uint8).astype(int)
    boundaries=[0,.30,.40,.55,.65,.80,1.]
    pools=[];seen=set();removed=[]
    for lo,hi in zip(boundaries,boundaries[1:]):
        seq=a[int(lo*len(a)):int(hi*len(a))]
        pool=[];n=0
        for i in range(0,len(seq)-64,65):
            b=seq[i:i+65];key=hashlib.sha256(b.tobytes()).hexdigest()
            if key in seen:n+=1;continue
            seen.add(key);pool.append(b)
        pools.append(pool);removed.append(n)
    return pools,removed

def main():
    start=time.perf_counter();text=Path('data/tinyshakespeare.txt').read_text()
    pools,removed=prepare(text)
    # Reserved final pool is excluded from gradients, features and labels.
    probe=sum((counts(s) for s in pools[3]),np.zeros((V,V)))
    target=sum((counts(s) for s in pools[4]),np.zeros((V,V)))
    train=list(range(200,212));dev=list(range(300,306));test=list(range(1000,1016))
    records=[];timing={'baseline_features':0.,'weight_features':0.,'labels':0.}
    for partition,seeds in enumerate([train,dev,test]):
        pool=pools[partition]
        for seed in seeds:
            rng=np.random.default_rng(seed);w=rng.normal(0,.01,(V,V));v=np.zeros_like(w)
            def draw():return counts(pool[rng.integers(len(pool))])
            for checkpoint in range(2):
                for _ in range(12):w,v=step(w,v,draw())
                pg=loss_grad(w,probe)[1]
                follow=[draw() for _ in range(15)]
                for candidate in range(16):
                    c=draw();t=time.perf_counter();l,g=loss_grad(w,c);delta=-LR*(.9*v+g)
                    b=[l,np.linalg.norm(g),np.sum(g*pg),-np.sum(delta*pg),np.linalg.norm(delta)]
                    timing['baseline_features']+=time.perf_counter()-t
                    t=time.perf_counter();e=[np.sum(w*g),np.linalg.norm(w*g),np.linalg.norm(delta)/(np.linalg.norm(w)+1e-12)]
                    timing['weight_features']+=time.perf_counter()-t
                    t=time.perf_counter();y=effects(w,v,c,follow,target);timing['labels']+=time.perf_counter()-t
                    records.append({'seed':seed,'checkpoint':checkpoint,'candidate':candidate,'baseline':b,'weight':e,'utility':y})
    noise=np.random.default_rng(33).normal(size=(len(records),3))
    def matrix(seeds,arm,h):
        indices=[i for i,r in enumerate(records) if r['seed'] in seeds]
        x=np.array([records[i]['baseline']+(records[i]['weight'] if arm=='weight' else list(noise[i]) if arm=='random' else []) for i in indices])
        y=np.array([records[i]['utility'][h] for i in indices]);return x,y
    t=time.perf_counter();output={};predictions={}
    for hidx,h in enumerate([1,4,16]):
        output[str(h)]={}
        for arm in ['baseline','weight','random','alignment','loss']:
            x,y=matrix(train,arm,hidx);xd,yd=matrix(dev,arm,hidx)
            alpha=None
            if arm not in ('alignment','loss'):
                scaler=StandardScaler().fit(x)
                scores=[]
                for a in [.1,1.,10.,100.]:
                    model=Ridge(alpha=a).fit(scaler.transform(x),y)
                    scores.append((np.mean(abs(model.predict(scaler.transform(xd))-yd)),a))
                alpha=min(scores)[1];model=Ridge(alpha=alpha).fit(scaler.transform(x),y)
            seed_metrics=[];prediction_rows=[]
            for seed in test:
                xt,yt=matrix([seed],arm,hidx)
                p=xt[:,3] if arm=='alignment' else xt[:,0] if arm=='loss' else model.predict(scaler.transform(xt))
                values=[]
                for j in [0,16]:
                    yy=yt[j:j+16];pp=p[j:j+16];chosen=yy[np.argsort(pp)[-4:]].mean();oracle=np.sort(yy)[-4:].mean()
                    values.append([spearmanr(yy,pp).statistic, np.mean(abs(yy-pp)),chosen,oracle-chosen])
                seed_metrics.append(np.mean(values,axis=0).tolist())
                prediction_rows.append({'seed':seed,'prediction':p.tolist(),'utility':yt.tolist()})
            means=np.mean(seed_metrics,axis=0)
            output[str(h)][arm]={'alpha':alpha,'spearman':float(means[0]),'mae':None if arm=='loss' else float(means[1]),'top4_utility':float(means[2]),'regret':float(means[3]),'seed_metrics':seed_metrics}
            predictions[f'{h}/{arm}']=prediction_rows
        diff=np.array(output[str(h)]['weight']['seed_metrics'])[:,0]-np.array(output[str(h)]['baseline']['seed_metrics'])[:,0]
        boot=np.random.default_rng(711).choice(diff,(10000,len(diff)),replace=True).mean(axis=1)
        output[str(h)]['comparison']={'difference':float(diff.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
    timing['analysis']=time.perf_counter()-t
    output['metadata']={'total_seconds':time.perf_counter()-start,'timing':timing,'rows':len(records),'labels':len(records)*3,'pool_sizes':list(map(len,pools)),'duplicate_chunks_removed':removed,'parameters':V*V,'train_seeds':train,'dev_seeds':dev,'test_seeds':test,'protocol_sha256':hashlib.sha256(Path('reports/02_protocol_ar.md').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path('data/tinyshakespeare.txt').read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'python':platform.python_version()}
    Path('results/text-pilot.json').write_text(json.dumps(output,indent=2))
    Path('results/text-records.json').write_text(json.dumps(records))
    Path('results/text-predictions.json').write_text(json.dumps(predictions))
    print(json.dumps({h:{k:{a:b for a,b in v.items() if a!='seed_metrics'} for k,v in output[h].items()} for h in ['1','4','16']},indent=2));print(output['metadata'])
if __name__=='__main__':main()
