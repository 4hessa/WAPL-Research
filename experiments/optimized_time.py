import json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from experiments.text_pilot import V,LR,prepare,counts,loss_grad,step
from experiments.state_coverage import training_rows
from experiments.invariance import features
from experiments.reuse import shared_scores

def fit_models():
    rr=json.loads(Path('results/state-records.json').read_text());train=training_rows(rr,'multi');dev=[r for r in rr if r['split']=='dev'];out={};alphas={}
    for arm in ['baseline','centered']:
        def xy(rows):return np.array([r['baseline']+(r['centered'] if arm=='centered' else []) for r in rows]),np.array([r['utility'] for r in rows])
        x,y=xy(train);xd,yd=xy(dev);sc=StandardScaler().fit(x);scores=[]
        for a in [.1,1.,10.,100.]:
            m=Ridge(alpha=a).fit(sc.transform(x),y);scores.append((np.mean(abs(m.predict(sc.transform(xd))-yd)),a))
        a=min(scores)[1];out[arm]=(sc,Ridge(alpha=a).fit(sc.transform(x),y));alphas[arm]=a
    return out,alphas

def feature_batch(w,v,cs,probe,arm):
    logp=w-logsumexp(w,axis=1,keepdims=True);p=np.exp(logp)
    gp=(probe.sum(axis=1,keepdims=True)*p-probe)/probe.sum();rows=[];grads=[]
    for c in cs:
        g=(c.sum(axis=1,keepdims=True)*p-c)/c.sum();d=-LR*(.9*v+g)
        row=[-np.sum(c*logp)/c.sum(),np.linalg.norm(g),np.sum(g*gp),-np.sum(d*gp),np.linalg.norm(d)]
        if arm=='centered':row+=features(w,g,d,True).tolist()
        rows.append(row);grads.append(g)
    return np.array(rows),grads

def run(w,v,pool,probe,ids,ri,arm,models,budget=1.):
    w,v=w.copy(),v.copy();n=0;last=0.;choices=[];start=time.perf_counter()
    for ix,j in zip(ids,ri):
        if time.perf_counter()-start>=budget:break
        if arm=='random':k=int(j);wn,vn=step(w,v,counts(pool[ix[k]]))
        else:
            cs=[counts(pool[i]) for i in ix]
            if arm=='alignment':
                scores,p=shared_scores(w,v,cs,probe);k=int(np.argmax(scores));c=cs[k];g=(c.sum(axis=1,keepdims=True)*p-c)/c.sum()
            else:
                x,gs=feature_batch(w,v,cs,probe,arm);sc,m=models[arm];k=int(np.argmax(m.predict(sc.transform(x))));g=gs[k]
            vn=.9*v+g;wn=w-LR*vn
        t=time.perf_counter()-start
        if t>budget:break
        w,v=wn,vn;n+=1;last=t;choices.append(int(ix[k]))
    elapsed=time.perf_counter()-start
    return w,{'updates':n,'transitions':n*64,'last_completed':last,'actual_seconds':elapsed,'overrun':max(0.,elapsed-budget),'choices':choices}

def main():
    start=time.perf_counter();pools,_=prepare(Path('data/emma.txt').read_text());pool=pools[0]+pools[1]+pools[2];probe=sum((counts(s) for s in pools[3]),np.zeros((V,V)));final=sum((counts(s) for s in pools[5]),np.zeros((V,V)))
    t=time.perf_counter();models,alphas=fit_models();fit_seconds=time.perf_counter()-t
    z=np.zeros((V,V));c=counts(pool[0]);feature_batch(z,z,[c]*8,probe,'centered');shared_scores(z,z,[c]*8,probe)
    arms=['random','alignment','baseline','centered'];runs=[]
    for seed in range(9000,9012):
        rng=np.random.default_rng(seed);w=rng.normal(0,.01,(V,V));v=np.zeros_like(w)
        for _ in range(24):w,v=step(w,v,counts(pool[rng.integers(len(pool))]))
        ids=rng.integers(len(pool),size=(50000,8));ri=rng.integers(8,size=50000);schedule=hashlib.sha256(ids.tobytes()+ri.tobytes()).hexdigest();initial=loss_grad(w,final)[0]
        for arm in arms[seed%4:]+arms[:seed%4]:
            wn,r=run(w,v,pool,probe,ids,ri,arm,models)
            assert r['updates']<len(ids) and r['last_completed']<=1.
            r.update({'seed':seed,'arm':arm,'initial_loss':initial,'final_loss':loss_grad(wn,final)[0],'schedule_sha256':schedule});runs.append(r)
    summary={a:{'loss':float(np.mean([r['final_loss'] for r in runs if r['arm']==a])),'updates':float(np.mean([r['updates'] for r in runs if r['arm']==a])),'seconds':float(np.mean([r['actual_seconds'] for r in runs if r['arm']==a])),'max_overrun':max(r['overrun'] for r in runs if r['arm']==a)} for a in arms}
    comparisons={}
    for a,b in [('centered','baseline'),('alignment','random'),('centered','random')]:
        d=np.array([next(r['final_loss'] for r in runs if r['seed']==s and r['arm']==a)-next(r['final_loss'] for r in runs if r['seed']==s and r['arm']==b) for s in range(9000,9012)])
        bs=np.random.default_rng(1112).choice(d,(10000,12),replace=True).mean(axis=1);comparisons[a+' minus '+b]={'difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
    paths=['experiments/optimized_time.py','experiments/reuse.py','experiments/text_pilot.py','experiments/invariance.py','experiments/state_coverage.py','results/state-records.json','reports/15_optimized_time_protocol_ar.md','data/emma.txt']
    result={'summary':summary,'comparisons':comparisons,'runs':runs,'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-start,'alphas':alphas,'emma_final_consumed':True,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}}
    Path('results/optimized-time.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['runs','sha256']},indent=2))
if __name__=='__main__':main()
