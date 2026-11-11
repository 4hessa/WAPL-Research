"""Frozen local protocol: reports/06_policy_protocol_ar.md."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from experiments.text_pilot import V,LR,prepare,counts,loss_grad,step
from experiments.invariance import features

def fit_predictors(records):
    models={};alphas={}
    for arm in ['baseline','centered']:
        def matrix(seeds):
            rows=[r for r in records if r['seed'] in seeds]
            return np.array([r['baseline']+(r['centered'] if arm=='centered' else []) for r in rows]),np.array([r['utility'][1] for r in rows])
        x,y=matrix(range(2200,2212));xd,yd=matrix(range(2300,2306))
        scaler=StandardScaler().fit(x);scores=[]
        for a in [.1,1.,10.,100.]:
            m=Ridge(alpha=a).fit(scaler.transform(x),y)
            scores.append((np.mean(abs(m.predict(scaler.transform(xd))-yd)),a))
        a=min(scores)[1];models[arm]=(scaler,Ridge(alpha=a).fit(scaler.transform(x),y));alphas[arm]=a
    return models,alphas

def choose(w,v,candidates,probe,arm,models,random_index):
    if arm=='random':return random_index
    pg=loss_grad(w,probe)[1];rows=[]
    for c in candidates:
        l,g=loss_grad(w,c);d=-LR*(.9*v+g)
        b=[l,np.linalg.norm(g),np.sum(g*pg),-np.sum(d*pg),np.linalg.norm(d)]
        rows.append(b+(features(w,g,d,True).tolist() if arm=='centered' else []))
    x=np.array(rows)
    if arm=='alignment':scores=x[:,3]
    else:
        scaler,model=models[arm];scores=model.predict(scaler.transform(x))
    return int(np.argmax(scores))

def main():
    started=time.perf_counter();pools,_=prepare(Path('data/tinyshakespeare.txt').read_text())
    probe=sum((counts(s) for s in pools[3]),np.zeros((V,V)))
    final=sum((counts(s) for s in pools[5]),np.zeros((V,V)))
    t=time.perf_counter();models,alphas=fit_predictors(json.loads(Path('results/centered-records.json').read_text()));fit_seconds=time.perf_counter()-t
    arms=['random','alignment','baseline','centered'];runs=[]
    for seed in range(4000,4012):
        rng=np.random.default_rng(seed);w=rng.normal(0,.01,(V,V));v=np.zeros_like(w)
        for _ in range(24):w,v=step(w,v,counts(pools[2][rng.integers(len(pools[2]))]))
        candidate_ids=rng.integers(len(pools[2]),size=(32,8));random_choices=rng.integers(8,size=32)
        initial=loss_grad(w,final)[0]
        shift=seed%4;order=arms[shift:]+arms[:shift]
        for arm in order:
            wa,va=w.copy(),v.copy();select_seconds=0.;train_seconds=0.;selected=[]
            for ids,ri in zip(candidate_ids,random_choices):
                t=time.perf_counter()
                # Random policy only constructs its selected example.
                if arm=='random':k=int(ri);c=counts(pools[2][ids[k]])
                else:
                    cs=[counts(pools[2][i]) for i in ids];k=choose(wa,va,cs,probe,arm,models,int(ri));c=cs[k]
                select_seconds+=time.perf_counter()-t
                t=time.perf_counter();wa,va=step(wa,va,c);train_seconds+=time.perf_counter()-t
                selected.append(int(ids[k]))
            runs.append({'seed':seed,'arm':arm,'initial_loss':initial,'final_loss':loss_grad(wa,final)[0], 'selection_seconds':select_seconds,'training_seconds':train_seconds,'updates':32,'trained_transitions':2048,'selected_ids':selected,'candidate_ids':candidate_ids.tolist()})
    summaries={a:{'final_loss':float(np.mean([r['final_loss'] for r in runs if r['arm']==a])),'mean_policy_seconds':float(np.mean([r['selection_seconds']+r['training_seconds'] for r in runs if r['arm']==a]))} for a in arms}
    comparisons={}
    for a,b in [('centered','baseline'),('centered','random'),('baseline','random'),('alignment','random')]:
        d=np.array([next(r['final_loss'] for r in runs if r['seed']==s and r['arm']==a)-next(r['final_loss'] for r in runs if r['seed']==s and r['arm']==b) for s in range(4000,4012)])
        boot=np.random.default_rng(812).choice(d,(10000,len(d)),replace=True).mean(axis=1)
        comparisons[a+' minus '+b]={'difference':float(d.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
    paths=['experiments/policy.py','experiments/text_pilot.py','experiments/invariance.py','results/centered-records.json','reports/06_policy_protocol_ar.md','data/tinyshakespeare.txt']
    result={'runs':runs,'summary':summaries,'comparisons':comparisons,'fit_seconds':fit_seconds,'total_seconds':time.perf_counter()-started,'alphas':alphas,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'final_partition_now_used':True}
    Path('results/policy.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
