"""Frozen-protocol comparison of pairwise and pointwise utility learning."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from experiments.utility_student import fit_models,prepare
from experiments.transformer_pilot import TinyTransformer,optimizer,clone,update,vec,loss
from experiments.transformer_horizon import rollout


def regret(y,p):
    return float(np.sort(y)[-2:].mean()-y[np.argsort(p,kind='stable')[-2:]].mean())


def fit_rankers(rows):
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='dev']
    sc=StandardScaler().fit([r['baseline'] for r in train])
    x=sc.transform([r['baseline'] for r in train]);y=np.array([r['teacher'] for r in train])
    xd=sc.transform([r['baseline'] for r in dev]);yd=np.array([r['teacher'] for r in dev])
    dx=[];dy=[]
    for key in sorted({(r['seed'],r['checkpoint']) for r in train}):
        ix=[i for i,r in enumerate(train) if (r['seed'],r['checkpoint'])==key]
        for i in ix:
            for j in ix:
                if i!=j:dx.append(x[i]-x[j]);dy.append(y[i]-y[j])
    groups=[[i for i,r in enumerate(dev) if (r['seed'],r['checkpoint'])==key] for key in sorted({(r['seed'],r['checkpoint']) for r in dev})]
    models={};selection={}
    for arm in ['pairwise','point_regret']:
        scores=[];candidates={}
        for alpha in [.1,1.,10.,100.]:
            m=Ridge(alpha=alpha,fit_intercept=arm!='pairwise').fit(dx if arm=='pairwise' else x,dy if arm=='pairwise' else y)
            p=m.predict(xd);score=float(np.mean([regret(yd[ix],p[ix]) for ix in groups]));scores.append((score,alpha));candidates[alpha]=m
        best=min(scores)[1];models[arm]=(sc,candidates[best]);selection[arm]={'alpha':best,'dev_regrets':scores}
    previous,alphas=fit_models(rows)
    models['point_mae']=previous['baseline'];models['calibrated']=previous['calibrated']
    selection['previous_alphas']=alphas
    return models,selection


def generate():
    pools=prepare('data/bryant.txt');pool=torch.cat(pools[:3]);probe=pools[3][:32];target=pools[5][:32];rows=[]
    for seed in range(15000,15012):
        torch.manual_seed(seed);rng=np.random.default_rng(seed);m=TinyTransformer();o=optimizer(m);done=0
        def draw(n):return pool[rng.integers(len(pool),size=n)]
        for cp in [12,48]:
            while done<cp:update(m,o,draw(4));done+=1
            tf=[draw(4) for _ in range(15)];yf=[draw(4) for _ in range(15)]
            o.zero_grad();loss(m,probe).backward();pg=vec(m,True);w=vec(m)
            for candidate in range(8):
                c=draw(1);mi,oi=clone(m,o);oi.zero_grad();l=loss(mi,c);l.backward();g=vec(mi,True);oi.step();d=vec(mi)-w
                b=[l.item(),g.norm().item(),g.dot(pg).item(),-d.dot(pg).item(),d.norm().item()]
                teacher=rollout(m,o,c,tf,{'probe':probe})['16']['probe'];y=rollout(m,o,c,yf,{'target':target})['16']['target']
                rows.append(dict(seed=seed,checkpoint=cp,candidate=candidate,baseline=b,teacher=teacher,target=y))
    return rows


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=json.loads(Path('results/student-records.json').read_text())
    t=time.perf_counter();models,selection=fit_rankers(source);fit_seconds=time.perf_counter()-t
    t=time.perf_counter();rows=generate();generation_seconds=time.perf_counter()-t
    x=np.array([r['baseline'] for r in rows]);y=np.array([r['target'] for r in rows]);predictions={};outputs={}
    for arm in list(models)+['teacher']:
        if arm=='teacher':p=np.array([r['teacher'] for r in rows])
        else:
            sc,m=models[arm];p=m.predict(sc.transform(x[:,3:4] if arm=='calibrated' else x))
        predictions[arm]=p.tolist();per=[]
        for seed in range(15000,15012):
            states=[]
            for cp in [12,48]:
                ix=[i for i,r in enumerate(rows) if r['seed']==seed and r['checkpoint']==cp]
                states.append([regret(y[ix],p[ix]),float(spearmanr(y[ix],p[ix]).statistic)])
            per.append(np.mean(states,axis=0).tolist())
        outputs[arm]={'mean':np.mean(per,axis=0).tolist(),'per_seed':per}
    delta=np.array(outputs['pairwise']['per_seed'])[:,0]-np.array(outputs['point_regret']['per_seed'])[:,0]
    bs=np.random.default_rng(2801).choice(delta,(10000,12),replace=True).mean(axis=1)
    files=['experiments/pairwise_student.py','experiments/utility_student.py','experiments/transformer_pilot.py','experiments/transformer_horizon.py','reports/28_pairwise_protocol_ar.md','results/student-records.json','data/bryant.txt']
    out={'outputs':outputs,'selection':selection,'primary':{'regret_difference':float(delta.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()},'metric_order':['top2_regret','spearman'],'rows':len(rows),'fit_seconds':fit_seconds,'generation_seconds':generation_seconds,'bryant_target_consumed':True,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    for name,value in [('pairwise-student',out),('pairwise-records',rows),('pairwise-predictions',predictions)]:Path('results/'+name+'.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
