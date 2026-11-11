"""Fixed-source, equal-row state coverage diagnostic; protocol 11."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from experiments.text_pilot import V,LR,prepare,counts,loss_grad,step,effects
from experiments.invariance import features

def training_rows(records,coverage):
    return [r for r in records if r['split']=='train' and ((coverage=='early' and r['checkpoint']==24) or (coverage=='multi' and r['candidate']<16))]

def main():
    start=time.perf_counter();pools,removed=prepare(Path('data/emma.txt').read_text())
    probe=sum((counts(s) for s in pools[3]),np.zeros((V,V)));target=sum((counts(s) for s in pools[4]),np.zeros((V,V)))
    records=[]
    for split,part,seeds in [('train',0,range(6000,6012)),('dev',1,range(6100,6106)),('test',2,range(7000,7012))]:
        for seed in seeds:
            rng=np.random.default_rng(seed);w=rng.normal(0,.01,(V,V));v=np.zeros_like(w);done=0
            def draw():return counts(pools[part][rng.integers(len(pools[part]))])
            for checkpoint in [24,120,480]:
                while done<checkpoint:w,v=step(w,v,draw());done+=1
                pg=loss_grad(w,probe)[1];follow=[draw() for _ in range(3)]
                for i in range(48 if split=='train' else 16):
                    c=draw();l,g=loss_grad(w,c);d=-LR*(.9*v+g)
                    b=[l,np.linalg.norm(g),np.sum(g*pg),-np.sum(d*pg),np.linalg.norm(d)]
                    records.append({'split':split,'seed':seed,'checkpoint':checkpoint,'candidate':i,'baseline':b,'centered':features(w,g,d,True).tolist(),'utility':effects(w,v,c,follow,target)[1]})
    def xy(rows,arm):return np.array([r['baseline']+(r['centered'] if arm=='centered' else []) for r in rows]),np.array([r['utility'] for r in rows])
    outputs={};predictions=[]
    for coverage in ['early','multi']:
        rows=training_rows(records,coverage);assert len(rows)==576
        for arm in ['baseline','centered']:
            x,y=xy(rows,arm);xd,yd=xy([r for r in records if r['split']=='dev'],arm)
            sc=StandardScaler().fit(x);options=[]
            for a in [.1,1.,10.,100.]:
                m=Ridge(alpha=a).fit(sc.transform(x),y);options.append((np.mean(abs(m.predict(sc.transform(xd))-yd)),a))
            a=min(options)[1];m=Ridge(alpha=a).fit(sc.transform(x),y);key=coverage+'_'+arm;outputs[key]={'alpha':a,'train_rows':len(rows),'states':{}}
            for checkpoint in [24,120,480]:
                metrics=[]
                for seed in range(7000,7012):
                    rr=[r for r in records if r['split']=='test' and r['seed']==seed and r['checkpoint']==checkpoint];xt,yt=xy(rr,arm);p=m.predict(sc.transform(xt))
                    metrics.append([float(spearmanr(yt,p).statistic),float(np.mean(abs(yt-p))),float(np.sort(yt)[-4:].mean()-yt[np.argsort(p)[-4:]].mean())])
                    predictions.append({'arm':key,'seed':seed,'checkpoint':checkpoint,'predictions':p.tolist(),'utilities':yt.tolist()})
                outputs[key]['states'][str(checkpoint)]={'mean':np.mean(metrics,axis=0).tolist(),'per_seed':metrics}
    outputs['alignment']={'states':{}}
    for cp in [24,120,480]:
        mm=[]
        for seed in range(7000,7012):
            rr=[r for r in records if r['split']=='test' and r['seed']==seed and r['checkpoint']==cp];y=np.array([r['utility'] for r in rr]);p=np.array([r['baseline'][3] for r in rr])
            mm.append([float(spearmanr(y,p).statistic),float(np.mean(abs(y-p))),float(np.sort(y)[-4:].mean()-y[np.argsort(p)[-4:]].mean())])
        outputs['alignment']['states'][str(cp)]={'mean':np.mean(mm,axis=0).tolist(),'per_seed':mm}
    comparisons={}
    for a,b in [('multi_centered','multi_baseline'),('multi_baseline','early_baseline')]:
        d=np.array(outputs[a]['states']['480']['per_seed'])[:,0]-np.array(outputs[b]['states']['480']['per_seed'])[:,0]
        bs=np.random.default_rng(1012).choice(d,(10000,12),replace=True).mean(axis=1)
        comparisons[a+' minus '+b]={'difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()}
    files=['experiments/state_coverage.py','experiments/text_pilot.py','experiments/invariance.py','reports/11_state_protocol_ar.md','data/emma.txt']
    result={'outputs':outputs,'comparisons':comparisons,'rows':len(records),'negative_labels':sum(r['utility']<0 for r in records),'seconds':time.perf_counter()-start,'pool_sizes':list(map(len,pools)),'removed_duplicates':removed,'metric_order':['spearman','mae','top4_regret'],'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},'reserved_final_partition_unused':True}
    Path('results/state-coverage.json').write_text(json.dumps(result,indent=2));Path('results/state-records.json').write_text(json.dumps(records));Path('results/state-predictions.json').write_text(json.dumps(predictions));print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
