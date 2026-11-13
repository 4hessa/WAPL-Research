"""Exploratory corrected-head feature ablation, separately by language."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from experiments.multilingual_readiness import ByteTransformer,byte_chunks,byte_loss
from experiments.multilingual_policy import optimizer,clone,update
from experiments.head_invariance import canonical_head,flatten,features,utility

ARMS=['baseline','corrected','raw','noise']

def design(rows,arm):
    return np.array([r['baseline']+(r[arm] if arm!='baseline' else []) for r in rows])

def fit_models(rows):
    tr=[r for r in rows if r['split']=='fit'];dv=[r for r in rows if r['split']=='tune'];y=np.array([r['utility'] for r in tr]);yd=np.array([r['utility'] for r in dv]);models={};selection={}
    for arm in ARMS:
        x=design(tr,arm);xd=design(dv,arm);sc=StandardScaler().fit(x);opts=[];fitted={}
        for alpha in [.1,1.,10.,100.]:
            model=Ridge(alpha=alpha).fit(sc.transform(x),y);opts.append((float(np.mean(abs(model.predict(sc.transform(xd))-yd))),alpha));fitted[alpha]=model
        alpha=min(opts)[1];models[arm]=(sc,fitted[alpha]);selection[arm]={'alpha':alpha,'tune_mae':opts}
    return models,selection

def generate(language,noise):
    train=byte_chunks(Path(f'data/ud_{language}_train.txt').read_bytes());dev=byte_chunks(Path(f'data/ud_{language}_dev.txt').read_bytes());n=len(train)
    pools={'fit':train[:int(n*.35)],'tune':train[int(n*.35):int(n*.5)],'validate':train[int(n*.5):int(n*.7)]}
    probe=train[int(n*.7):int(n*.85)][:64];monitor=train[int(n*.85):][:128];dev_ids=np.random.default_rng(4800).choice(len(dev),128,replace=False);target=dev[dev_ids];rows=[]
    for split,seeds in [('fit',range(25000,25008)),('tune',range(25100,25104)),('validate',range(25200,25208))]:
        pool=pools[split]
        for seed in seeds:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=ByteTransformer();o=optimizer(m);done=0
            def draw(n):return pool[rng.integers(len(pool),size=n)]
            for checkpoint in [16,48]:
                while done<checkpoint:update(m,o,draw(4));done+=1
                follow=[draw(4) for _ in range(3)];pm,po=clone(m,o);po.zero_grad();byte_loss(pm,probe).backward();pg=flatten({k:p.grad for k,p in pm.named_parameters()});w={k:p.detach().clone() for k,p in m.named_parameters()}
                for candidate_id in range(4):
                    c=draw(1);cm,co=clone(m,o);co.zero_grad();l=byte_loss(cm,c);l.backward();g={k:p.grad.detach().clone() for k,p in cm.named_parameters()};co.step();d={k:p.detach()-w[k] for k,p in cm.named_parameters()};dc=flatten(canonical_head(d));gv=flatten(g)
                    baseline=[float(l.detach()),float(gv.norm()),float(gv.dot(pg)),float(-dc.dot(pg)),float(dc.norm())]
                    y=utility(m,o,c,follow,target if split=='validate' else monitor)
                    rows.append(dict(language=language,split=split,seed=seed,checkpoint=checkpoint,candidate=candidate_id,baseline=baseline,corrected=features(w,g,d,True).tolist(),raw=features(w,g,d).tolist(),noise=noise.normal(size=3).tolist(),utility=y))
        print(f'{language} {split} labels complete',flush=True)
    return rows,dev_ids.tolist()

def evaluate(rows,models):
    rr=[r for r in rows if r['split']=='validate'];y=np.array([r['utility'] for r in rr]);outputs={};predictions={}
    for arm in ARMS+['alignment']:
        if arm=='alignment':p=np.array([r['baseline'][3] for r in rr])
        else:sc,m=models[arm];p=m.predict(sc.transform(design(rr,arm)))
        predictions[arm]=p.tolist();per=[]
        for seed in range(25200,25208):
            states=[]
            for cp in [16,48]:
                ix=[i for i,r in enumerate(rr) if r['seed']==seed and r['checkpoint']==cp];yy=y[ix];pp=p[ix];states.append([float(np.mean(abs(yy-pp))),float(spearmanr(yy,pp).statistic),float(np.sort(yy)[-2:].mean()-yy[np.argsort(pp,kind='stable')[-2:]].mean())])
            per.append(np.mean(states,axis=0).tolist())
        outputs[arm]={'mean':np.mean(per,axis=0).tolist(),'per_seed':per}
    comparisons={}
    for control in ['baseline','raw','noise']:
        d=np.array(outputs['corrected']['per_seed'])[:,0]-np.array(outputs[control]['per_seed'])[:,0];bs=np.random.default_rng(4802).choice(d,(10000,8),replace=True).mean(axis=1)
        comparisons['corrected minus '+control]={'mae_difference':float(d.mean()),'exploratory_ci95':np.quantile(bs,[.025,.975]).tolist()}
    return outputs,comparisons,predictions

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();all_rows=[];outputs={};comparisons={};predictions={};selection={};dev_ids={};noise=np.random.default_rng(4801)
    manifest=json.loads(Path('data/multilingual_sources.json').read_text())
    for language in ['ar','en']:
        for split in ['train','dev']:
            e=manifest['languages'][language]['splits'][split];assert hashlib.sha256(Path(e['path']).read_bytes()).hexdigest()==e['sha256']
        rows,dev_ids[language]=generate(language,noise);models,selection[language]=fit_models(rows);outputs[language],comparisons[language],predictions[language]=evaluate(rows,models);all_rows.extend(rows)
    files=['experiments/corrected_features.py','experiments/head_invariance.py','experiments/multilingual_readiness.py','experiments/multilingual_policy.py','reports/48_corrected_features_protocol_ar.md','data/multilingual_sources.json']+[f'data/ud_{l}_{s}.txt' for l in ['ar','en'] for s in ['train','dev']]
    result={'status':'exploratory_same_sources_previously_used_in_development','outputs':outputs,'comparisons':comparisons,'selection':selection,'dev_indices':dev_ids,'rows':len(all_rows),'metric_order':['mae','spearman','top2_regret'],'seconds':time.perf_counter()-start,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    for name,value in [('corrected-features',result),('corrected-feature-records',all_rows),('corrected-feature-predictions',predictions)]:Path('results/'+name+'.json').write_text(json.dumps(value,indent=2)+'\n')
    p=Path('data/evaluation_status.json');ledger=json.loads(p.read_text());ledger['stage']=49
    for item in ledger['datasets'].values():item['dev']='128 chunks evaluated as utility-validation monitor in exploratory stage 49; exposed';item['test']='evaluated in stage 45; exposed; not read by stage 49'
    p.write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
