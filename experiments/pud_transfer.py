"""Frozen PUD transfer evaluation: ten folds averaged within each model seed."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from experiments.multilingual_readiness import ByteTransformer,byte_chunks,byte_loss
from experiments.multilingual_policy import optimizer,clone,update
from experiments.head_invariance import canonical_head,flatten,features,utility


def partition(rows,folds,index):
    target=set(folds[index]);probe=set(folds[(index+1)%10]);candidate=set(r['source_index'] for r in rows)-target-probe
    assert not target.intersection(probe) and not target.intersection(candidate)
    lookup={r['source_index']:r['text'] for r in rows}
    def chunks(ids):return byte_chunks(('\n'.join(lookup[i] for i in sorted(ids))+'\n').encode('utf-8'))
    return chunks(candidate),chunks(probe)[:64],chunks(target)


def frozen_scores(rows,arm,frozen):
    x=np.array([r['baseline']+(r[arm] if arm!='baseline' else []) for r in rows]);f=frozen[arm]
    return ((x-np.array(f['feature_mean']))/np.array(f['feature_scale']))@np.array(f['coefficient'])+f['intercept']


def summarize(rows,frozen,plan):
    outputs={};comparisons={};predictions={};states={}
    for language in plan['languages']:
        rr=[r for r in rows if r['language']==language];outputs[language]={};predictions[language]={};states[language]={}
        for arm in plan['rankers']+['alignment']:
            p=np.array([r['baseline'][3] for r in rr]) if arm=='alignment' else frozen_scores(rr,arm,frozen['models'][language]);predictions[language][arm]=p.tolist();per=[];all_states=[]
            for seed in plan['model_seeds']:
                mm=[]
                for fold in range(10):
                    ix=[i for i,r in enumerate(rr) if r['seed']==seed and r['fold']==fold];assert len(ix)==4;y=np.array([rr[i]['utility'] for i in ix]);pp=p[ix]
                    values=[float(spearmanr(y,pp).statistic),float(np.sort(y)[-2:].mean()-y[np.argsort(pp,kind='stable')[-2:]].mean())];mm.append(values);all_states.append({'seed':seed,'fold':fold,'spearman':values[0],'top2_regret':values[1]})
                per.append(np.mean(mm,axis=0).tolist())
            outputs[language][arm]={'mean':np.mean(per,axis=0).tolist(),'per_seed':per};states[language][arm]=all_states
        comparisons[language]={}
        for control in ['baseline','alignment']:
            d=np.array(outputs[language]['corrected']['per_seed'])[:,1]-np.array(outputs[language][control]['per_seed'])[:,1];bs=np.random.default_rng(plan['bootstrap']['seed']).choice(d,(plan['bootstrap']['resamples'],len(d)),replace=True).mean(axis=1)
            comparisons[language]['corrected minus '+control]={'regret_difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'role':'primary' if language=='ar' and control=='baseline' else 'secondary'}
    return outputs,comparisons,predictions,states


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.perf_counter();plan=json.loads(Path('research/pud-transfer-protocol.json').read_text());manifest=json.loads(Path('data/pud_sources.json').read_text());frozen=json.loads(Path('research/pud-frozen-rankers.json').read_text())
    for path,h in frozen['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    rows=[];coverage={}
    for language in plan['languages']:
        info=manifest['processed'][language];blob=Path(info['path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==info['sha256'];sentences=[json.loads(line) for line in blob.decode().splitlines()];coverage[language]=[]
        for fold in range(10):
            pool,probe,target=partition(sentences,manifest['folds_source_indices'],fold);assert len(probe)==64
            coverage[language].append({'fold':fold,'candidate_chunks':len(pool),'probe_chunks':len(probe),'target_chunks':len(target),'target_transitions':len(target)*64})
            for seed in plan['model_seeds']:
                torch.manual_seed(seed);rng=np.random.default_rng(seed);model=ByteTransformer();opt=optimizer(model)
                def draw(n):return pool[rng.integers(len(pool),size=n)]
                for _ in range(32):update(model,opt,draw(4))
                follow=[draw(4) for _ in range(3)];pm,po=clone(model,opt);po.zero_grad();byte_loss(pm,probe).backward();pg=flatten({k:p.grad for k,p in pm.named_parameters()});w={k:p.detach().clone() for k,p in model.named_parameters()}
                for candidate in range(4):
                    batch=draw(1);cm,co=clone(model,opt);co.zero_grad();l=byte_loss(cm,batch);l.backward();g={k:p.grad.detach().clone() for k,p in cm.named_parameters()};co.step();delta={k:p.detach()-w[k] for k,p in cm.named_parameters()};dc=flatten(canonical_head(delta));gv=flatten(g)
                    baseline=[float(l.detach()),float(gv.norm()),float(gv.dot(pg)),float(-dc.dot(pg)),float(dc.norm())]
                    rows.append({'language':language,'fold':fold,'seed':seed,'candidate':candidate,'baseline':baseline,'corrected':features(w,g,delta,True).tolist(),'raw':features(w,g,delta).tolist(),'utility':utility(model,opt,batch,follow,target)})
            print(f'{language} fold {fold} complete',flush=True)
    outputs,comparisons,predictions,states=summarize(rows,frozen,plan)
    files=['experiments/pud_transfer.py','experiments/head_invariance.py','experiments/multilingual_readiness.py','experiments/multilingual_policy.py','research/pud-transfer-protocol.json','research/pud-frozen-rankers.json','data/pud_sources.json']+[v['path'] for v in manifest['processed'].values()]
    result={'status':'prospective_frozen_ranker_transfer','outputs':outputs,'comparisons':comparisons,'coverage':coverage,'metric_order':['spearman','top2_regret'],'rows':len(rows),'seconds':time.perf_counter()-start,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    for name,value in [('pud-transfer',result),('pud-transfer-records',rows),('pud-transfer-predictions',predictions),('pud-transfer-state-metrics',states)]:Path('results/'+name+'.json').write_text(json.dumps(value,indent=2)+'\n')
    p=Path('data/evaluation_status.json');ledger=json.loads(p.read_text());ledger['additional_sources']['PUD_ar_en'].update(stage=55,status='all ten PUD folds used in model utility evaluation; exposed for subsequent development');p.write_text(json.dumps(ledger,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
