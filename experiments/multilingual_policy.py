"""Arabic-primary frozen policy comparison; all complete test chunks evaluated."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.multilingual_readiness import ByteTransformer,byte_chunks,byte_loss

SEEDS=range(23000,23008)
ARMS=['random','sparse_loss']

def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
def clone(model,opt):
    m=copy.deepcopy(model);o=optimizer(m);o.load_state_dict(copy.deepcopy(opt.state_dict()));return m,o

def update(model,opt,batch):
    opt.zero_grad();l=byte_loss(model,batch);l.backward();opt.step()

@torch.no_grad()
def evaluate(model,data,batch_size=64):
    total=0.
    for start in range(0,len(data),batch_size):
        batch=data[start:start+batch_size];total+=float(byte_loss(model,batch))*len(batch)
    return total/len(data)

@torch.no_grad()
def scores(model,candidates):
    logits=model(candidates[:,:-1]);return torch.nn.functional.cross_entropy(logits.reshape(-1,256),candidates[:,1:].reshape(-1),reduction='none').reshape(len(candidates),64).mean(dim=1)

def run_policy(base,opt,pool,ids,random_choices,arm,mode,clock=time.perf_counter,budget=3.,fixed_updates=256):
    if arm not in ARMS or mode not in ['updates','time']:raise ValueError('Unknown policy or budget')
    model,opt=clone(base,opt);accepted=[];attempted=0;last=0.;start=clock()
    limit=fixed_updates if mode=='updates' else len(ids)
    for step in range(limit):
        if mode=='time' and clock()-start>=budget:break
        attempted+=1;trial,to=clone(model,opt);candidates=pool[ids[step]]
        choice=int(torch.argmax(scores(trial,candidates))) if arm=='sparse_loss' and step%16==0 else int(random_choices[step])
        update(trial,to,candidates[choice:choice+1]);completed=clock()-start
        if mode=='time' and completed>budget:break
        model,opt=trial,to;last=completed;accepted.append(choice)
    actual=clock()-start
    return model,dict(updates=len(accepted),attempted=attempted,choices=accepted,last_accepted_seconds=last,actual_seconds=actual,overrun_seconds=max(0.,actual-budget) if mode=='time' else 0.,schedule_exhausted=len(accepted)==len(ids))

def summarize(rows):
    outputs={};comparisons={}
    for language in ['ar','en']:
        outputs[language]={};comparisons[language]={}
        for mode in ['updates','time']:
            outputs[language][mode]={}
            for arm in ARMS:
                rr=[r for r in rows if (r['language'],r['mode'],r['arm'])==(language,mode,arm)];per=[]
                for seed in SEEDS:
                    ss=sorted([r for r in rr if r['seed']==seed],key=lambda r:r['repeat']);assert [r['repeat'] for r in ss]==([0,1] if mode=='time' else [0])
                    per.append(float(np.mean([r['final_loss'] for r in ss])))
                outputs[language][mode][arm]={k:float(np.mean([r[k] for r in rr])) for k in ['final_loss','updates','actual_seconds','overrun_seconds']}
                outputs[language][mode][arm]['per_seed_mean_loss']=per
            oo=outputs[language][mode];d=np.array(oo['sparse_loss']['per_seed_mean_loss'])-np.array(oo['random']['per_seed_mean_loss']);bs=np.random.default_rng(4401).choice(d,(10000,8),replace=True).mean(axis=1)
            comparisons[language][mode]={'difference':float(d.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'role':'primary' if language=='ar' and mode=='time' else 'secondary_diagnostic'}
    return outputs,comparisons

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);manifest=json.loads(Path('data/multilingual_sources.json').read_text());rows=[];data_info={}
    for language in ['ar','en']:
        blobs={}
        for split in ['train','test']:
            e=manifest['languages'][language]['splits'][split];blob=Path(e['path']).read_bytes();assert hashlib.sha256(blob).hexdigest()==e['sha256'];blobs[split]=blob
        pool=byte_chunks(blobs['train']);target=byte_chunks(blobs['test']);data_info[language]={'train_chunks':len(pool),'test_chunks':len(target),'test_transitions':len(target)*64,'discarded_test_tail_bytes':len(blobs['test'])%65}
        for seed in SEEDS:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);base=ByteTransformer();bo=optimizer(base)
            for _ in range(32):update(base,bo,pool[rng.integers(len(pool),size=4)])
            ids=rng.integers(len(pool),size=(20000,8));choices=rng.integers(8,size=20000);h=hashlib.sha256(ids.tobytes()+choices.tobytes()).hexdigest();initial=evaluate(base,target)
            modes=['updates','time'] if seed%2==0 else ['time','updates']
            for mode in modes:
                for repeat in range(2 if mode=='time' else 1):
                    arms=ARMS if (seed+repeat)%2==0 else ARMS[::-1]
                    for arm in arms:
                        model,record=run_policy(base,bo,pool,ids,choices,arm,mode)
                        final=evaluate(model,target);record.update(language=language,seed=seed,mode=mode,repeat=repeat,arm=arm,initial_loss=initial,final_loss=final,schedule_sha256=h);rows.append(record)
                        assert not record['schedule_exhausted']
                        assert (record['last_accepted_seconds']<=3.) if mode=='time' else (record['updates']==256)
            print(f'{language} seed {seed} completed',flush=True)
    outputs,comparisons=summarize(rows)
    files=['experiments/multilingual_policy.py','experiments/multilingual_readiness.py','reports/44_multilingual_protocol_ar.md','data/multilingual_sources.json']+[f'data/ud_{l}_{s}.txt' for l in ['ar','en'] for s in ['train','test']]
    result={'outputs':outputs,'comparisons':comparisons,'data_info':data_info,'budget_seconds':3.,'fixed_updates':256,'timing_repeats':2,'test_evaluated':['ar','en'],'dev_evaluated':False,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    Path('results/multilingual-policy.json').write_text(json.dumps(result,indent=2)+'\n');Path('results/multilingual-policy-records.json').write_text(json.dumps(rows,indent=2)+'\n')
    Path('data/evaluation_status.json').write_text(json.dumps({'stage':45,'datasets':{l:{'train':'used for training','dev':'prepared only; no model evaluation in stages 43–45','test':'evaluated in stage 45; exposed for future development'} for l in ['ar','en']},'preparation_manifest_status':'Historical snapshot from stage 43; see this ledger for current exposure'},indent=2)+'\n');print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
