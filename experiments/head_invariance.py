"""Output-softmax shift diagnostic; training data only, float64 arithmetic."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from experiments.multilingual_readiness import ByteTransformer,byte_chunks,byte_loss
from experiments.multilingual_policy import optimizer,clone,update


def canonical_head(tensors):
    out={k:v.detach().clone() for k,v in tensors.items()}
    out['head.weight']-=out['head.weight'].mean(dim=0,keepdim=True)
    out['head.bias']-=out['head.bias'].mean()
    return out

def flatten(tensors):return torch.cat([v.flatten() for v in tensors.values()])

def features(weights,gradients,delta,corrected=False):
    if corrected:weights=canonical_head(weights);delta=canonical_head(delta)
    w,g,d=map(flatten,[weights,gradients,delta])
    return torch.stack([w.dot(g),(w*g).norm(),d.norm()/(w.norm()+1e-12)])

def shifted_copy(model,opt,shift):
    m,o=clone(model,opt)
    with torch.no_grad():m.head.weight.add_(shift[None]);m.head.bias.add_(.25)
    return m,o

def candidate_details(model,opt,candidate):
    m,o=clone(model,opt);w={k:p.detach().clone() for k,p in m.named_parameters()};o.zero_grad();l=byte_loss(m,candidate);l.backward();g={k:p.grad.detach().clone() for k,p in m.named_parameters()};o.step();d={k:p.detach()-w[k] for k,p in m.named_parameters()}
    return float(l.detach()),g,features(w,g,d),features(w,g,d,True)

def utility(model,opt,candidate,follow,monitor):
    inc,io=clone(model,opt);skip,so=clone(model,opt);update(inc,io,candidate)
    for batch in follow:update(inc,io,batch);update(skip,so,batch)
    with torch.no_grad():return float(byte_loss(skip,monitor)-byte_loss(inc,monitor))

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);rows=[]
    for language in ['ar','en']:
        pool=byte_chunks(Path(f'data/ud_{language}_train.txt').read_bytes());monitor=pool[-16:]
        for seed in [24000,24001]:
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=ByteTransformer().double();o=optimizer(m)
            for _ in range(16):update(m,o,pool[rng.integers(len(pool),size=4)])
            shift=torch.tensor(rng.normal(0,.5,size=32),dtype=torch.float64);sm,so=shifted_copy(m,o,shift)
            with torch.no_grad():prob_error=float((m(monitor[:,:-1]).softmax(-1)-sm(monitor[:,:-1]).softmax(-1)).abs().max())
            follow=[pool[rng.integers(len(pool),size=4)] for _ in range(3)]
            for i in range(4):
                candidate=pool[rng.integers(len(pool),size=1)];l,g,raw,corrected=candidate_details(m,o,candidate);ls,gs,raws,correcteds=candidate_details(sm,so,candidate)
                u=utility(m,o,candidate,follow,monitor);us=utility(sm,so,candidate,follow,monitor)
                rows.append(dict(language=language,seed=seed,candidate=i,probability_error=prob_error,loss_error=abs(l-ls),gradient_error=float((flatten(g)-flatten(gs)).abs().max()),utility_error=abs(u-us),utility=u,shifted_utility=us,raw=raw.tolist(),shifted_raw=raws.tolist(),corrected=corrected.tolist(),shifted_corrected=correcteds.tolist()))
    maxima={k:max(r[k] for r in rows) for k in ['probability_error','loss_error','gradient_error','utility_error']}
    for name in ['raw','corrected']:maxima[name+'_feature_changes']=np.max([np.abs(np.array(r[name])-r['shifted_'+name]) for r in rows],axis=0).tolist()
    assert max(maxima[k] for k in ['probability_error','loss_error','gradient_error','utility_error'])<1e-8
    assert max(maxima['corrected_feature_changes'])<1e-8
    files=['experiments/head_invariance.py','experiments/multilingual_readiness.py','experiments/multilingual_policy.py','reports/46_head_invariance_protocol_ar.md','data/ud_ar_train.txt','data/ud_en_train.txt']
    result={'status':'training_only_symmetry_diagnostic_not_prediction_efficacy','rows':rows,'maxima':maxima,'feature_order':['weight_dot_gradient','elementwise_product_norm','relative_update_norm'],'tolerance':1e-8,'dtype':'float64','sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    Path('results/head-invariance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(maxima,indent=2))
if __name__=='__main__':main()
