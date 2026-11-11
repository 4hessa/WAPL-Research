import json
from pathlib import Path
import numpy as np
from experiments.text_pilot import V,LR,counts,loss_grad,step,effects,prepare

def features(w,g,delta,center=False):
    if center:w=w-w.mean(axis=1,keepdims=True)
    return np.array([np.sum(w*g),np.linalg.norm(w*g),np.linalg.norm(delta)/(np.linalg.norm(w)+1e-12)])

def main():
    pools,_=prepare(Path('data/tinyshakespeare.txt').read_text())
    rng=np.random.default_rng(7777);w=rng.normal(0,.01,(V,V));v=np.zeros_like(w)
    for _ in range(24):w,v=step(w,v,counts(pools[0][rng.integers(len(pools[0]))]))
    shifted=w+rng.normal(size=(V,1))
    target=sum((counts(s) for s in pools[4]),np.zeros_like(w))
    follow=[counts(pools[0][rng.integers(len(pools[0]))]) for _ in range(15)]
    records=[]
    for s in pools[0][:32]:
        c=counts(s);l,g=loss_grad(w,c);ls,gs=loss_grad(shifted,c);d=-LR*(.9*v+g);ds=-LR*(.9*v+gs)
        records.append({'loss_error':abs(l-ls),'gradient_max_error':float(np.max(abs(g-gs))), 'utility_max_error':float(np.max(abs(np.array(effects(w,v,c,follow,target))-effects(shifted,v,c,follow,target)))), 'raw_change':abs(features(w,g,d)-features(shifted,gs,ds)).tolist(), 'centered_change':abs(features(w,g,d,True)-features(shifted,gs,ds,True)).tolist()})
    result={'candidates':len(records),'max_loss_error':max(r['loss_error'] for r in records),'max_gradient_error':max(r['gradient_max_error'] for r in records),'max_utility_error':max(r['utility_max_error'] for r in records),'max_raw_feature_changes':np.max([r['raw_change'] for r in records],axis=0).tolist(),'max_centered_feature_changes':np.max([r['centered_change'] for r in records],axis=0).tolist(),'records':records}
    assert result['max_utility_error']<1e-10
    assert max(result['max_centered_feature_changes'])<1e-10
    Path('results/invariance.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='records'})
if __name__=='__main__':main()
