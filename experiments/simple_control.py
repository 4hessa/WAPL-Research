"""Independent comparison of pairwise utility with simple calibrated alignment."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from experiments.pairwise_student import fit_rankers,generate,regret

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=json.loads(Path('results/student-records.json').read_text())
    t=time.perf_counter();models,selection=fit_rankers(source);fit_seconds=time.perf_counter()-t
    t=time.perf_counter();rows=generate("data/edgeworth.txt", range(16000,16016));generation_seconds=time.perf_counter()-t
    x=np.array([r['baseline'] for r in rows]);y=np.array([r['target'] for r in rows]);predictions={};outputs={}
    for arm in list(models)+['teacher']:
        if arm=='teacher':p=np.array([r['teacher'] for r in rows])
        else:
            sc,m=models[arm];p=m.predict(sc.transform(x[:,3:4] if arm=='calibrated' else x))
        predictions[arm]=p.tolist();per=[]
        for seed in range(16000,16016):
            states=[]
            for cp in [12,48]:
                ix=[i for i,r in enumerate(rows) if r['seed']==seed and r['checkpoint']==cp]
                states.append([regret(y[ix],p[ix]),float(spearmanr(y[ix],p[ix]).statistic)])
            per.append(np.mean(states,axis=0).tolist())
        outputs[arm]={'mean':np.mean(per,axis=0).tolist(),'per_seed':per}
    delta=np.array(outputs['pairwise']['per_seed'])[:,0]-np.array(outputs['calibrated']['per_seed'])[:,0]
    bs=np.random.default_rng(3001).choice(delta,(10000,16),replace=True).mean(axis=1)
    files=['experiments/simple_control.py','experiments/pairwise_student.py','experiments/utility_student.py','experiments/transformer_pilot.py','experiments/transformer_horizon.py','reports/30_simple_control_protocol_ar.md','results/student-records.json','data/edgeworth.txt']
    out={'calibrated_slope_standardized':float(models['calibrated'][1].coef_[0]),'outputs':outputs,'selection':selection,'primary':{'regret_difference':float(delta.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist()},'metric_order':['top2_regret','spearman'],'rows':len(rows),'fit_seconds':fit_seconds,'generation_seconds':generation_seconds,'edgeworth_target_consumed':True,'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    for name,value in [('simple-control',out),('simple-control-records',rows),('simple-control-predictions',predictions)]:Path('results/'+name+'.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
