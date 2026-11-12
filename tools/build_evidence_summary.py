"""Extract distinct study contrasts without pooling incompatible outcomes."""
import hashlib,json
from pathlib import Path

SPECS=[
 (18,'transformer-pilot','comparison','difference','ci95','Spearman','weight minus baseline','Persuasion','diagnostic',True),
 (24,'horizon-replication','comparisons/A','mae_difference','ci95','utility MAE','horizon-matched simulation minus one-step probe','Alice','new-source diagnostic',False),
 (26,'utility-student','comparisons/baseline minus calibrated','mae_difference','ci95','utility MAE','five-feature student minus calibration','Brown','new-source diagnostic',False),
 (26,'utility-student','comparisons/weight minus baseline','mae_difference','ci95','utility MAE','weight features minus baseline','Brown','secondary',False),
 (29,'pairwise-student','primary','regret_difference','ci95','top-two regret','pairwise minus pointwise regret-tuned','Bryant','prospective new-source primary',False),
 (31,'simple-control','primary','regret_difference','ci95','top-two regret','pairwise minus calibrated alignment','Edgeworth','prospective new-source primary',False),
 (33,'sequential-policy','primary','loss_difference','ci95','final cross-entropy','pairwise minus alignment','Milton','equal-update primary',False),
 (35,'deadline-policy','primary','loss_difference','ci95','final cross-entropy','pairwise minus random','Whitman','equal-deadline primary',False),
 (37,'sparse-policy','comparisons/sparse_pairwise minus random','difference','exploratory_ci95','final cross-entropy','sparse pairwise minus random','Whitman','exposed-source exploratory',False),
 (39,'loss-proxy-policy','comparisons/sparse_loss minus random','difference','exploratory_ci95','final cross-entropy','sparse highest-loss minus random','Whitman','exposed-source exploratory',False),
 (41,'loss-replication','comparisons/sparse_loss minus random','difference','ci95','final cross-entropy','sparse highest-loss minus random','Blake','prospective new-source primary; repeats averaged within seed',False),
]

def main():
    entries=[]
    for report,stem,key,dk,ck,metric,contrast,corpus,design,higher in SPECS:
        p=Path('results')/(stem+'.json');r=json.loads(p.read_text());v=r
        for part in key.split('/'):v=v[part]
        d=v[dk];lo,hi=v[ck];assert lo<=hi
        entries.append(dict(report=report,source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),json_key=key,metric=metric,contrast=contrast,corpus=corpus,design=design,higher_is_better=higher,difference=d,ci95=[lo,hi],interval_includes_zero=lo<=0<=hi))
    out={'scope':'Selected contrasts through report 41; not a meta-analysis or pooled success rate.','limitations':['Different outcomes and controls cannot be pooled.','Several studies reuse text, probe panels, model family, or training labels.','Confidence intervals are conditional on fixed text partitions.','New-source status refers to this project, not an external preregistration.'],'entries':entries}
    Path('research/evidence-summary.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# Evidence map through report 41','','Selected contrasts, not a pooled analysis. See Arabic report 42 for interpretation.','','| Report | Source | Outcome | Contrast | Difference | 95% interval | Design |','|---|---|---|---|---:|---|---|']
    for e in entries:
        lines.append(f'| {e["report"]} | [{e["corpus"]}](../{e["source"]}) | {e["metric"]} | {e["contrast"]} | {e["difference"]:.8f} | [{e["ci95"][0]:.8f}, {e["ci95"][1]:.8f}] | {e["design"]} |')
    lines+=['','A negative difference favors the first method for MAE, regret and cross-entropy. A positive difference favors it for Spearman. Intervals including zero do not establish equivalence. Secondary and exploratory intervals are not additional confirmatory wins.','', 'Regenerate with `python -m tools.build_evidence_summary`. Source-result hashes are recorded in the JSON map; historical code hashes inside each source belong to its original Git revision.']
    Path('research/evidence-summary.md').write_text('\n'.join(lines)+'\n');print(f'Extracted {len(entries)} contrasts without pooling')
if __name__=='__main__':main()
