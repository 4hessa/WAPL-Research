"""Verify stored post-hoc sensitivity results and deterministic reproduction."""
from pathlib import Path
import json, hashlib, subprocess, os
root=Path(__file__).resolve().parents[1];os.chdir(root)
p=Path('results/pud-sensitivity.json');before=p.read_bytes()
subprocess.run([__import__('sys').executable,'-m','experiments.pud_sensitivity'],check=True,stdout=subprocess.DEVNULL)
assert before==p.read_bytes()
r=json.loads(before)
for path,digest in r['sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
for lang,v in r['results'].items():
 d=v['delta_by_seed_fold'];assert len(d)==8 and all(len(row)==10 for row in d)
 assert abs(sum(map(sum,d))/80-v['mean_difference'])<1e-18
 assert sum(v['state_counts'].values())==80
 assert {(c['seed'],c['fold']) for c in v['choices']}=={(s,f) for s in r['seed_order'] for f in r['fold_order']}
 for i,x in enumerate(v['leave_one_seed_out']):assert abs(x-sum(sum(row) for j,row in enumerate(d) if j!=i)/70)<1e-18
 for i,x in enumerate(v['leave_one_fold_out']):assert abs(x-sum(sum(y for j,y in enumerate(row) if j!=i) for row in d)/72)<1e-18
 for arm,c in v['choice_comparisons'].items():
  pairs=[(set(x['selected_positions']['corrected']),set(x['selected_positions'][arm])) for x in v['choices']]
  assert c['identical_top2_states']==sum(a==b for a,b in pairs)
  assert c['mean_top2_overlap_fraction']==sum(len(a&b)/2 for a,b in pairs)/80
Path('results/pud-sensitivity-audit.json').write_text(json.dumps({'status':'passed','byte_identical_rerun':True,'source_hashes_verified':True,'grids_and_omission_means_verified':True,'choice_overlap_verified':True,'new_transformer_runs':0,'new_hypothesis_tests':0,'result_sha256':hashlib.sha256(before).hexdigest()},indent=2)+'\n')
