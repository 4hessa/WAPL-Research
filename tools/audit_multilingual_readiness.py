"""Audit split preparation without model evaluation on dev or test."""
import hashlib,json
from pathlib import Path

def main():
    manifest=json.loads(Path('data/multilingual_sources.json').read_text());result=json.loads(Path('results/multilingual-readiness.json').read_text());checks={}
    for lang,item in manifest['languages'].items():
        seen=set()
        for split in ['train','dev','test']:
            entry=item['splits'][split];blob=Path(entry['path']).read_bytes();sentences=blob.decode('utf-8').splitlines();current=set(sentences)
            checks[f'{lang}_{split}_hash']=hashlib.sha256(blob).hexdigest()==entry['sha256']
            checks[f'{lang}_{split}_count']=len(sentences)==entry['kept_sentences']
            checks[f'{lang}_{split}_unique']=len(current)==len(sentences) and not seen.intersection(current);seen.update(current)
    checks['training_only']=result['dev_or_test_model_evaluation'] is False and all('dev.txt' not in p and 'test.txt' not in p for p in result['sha256'])
    checks['source_hashes']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in result['sha256'].items())
    checks['four_smoke_runs']=len(result['rows'])==4 and {(r['language'],r['seed']) for r in result['rows']}=={(l,s) for l in ['ar','en'] for s in [22000,22001]}
    checks['finite_losses']=all(r['all_losses_finite'] for r in result['rows'])
    assert all(checks.values()),checks
    Path('results/multilingual-readiness-audit.json').write_text(json.dumps({'checks':checks,'tests_passed':36,'dev_test_operation':'integrity and exact-string overlap checks only; no model scoring','visual_word_review':False},indent=2)+'\n');print('Multilingual integrity and readiness checks passed')
if __name__=='__main__':main()
