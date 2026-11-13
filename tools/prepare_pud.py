"""Prepare parallel PUD folds without scoring any new-source model outcome."""
import hashlib,json,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCES={'ar':('Arabic','ar_pud'),'en':('English','en_pud')}

def parse_sentences(blob):
    result=[]
    for block in blob.decode('utf-8').strip().split('\n\n'):
        fields={}
        for line in block.splitlines():
            if line.startswith('# sent_id = '):fields['id']=line[len('# sent_id = '):]
            if line.startswith('# text = '):fields['text']=line[len('# text = '):]
        if fields:
            if set(fields)!={'id','text'}:raise ValueError('Missing sentence ID or text')
            result.append(fields)
    return result

def main():
    raw=ROOT/'data/pud_raw';raw.mkdir(exist_ok=True);manifest_path=ROOT/'data/pud_sources.json';prior=json.loads(manifest_path.read_text()) if manifest_path.exists() else None;jobs=[]
    for lang,(name,prefix) in SOURCES.items():
        for kind,filename in [('readme','README.md'),('corpus',prefix+'-ud-test.conllu')]:jobs.append((lang,kind,f'https://raw.githubusercontent.com/UniversalDependencies/UD_{name}-PUD/r2.15/{filename}'))
    def fetch(job):
        lang,kind,url=job;p=raw/f'{lang}_{kind}.raw';blob=p.read_bytes() if p.exists() else urllib.request.urlopen(url,timeout=60).read();h=hashlib.sha256(blob).hexdigest()
        if prior:assert h==prior['sources'][lang]['files'][kind]['sha256']
        p.write_bytes(blob);return lang,kind,url,blob
    with ThreadPoolExecutor(max_workers=4) as pool:downloaded=list(pool.map(fetch,jobs))
    sources={};sentences={};old_sets={}
    for lang in SOURCES:
        files={kind:(url,blob) for l,kind,url,blob in downloaded if l==lang};readme=files['readme'][1].decode();licenses=[v.split(':',1)[1].strip() for v in readme.splitlines() if v.startswith('License:')];assert len(licenses)==1
        sources[lang]={'declared_license':licenses[0],'files':{k:{'url':url,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)} for k,(url,blob) in files.items()}}
        sentences[lang]=parse_sentences(files['corpus'][1]);assert len(sentences[lang])==1000
        old_sets[lang]=set()
        for split in ['train','dev','test']:old_sets[lang].update((ROOT/f'data/ud_{lang}_{split}.txt').read_text().splitlines())
    assert [r['id'] for r in sentences['ar']]==[r['id'] for r in sentences['en']], 'Parallel sentence IDs do not match'
    seen={l:set() for l in SOURCES};kept=[];excluded=[];reasons={'prior_exact_overlap':0,'within_pud_exact_duplicate':0,'empty':0}
    for i in range(1000):
        texts={l:sentences[l][i]['text'] for l in SOURCES};reason='empty' if any(not t for t in texts.values()) else 'prior_exact_overlap' if any(texts[l] in old_sets[l] for l in SOURCES) else 'within_pud_exact_duplicate' if any(texts[l] in seen[l] for l in SOURCES) else None
        if reason:excluded.append({'source_index':i,'reason':reason});reasons[reason]+=1;continue
        kept.append(i)
        for l in SOURCES:seen[l].add(texts[l])
    permutation=np.random.default_rng(5400).permutation(kept);folds=[v.tolist() for v in np.array_split(permutation,10)];processed={}
    for lang in SOURCES:
        rows=[{'source_index':i,'id':sentences[lang][i]['id'],'text':sentences[lang][i]['text']} for i in kept];blob=('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n').encode();p=raw/f'{lang}_sentences.jsonl';p.write_bytes(blob);processed[lang]={'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(blob).hexdigest(),'sentences':len(rows)}
    refs={f'data/ud_{l}_{s}.txt':hashlib.sha256((ROOT/f'data/ud_{l}_{s}.txt').read_bytes()).hexdigest() for l in SOURCES for s in ['train','dev','test']}
    out={'release':'r2.15','sources':sources,'original_parallel_sentences':1000,'kept_parallel_sentences':len(kept),'exclusions':excluded,'exclusion_counts':reasons,'processed':processed,'fold_seed':5400,'folds_source_indices':folds,'fold_sentence_counts':[len(f) for f in folds],'overlap_reference_sha256':refs,'operations':'exact text filtering jointly across languages; metadata and fold preparation only','new_source_model_evaluation':False,'limitations':['Parallel translations are not independent language samples.','Exact-string filtering does not exclude near duplicates or shared documents.','Corpus is officially test-only; prospective research should use ten-fold cross-validation rather than claim an official benchmark score.']}
    if prior:assert out==prior,'Prepared manifest changed'
    manifest_path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:out[k] for k in ['kept_parallel_sentences','exclusion_counts','fold_sentence_counts']},indent=2))
if __name__=='__main__':main()
