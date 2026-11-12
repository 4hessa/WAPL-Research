"""Restore pinned UD text splits, keeping original downloads outside Git."""
import hashlib,json,urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES={'ar':('UD_Arabic-PADT','ar_padt'),'en':('UD_English-EWT','en_ewt')}
TAG='r2.15'

def extract_text(blob):
    return [line[len('# text = '):] for line in blob.decode('utf-8').splitlines() if line.startswith('# text = ')]

def sha(blob):return hashlib.sha256(blob).hexdigest()

def main():
    manifest_path=ROOT/'data/multilingual_sources.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    raw=ROOT/'data/ud_raw';raw.mkdir(exist_ok=True)
    jobs=[]
    for language,(repo,prefix) in SOURCES.items():
        for key,filename in [('readme','README.md')]+[(split,f'{prefix}-ud-{split}.conllu') for split in ['train','dev','test']]:
            url=f'https://raw.githubusercontent.com/UniversalDependencies/{repo}/{TAG}/{filename}'
            jobs.append((language,key,url,raw/f'{language}_{key}.raw'))
    def fetch(job):
        language,key,url,path=job
        if path.exists():blob=path.read_bytes()
        else:
            blob=urllib.request.urlopen(url,timeout=60).read();path.write_bytes(blob)
        if previous:assert sha(blob)==previous['languages'][language]['files'][key]['sha256'],f'Hash mismatch: {language}/{key}'
        return language,key,url,blob
    with ThreadPoolExecutor(max_workers=4) as pool:downloaded=list(pool.map(fetch,jobs))
    manifest={'release':TAG,'extraction':'CoNLL-U # text comments only; exact string dedup in train/dev/test order; no Unicode normalization','test_model_evaluation':False,'languages':{}}
    for language,(repo,prefix) in SOURCES.items():
        files={key:(url,blob) for lang,key,url,blob in downloaded if lang==language}
        readme=files['readme'][1].decode('utf-8');licenses=[line for line in readme.splitlines() if line.startswith('License:')]
        assert len(licenses)==1
        item={'repository':f'https://github.com/UniversalDependencies/{repo}','declared_license':licenses[0].split(':',1)[1].strip(),'license_scope':'Publisher declaration; rights in source text are not independently cleared','files':{},'splits':{}}
        for key,(url,blob) in files.items():item['files'][key]={'url':url,'sha256':sha(blob),'bytes':len(blob)}
        seen=set()
        for split in ['train','dev','test']:
            sentences=extract_text(files[split][1]);assert sentences
            kept=[]
            for sentence in sentences:
                if sentence and sentence not in seen:seen.add(sentence);kept.append(sentence)
            text=('\n'.join(kept)+'\n').encode('utf-8');path=ROOT/f'data/ud_{language}_{split}.txt';path.write_bytes(text)
            item['splits'][split]={'source_sentences':len(sentences),'kept_sentences':len(kept),'removed_exact_duplicates_or_empty':len(sentences)-len(kept),'utf8_bytes':len(text),'sha256':sha(text),'path':str(path.relative_to(ROOT))}
        manifest['languages'][language]=item
    if previous:assert manifest==previous,'Manifest changed; investigate instead of silently replacing'
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({lang:{'license':v['declared_license'],'splits':v['splits']} for lang,v in manifest['languages'].items()},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
