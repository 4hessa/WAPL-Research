"""Restore exact experiment inputs using saved provenance and SHA-256 checks."""
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    cache = {}
    for metadata, filename in [('source.json', 'tinyshakespeare.txt'),
                               ('moby_source.json', 'moby_dick.txt'),
                               ('emma_source.json', 'emma.txt'),
                               ('persuasion_source.json', 'persuasion.txt'),
                               ('thursday_source.json', 'thursday.txt'),
                               ('alice_source.json', 'alice.txt'),
                               ('brown_source.json', 'brown.txt'),
                               ('bryant_source.json', 'bryant.txt'),
                               ('edgeworth_source.json', 'edgeworth.txt'),
                               ('milton_source.json', 'milton.txt')]:
        info = json.loads((ROOT / 'data' / metadata).read_text())
        expected = info.get('text_sha256', info.get('sha256'))
        destination = ROOT / 'data' / filename
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f'Refusing to overwrite mismatched file: {filename}')
            print(f'Verified existing {filename}')
            continue
        url = info['url']
        if url not in cache:
            with urllib.request.urlopen(url, timeout=60) as response:
                cache[url] = response.read()
        blob = cache[url]
        if 'archive_sha256' in info:
            assert hashlib.sha256(blob).hexdigest() == info['archive_sha256'], 'Archive hash mismatch'
        member = info.get('archive_member', info.get('member'))
        if member:
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                blob = archive.read(member)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError(f'Source hash mismatch: {filename}')
        destination.write_bytes(blob)
        print(f'Downloaded and verified {filename}')


if __name__ == '__main__':
    main()
