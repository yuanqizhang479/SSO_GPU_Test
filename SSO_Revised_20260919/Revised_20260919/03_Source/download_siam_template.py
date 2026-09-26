#!/usr/bin/env python3
"""Download the complete official SIAM macro bundle (no manuscript upload)."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import urllib.request
import zipfile

URL = 'https://epubs.siam.org/pb-assets/macros/standard/siamart_251216.zip'
REQUIRED = {'siamart251216.cls', 'siamplain.bst', 'docsiamart.tex',
            'docsiamart.pdf', 'references.bib', 'ex_article.tex',
            'ex_supplement.tex', 'ex_shared.tex'}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', type=Path, help='Use an official bundle downloaded manually')
    args = ap.parse_args()
    if args.zip:
        blob = args.zip.read_bytes()
        source = str(args.zip)
    else:
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                blob = response.read()
        except Exception as exc:
            raise SystemExit('Official download failed: %s\nDownload the complete standard macro ZIP from https://epubs.siam.org/journal-authors#macros and run python download_siam_template.py --zip PATH.\nFor an immediate portable build: python build_pdfs.py --format review' % exc)
        source = URL
    archive = zipfile.ZipFile(io.BytesIO(blob))
    names = {PurePosixPath(name).name for name in archive.namelist()}
    missing = REQUIRED - names
    if missing:
        raise SystemExit('Incomplete distribution; missing: ' + ', '.join(sorted(missing)))
    target = Path(__file__).resolve().parent / 'vendor' / 'siam_template'
    target.mkdir(parents=True, exist_ok=True)
    for member in archive.infolist():
        rel = PurePosixPath(member.filename)
        if rel.is_absolute() or '..' in rel.parts:
            raise SystemExit('Unsafe ZIP member: ' + member.filename)
        if member.is_dir():
            continue
        destination = target.joinpath(*rel.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(member))
    (target / 'bundle_provenance.json').write_text(json.dumps({
        'source': source, 'official_url': URL,
        'sha256': hashlib.sha256(blob).hexdigest(),
        'note': 'Complete distribution retained; official files are unmodified.'
    }, indent=2) + '\n')
    print('Installed complete SIAM bundle under', target)

if __name__ == '__main__':
    main()
