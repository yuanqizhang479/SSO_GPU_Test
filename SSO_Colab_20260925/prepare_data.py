"""Pinned HF text streams -> document-separated uint32 token files and auditable manifest.
No remote executable dataset code is trusted. Validation is assigned by content hash.
OLMo's seven sources are prepared separately by TOKEN quota, then interleaved in training.
"""
import argparse, hashlib, json, math, os
from pathlib import Path
import numpy as np

def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def write_data(config, out, streams, encode, vocab_size, eos_id, metadata):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists(): raise FileExistsError('Refusing to overwrite a prepared dataset')
    if any(out.iterdir()): raise FileExistsError('Use a new empty output directory; partial data are not reusable')
    seen = set(); summary = []; total_weight = sum(s['weight'] for s in config['sources'])
    for source, stream in zip(config['sources'], streams):
        name = source['name']; quotas = {
            'train': math.ceil(config['train_tokens'] * source['weight'] / total_weight),
            'val': max(config.get('min_val_source_tokens', 4096), math.ceil(config['val_tokens'] * source['weight'] / total_weight)),
            'test': max(config.get('min_val_source_tokens', 4096), math.ceil(config.get('test_tokens',config['val_tokens']) * source['weight'] / total_weight))}
        counts = {'train': 0, 'val': 0, 'test': 0}; docs = {'train': 0, 'val': 0, 'test': 0}; skipped = 0
        handles = {s: open(out / (name + '_' + s + '.bin'), 'wb') for s in counts}
        try:
            with open(out / (name + '_documents.jsonl'), 'w', encoding='utf8') as df:
                for index, row in enumerate(stream):
                    text = row.get('text')
                    if not isinstance(text, str) or not text.strip(): continue
                    # Exact normalized-text deduplication across every source and split.
                    text = text.replace('\r\n', '\n').strip()
                    content_sha = hashlib.sha256(text.encode('utf8')).hexdigest()
                    if content_sha in seen: skipped += 1; continue
                    seen.add(content_sha)
                    bucket = int(content_sha[:16], 16) % 10000
                    width = config.get('validation_permyriad', 100)
                    split = 'val' if bucket < width else ('test' if bucket < 2*width else 'train')
                    if counts[split] >= quotas[split]:
                        if all(counts[s] >= quotas[s] for s in counts): break
                        continue
                    tokens = list(encode(text)) + [eos_id]
                    if tokens and (min(tokens) < 0 or max(tokens) >= vocab_size): raise ValueError('Token outside vocabulary')
                    start = counts[split]
                    # Whole documents are kept; quotas may be exceeded by one document.
                    np.asarray(tokens, dtype='<u4').tofile(handles[split]); counts[split] += len(tokens); docs[split] += 1
                    df.write(json.dumps({'source': name, 'source_row': index,
                        'source_id': str(row.get('id', row.get('url', index))), 'content_sha256': content_sha,
                        'split': split, 'token_start': start, 'token_end': counts[split]}, ensure_ascii=False) + '\n')
                    if all(counts[s] >= quotas[s] for s in counts): break
            if not all(counts[s] >= quotas[s] for s in counts):
                raise RuntimeError(f'Stream {name} exhausted before quotas: {counts} vs {quotas}')
        finally:
            for f in handles.values(): f.close()
        summary.append({'name': name, 'weight': source['weight'] / total_weight, 'tokens': counts,
                        'documents': docs, 'duplicates_skipped': skipped, 'requested_quota': quotas})
    files = {p.name: {'sha256': digest_file(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir())}
    manifest = {'schema': 1, 'dataset_label': config['dataset_label'], 'config': config,
        'vocab_size': vocab_size, 'eos_token_id': eos_id, 'dtype': '<u4', 'sources': summary,
        'files': files, 'metadata': metadata, 'split_rule': 'sha256(normalized text)[:16] mod 10000',
        'sampling': 'buffered source-wise shuffle, finite prefix until quotas; not uniform across full corpus',
        'deduplication': 'exact normalized text across all accepted and visited sources; no near-dedup guarantee'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'sources': summary}, indent=2))
    return manifest

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config'); ap.add_argument('--replay-manifest'); ap.add_argument('--out', required=True)
    ap.add_argument('--cache', default=None); args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    import datasets, transformers, huggingface_hub
    if bool(args.config)==bool(args.replay_manifest): raise ValueError('Provide exactly one of --config or --replay-manifest')
    if args.replay_manifest:
        old=json.loads(Path(args.replay_manifest).read_text());c=old['config']
        revisions={s['name']:s['resolved_revision'] for s in old['metadata']['resolved_sources']}
        for source in c['sources']: source['revision']=revisions[source['name']]
        c['tokenizer_revision']=old['metadata']['tokenizer_revision']
    else: c = json.loads(Path(args.config).read_text())
    api = HfApi()
    tokenizer_id = c['tokenizer']; tr = api.model_info(tokenizer_id, revision=c.get('tokenizer_revision', 'main')).sha
    tok = AutoTokenizer.from_pretrained(tokenizer_id, revision=tr, cache_dir=args.cache, trust_remote_code=False)
    if tok.eos_token_id is None: raise ValueError('Tokenizer must define EOS')
    resolved = []; streams = []
    for s in c['sources']:
        revision = api.dataset_info(s['repo'], revision=s.get('revision', 'main')).sha
        resolved.append({**s, 'resolved_revision': revision})
        kw = {'path': s['repo'], 'split': s.get('split', 'train'), 'revision': revision, 'streaming': True, 'cache_dir': args.cache}
        if s.get('subset'): kw['name'] = s['subset']
        stream = load_dataset(**kw)
        stream = stream.shuffle(seed=c.get('shuffle_seed',20260919), buffer_size=c.get('shuffle_buffer_size',10000))
        streams.append(stream)
    metadata = {'resolved_sources': resolved, 'tokenizer': tokenizer_id, 'tokenizer_revision': tr,
                'versions': {'datasets': datasets.__version__, 'transformers': transformers.__version__, 'huggingface_hub': huggingface_hub.__version__}}
    write_data(c, args.out, streams, lambda x: tok.encode(x, add_special_tokens=False), len(tok), tok.eos_token_id, metadata)
    tok.save_pretrained(Path(args.out) / 'tokenizer')
    # Include local tokenizer hashes in the final manifest after saving.
    p = Path(args.out); m = json.loads((p/'manifest.json').read_text())
    m['tokenizer_files'] = {str(f.relative_to(p)): digest_file(f) for f in (p/'tokenizer').rglob('*') if f.is_file()}
    (p/'manifest.json').write_text(json.dumps(m, ensure_ascii=False, indent=2)+'\n')

if __name__ == '__main__': main()
