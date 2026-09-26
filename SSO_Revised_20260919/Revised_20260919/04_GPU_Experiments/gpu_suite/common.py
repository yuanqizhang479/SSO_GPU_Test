"""Small shared helpers. No network calls during training."""
import hashlib, json, os, random, platform, sys
from pathlib import Path
import numpy as np
import torch

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + __import__('uuid').uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)

def environment():
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'numpy': np.__version__, 'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(),
            'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}

def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def set_rng_state(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
