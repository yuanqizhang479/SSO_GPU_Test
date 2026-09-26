#!/usr/bin/env python3
"""Read-only request/group accounting for the publicly linked benchmark archive.

No model inference. Counts alone do not identify Xie Table 3's actual evaluation path.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

URL = "https://raw.githubusercontent.com/richardodliu/Megatron_benchmark/main/benchmark.zip"
SHA = "904dfc3c2ba874d92d6dfb521ae2fe919951cbbd6a63e69a8ecdaff63f3f5568"
TASKS = ["csqa/rc_0shot", "piqa/rc_0shot", "arc_easy/rc_0shot", "arc_challenge/rc_5shot",
         "hellaswag/rc_5shot", "boolq/rc_0shot", "winogrande/rc_0shot", "lambada/ppl_0shot"]


def inspect(archive):
    blob = archive.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != SHA:
        raise ValueError(f"Archive changed or corrupt: expected {SHA}, got {digest}; do not mix versions")
    report = {"source": URL, "sha256": digest, "model_inference": False, "tasks": {}}
    with zipfile.ZipFile(archive) as z:
        for task in TASKS:
            name = "benchmark/" + task + "/requests.jsonl.gz"
            rows = [json.loads(row) for row in gzip.decompress(z.read(name)).splitlines()]
            ids = {row.get("doc_id", row.get("idx", 0)) for row in rows}
            offset = {x for x in ids if isinstance(x, int) and x >= 1_000_000}
            main = ids - offset
            report["tasks"][task] = {
                "requests": len(rows), "groups_if_using_raw_doc_id": len(ids),
                "primary_ids": len(main), "offset_ids": len(offset),
                "all_offset_ids_have_primary_pair": all(x - 1_000_000 in main for x in offset),
                "offset_context_examples": sorted({str(row.get("request", {}).get("context", ""))
                    for row in rows if row.get("doc_id") in offset})[:3],
            }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("benchmark.zip"))
    p.add_argument("--download", action="store_true", help="download only if the path does not already exist")
    p.add_argument("--out", type=Path, default=Path("benchmark_inventory.json"))
    args = p.parse_args()
    if args.download and not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            args.archive.write_bytes(r.read())
    report = inspect(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for name, task in report["tasks"].items():
        print(name, "primary=", task["primary_ids"], "offset=", task["offset_ids"],
              "raw_groups=", task["groups_if_using_raw_doc_id"])


if __name__ == "__main__":
    main()
