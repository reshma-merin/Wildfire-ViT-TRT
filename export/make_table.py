"""Turn results/results.jsonl into a markdown table for the README / paper.
   python export/make_table.py results/results.jsonl
"""
import json, sys

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "results/results.jsonl")]
rows.sort(key=lambda r: (r["batch"], r["p50_ms"]))
print("| Backend | Batch | p50 (ms) | p95 (ms) | p99 (ms) | Throughput (img/s) |")
print("|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r['backend']} | {r['batch']} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | "
          f"{r['p99_ms']:.2f} | {r['throughput_ips']:.1f} |")