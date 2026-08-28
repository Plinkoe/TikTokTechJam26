import json
import csv
import os

LOGPATH = os.path.join("run_logs", "iterations.jsonl")

def summarize(inpath=LOGPATH, out_json="run_logs/summary.json", out_csv="run_logs/summary.csv"):
    if not os.path.exists(inpath):
        print("No run log found at", inpath); return
    records = []
    with open(inpath, encoding="utf-8") as f:
        for ln in f:
            records.append(json.loads(ln))

    summary = []
    for i, r in enumerate(records, 1):
        m = r.get("params", {}).get("model") or ("fm" if "k" in r.get("params", {}) else "fm")
        valid = r.get("metrics", {}).get("valid")
        test = r.get("metrics", {}).get("test")
        summary.append({
            "iter": i,
            "timestamp": r.get("timestamp"),
            "model": m,
            "hypothesis": r.get("hypothesis"),
            "valid_primary": valid.get("primary") if valid else None,
            "test_primary": test.get("primary") if test else None,
            "duration_sec": r.get("duration_sec"),
            "error": r.get("error") is not None,
        })

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_csv, "w", newline='', encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader(); w.writerows(summary)

    print(f"Wrote {out_json} and {out_csv} with {len(summary)} rows")


if __name__ == '__main__':
    summarize()
