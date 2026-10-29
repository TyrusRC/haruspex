"""Assemble a Hugging Face benchmark dataset and the model repo's `.eval_results/`.

Follows https://huggingface.co/docs/hub/eval-results:
  - a BENCHMARK DATASET repo holds the eval splits + a root `eval.yaml` (one task per
    leaderboard); it is validated at push and must be allow-listed by HF (beta) before
    its leaderboard aggregates;
  - each MODEL repo carries `.eval_results/*.yaml`, one entry per (dataset task, value),
    shown on the model page and — once the dataset is an allow-listed Benchmark — on its
    leaderboard.

Self-split accuracy hides overfitting (the Juliet-trained code adapter scored 0.71 on a
Juliet split but 0.50 on independent code sets), so results are split into two kinds and
labelled as such:
  - INDEPENDENT: a public dataset the adapter never trained on (the honest number). Its
    result references THAT dataset's Hub id directly.
  - in-distribution: a held-out split of our own data, referencing the `haruspex-bench`
    dataset this script assembles; the note marks it in-distribution.

This script only WRITES the files from runs/<dir>/<eval json>; it never uploads (publishing
a public dataset is licence-sensitive). Outputs:
  --dataset-dir DIR    the benchmark dataset: eval.yaml + the licence-clear in-dist splits
  --eval-results DIR   the model repo's .eval_results/*.yaml

  python scripts/publish_eval_results.py --runs runs \
      --dataset-dir ~/datasets/haruspex-bench --eval-results runs/sec-v2-lr1e-4-s2/.eval_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH_ID = "TyrusRC/haruspex-bench"   # our dataset, for the in-distribution splits

# task_id -> dict. `dataset` is the Hub id the result references; `independent` marks a
# public set never trained on. `split` (in-dist only) is copied into haruspex-bench.
RESULTS = {
    "prompt_injection_deepset": {
        "run": "llm", "eval": "eval_deepset.json", "dataset": "deepset/prompt-injections",
        "independent": True, "desc": "prompt-injection detection, independent public set"},
    "security_boolean": {
        "run": "sec-v2-lr1e-4-s2", "eval": "eval.json", "dataset": BENCH_ID,
        "independent": False, "split": "security_test.jsonl",
        "desc": "web/offensive security decisions", "lic": "own synthetic"},
    "mobile_pack": {
        "run": "mobile", "eval": "eval.json", "dataset": BENCH_ID, "independent": False,
        "split": "mobile_test.jsonl", "desc": "mobile app security", "lic": "own synthetic"},
    "cloud_pack": {
        "run": "cloud", "eval": "eval.json", "dataset": BENCH_ID, "independent": False,
        "split": "cloud_test.jsonl", "desc": "cloud posture", "lic": "own synthetic"},
    "network_pack": {
        "run": "network", "eval": "eval.json", "dataset": BENCH_ID, "independent": False,
        "split": "network_test.jsonl", "desc": "network security", "lic": "own synthetic"},
    "guide_triage": {
        "run": "soc-v2", "eval": "eval.json", "dataset": BENCH_ID, "independent": False,
        "split": "guide_test.jsonl", "desc": "Microsoft GUIDE incident triage",
        "lic": "CDLA-Permissive-2.0 (Microsoft GUIDE)"},
}


def _metrics(runs: Path, r: dict) -> dict | None:
    f = runs / r["run"] / r["eval"]
    return json.loads(f.read_text()) if f.exists() else None


def write_dataset(dataset_dir: Path, runs: Path) -> list[str]:
    import shutil
    (dataset_dir / "data").mkdir(parents=True, exist_ok=True)
    tasks_yaml, shipped = [], []
    for task_id, r in RESULTS.items():
        if r["independent"] or _metrics(runs, r) is None:   # independent sets live in their own repo
            continue
        src = REPO / "data" / r["split"]
        if not src.exists():
            continue
        shutil.copyfile(src, dataset_dir / "data" / r["split"])
        shipped.append(f"{task_id} ({r['lic']})")
        tasks_yaml.append(f"  - id: {task_id}\n    config: {task_id}\n    split: test\n"
                          f"    # {r['desc']}; source split data/{r['split']}")
    (dataset_dir / "eval.yaml").write_text(
        "name: Haruspex Security Decisions\n"
        "description: >\n"
        "  Held-out, in-distribution test splits for the Haruspex security-decision adapters\n"
        "  (web, mobile, cloud, network, SOC triage). Accuracy is top-1 over the full label\n"
        "  set. Independent, never-trained-on results are reported against their own public\n"
        "  datasets, not here.\n"
        "evaluation_framework: custom  # TODO: replace with an HF-allow-listed framework id\n"
        "tasks:\n" + "\n".join(tasks_yaml) + "\n")
    (dataset_dir / "README.md").write_text(
        "---\nlicense: other\ntask_categories:\n  - text-classification\n"
        "tags:\n  - security\n  - benchmark\n---\n\n"
        "# Haruspex security-decision benchmark (in-distribution splits)\n\n"
        "Held-out test splits for [`haruspex`](https://huggingface.co/TyrusRC/haruspex). "
        "Each `.jsonl` row is `{type, state, question, criteria/candidates/levels, target}`. "
        "These are in-distribution splits of the training data; independent results are "
        "reported in the model card against public datasets (e.g. deepset/prompt-injections, "
        "the OWASP Benchmark).\n\n"
        + "\n".join(f"- `data/{RESULTS[t]['split']}` — {RESULTS[t]['desc']} ({RESULTS[t]['lic']})"
                    for t in RESULTS if not RESULTS[t]["independent"]
                    and (dataset_dir / 'data' / RESULTS[t]['split']).exists()) + "\n")
    return shipped


def write_eval_results(out_dir: Path, runs: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task_id, r in RESULTS.items():
        m = _metrics(runs, r)
        if m is None or "accuracy" not in m:
            continue
        kind = "independent" if r["independent"] else "in-distribution split"
        (out_dir / f"{task_id}.yaml").write_text(
            f"- dataset:\n    id: {r['dataset']}\n    task_id: {task_id}\n"
            f"  value: {round(float(m['accuracy']), 4)}\n"
            f"  notes: \"{kind}; top-1 accuracy over {m.get('test_n', '?')} rows; {r['desc']}\"\n")
        written.append(f"{task_id}={m['accuracy']:.4f} [{kind}] -> {r['dataset']}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Write HF benchmark dataset + model .eval_results")
    ap.add_argument("--runs", default="runs", help="dir holding <run>/<eval json>")
    ap.add_argument("--dataset-dir", help="write the in-distribution benchmark dataset here")
    ap.add_argument("--eval-results", help="write the model repo's .eval_results/*.yaml here")
    args = ap.parse_args()
    runs = Path(args.runs)
    if not args.dataset_dir and not args.eval_results:
        ap.error("pass --dataset-dir and/or --eval-results")
    if args.dataset_dir:
        shipped = write_dataset(Path(args.dataset_dir).expanduser(), runs)
        print(f"[eval] dataset -> {args.dataset_dir}: {len(shipped)} in-distribution splits")
        for s in shipped:
            print(f"       {s}")
        print("[eval] NOTE: eval.yaml evaluation_framework is 'custom' — HF must allow-list a "
              "framework id and the dataset before its leaderboard aggregates.")
    if args.eval_results:
        written = write_eval_results(Path(args.eval_results).expanduser(), runs)
        print(f"[eval] .eval_results -> {args.eval_results}: {len(written)} results")
        for w in written:
            print(f"       {w}")


if __name__ == "__main__":
    main()
