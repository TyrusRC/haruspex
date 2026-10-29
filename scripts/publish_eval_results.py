"""Assemble a Hugging Face benchmark dataset and the model repo's `.eval_results/`.

Follows https://huggingface.co/docs/hub/eval-results:
  - a BENCHMARK DATASET repo holds the eval splits + a root `eval.yaml` (one task
    per leaderboard); it is validated at push and must be allow-listed by HF (beta)
    before its leaderboard aggregates;
  - each MODEL repo carries `.eval_results/*.yaml`, one entry per (dataset task,
    value), which show on the model page and, once the dataset is an allow-listed
    Benchmark, on its leaderboard.

This script only WRITES those files from the real runs/<domain>/eval.json metrics;
it never uploads (publishing a public dataset is licence-sensitive and outward-facing
— do that by hand after reviewing, e.g. with `hf upload`). Two outputs:

  --dataset-dir DIR   the benchmark dataset: eval.yaml + the licence-clear splits
  --eval-results DIR  the model repo's .eval_results/*.yaml referencing the tasks

Only splits we may redistribute are copied into the dataset: our own synthetic packs,
Microsoft GUIDE (CDLA-Permissive-2.0, with attribution) and Juliet (public domain).
The OWASP Benchmark rows are GPL and test-only, so they are NOT shipped — its score is
reported in the model card prose against the upstream dataset instead.

  python scripts/publish_eval_results.py --runs runs \
      --dataset-dir ~/datasets/haruspex-bench --eval-results runs/sec-v2-lr1e-4-s2/.eval_results
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET_ID = "TyrusRC/haruspex-bench"      # the benchmark dataset these results reference

# task_id -> (run dir under --runs, test split under data/, one-line task description,
#             licence note for the dataset card). A task is one leaderboard.
TASKS = {
    "security_boolean": ("sec-v2-lr1e-4-s2", "security_test.jsonl",
                         "Web/offensive security yes-no and triage decisions", "own synthetic"),
    "mobile_pack": ("mobile", "mobile_test.jsonl", "Mobile app security decisions", "own synthetic"),
    "cloud_pack": ("cloud", "cloud_test.jsonl", "Cloud posture decisions", "own synthetic"),
    "network_pack": ("network", "network_test.jsonl", "Network security decisions", "own synthetic"),
    "llm_pack": ("llm", "llm_test.jsonl", "LLM/AI-app security decisions", "own synthetic"),
    "guide_triage": ("soc-v2", "guide_test.jsonl",
                     "Microsoft GUIDE incident triage grade", "CDLA-Permissive-2.0 (Microsoft GUIDE)"),
    "juliet_code": ("code-v1", "juliet_dev.jsonl",
                    "Code-level vulnerability (NIST Juliet Java)", "public domain / CC0 (NIST)"),
}


def _metrics(run_dir: Path) -> dict | None:
    f = run_dir / "eval.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def write_dataset(dataset_dir: Path, runs: Path) -> list[str]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "data").mkdir(exist_ok=True)
    tasks_yaml, shipped = [], []
    for task_id, (run, split, desc, lic) in TASKS.items():
        src = REPO / "data" / split
        if not (runs / run / "eval.json").exists() or not src.exists():
            continue
        shutil.copyfile(src, dataset_dir / "data" / split)
        shipped.append(f"{task_id} ({lic})")
        tasks_yaml.append(f"  - id: {task_id}\n    config: {task_id}\n    split: test\n"
                          f"    # {desc}; source split data/{split}")
    # eval.yaml: `evaluation_framework` must be a value HF allow-lists (see the eval.ts
    # enum). No entry fits a bespoke classifier yet, so this is marked for HF to confirm.
    (dataset_dir / "eval.yaml").write_text(
        "name: Haruspex Security Decisions\n"
        "description: >\n"
        "  Typed, calibrated security decisions (boolean / choice / score) across web,\n"
        "  mobile, cloud, network, LLM-app, SOC incident triage and code-level\n"
        "  vulnerability. Held-out test splits; accuracy is top-1 over the full label set.\n"
        "evaluation_framework: custom  # TODO: replace with an HF-allow-listed framework id\n"
        "tasks:\n" + "\n".join(tasks_yaml) + "\n")
    (dataset_dir / "README.md").write_text(
        "---\nlicense: other\ntask_categories:\n  - text-classification\n"
        "tags:\n  - security\n  - benchmark\n---\n\n"
        "# Haruspex security-decision benchmark\n\n"
        "Held-out test splits for [`" + DATASET_ID.split('/')[-1] + "`]"
        "(https://huggingface.co/" + DATASET_ID.replace('-bench', '') + "). "
        "Each `.jsonl` row is `{type, state, question, criteria/candidates/levels, target}`.\n\n"
        "Splits and their sources:\n\n"
        + "\n".join(f"- `data/{TASKS[t][1]}` — {TASKS[t][2]} ({TASKS[t][3]})" for t in TASKS
                    if (dataset_dir / 'data' / TASKS[t][1]).exists())
        + "\n\nMicrosoft GUIDE rows are under CDLA-Permissive-2.0; Juliet rows are US "
          "public domain (CC0 elsewhere). The OWASP Benchmark (GPL) is NOT included — it "
          "stays a test-only outside check reported in the model card.\n")
    return shipped


def write_eval_results(out_dir: Path, runs: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for task_id, (run, split, desc, lic) in TASKS.items():
        m = _metrics(runs / run)
        if m is None or "accuracy" not in m:
            continue
        (out_dir / f"{task_id}.yaml").write_text(
            f"- dataset:\n    id: {DATASET_ID}\n    task_id: {task_id}\n"
            f"  value: {round(float(m['accuracy']), 4)}\n"
            f"  notes: \"top-1 accuracy over {m.get('test_n', '?')} held-out rows; "
            f"{desc}\"\n")
        written.append(f"{task_id}={m['accuracy']}")
    return written


def main():
    ap = argparse.ArgumentParser(description="Write HF benchmark dataset + model .eval_results")
    ap.add_argument("--runs", default="runs", help="dir holding <domain>/eval.json")
    ap.add_argument("--dataset-dir", help="write the benchmark dataset here (eval.yaml + splits)")
    ap.add_argument("--eval-results", help="write the model repo's .eval_results/*.yaml here")
    args = ap.parse_args()
    runs = Path(args.runs)
    if not args.dataset_dir and not args.eval_results:
        ap.error("pass --dataset-dir and/or --eval-results")
    if args.dataset_dir:
        shipped = write_dataset(Path(args.dataset_dir).expanduser(), runs)
        print(f"[eval] dataset -> {args.dataset_dir}: {len(shipped)} splits")
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
