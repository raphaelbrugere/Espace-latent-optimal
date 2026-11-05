"""Build the leaderboard from results/<split>/*.json.

Each task is ranked by its metric. The overall ranking uses the average gain
over the baseline across all tasks (a task not submitted counts as no gain). Reference
backbones (external weights, flagged in their pretraining log) are shown in italics and
take no rank. Submissions with rule violations are listed separately.

    python -m arena.leaderboard --split test
"""
import argparse
import json
import pathlib

BASELINE = "baseline"


def ranker():
    """Rank competing teams only: references (external weights) get 'ref' and take no place."""
    count = [0]

    def rank(r):
        if r.get("reference"):
            return "ref"
        count[0] += 1
        return str(count[0])
    return rank


def label(r):
    return f"*{r['team']} (reference)*" if r.get("reference") else r["team"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="leaderboard.md")
    args = p.parse_args(argv)
    runs = [json.loads(f.read_text()) for f in sorted((pathlib.Path(args.results) / args.split).glob("*.json"))]
    tasks = sorted({t for r in runs for t in r["tasks"]})
    base = next((r for r in runs if r["team"] == BASELINE), None)
    base_score = {t: base["tasks"][t]["score"] for t in base["tasks"]} if base else {}

    def valid(r, t=None):
        if r["violations"]:
            return False
        return t is None or not r["tasks"][t]["violations"]

    lines = [f"# Leaderboard ({args.split})", ""]
    overall = []
    for r in runs:
        if not valid(r):
            continue
        gains = [r["tasks"][t]["score"] - base_score.get(t, 0.0)
                 if t in r["tasks"] and valid(r, t) else 0.0 for t in tasks]
        overall.append((sum(gains) / max(1, len(tasks)), r))
    lines += ["## Overall: mean gain over baseline", "",
              "| # | team | backbone (M params) | GPU-h | " + " | ".join(tasks) + " | mean gain |",
              "|---|---|---|---|" + "---|" * len(tasks) + "---|"]
    rank = ranker()
    for g, r in sorted(overall, key=lambda x: -x[0]):
        cells = " | ".join("-" if t not in r["tasks"] else "rejected" if not valid(r, t)
                           else f"{r['tasks'][t]['score']:.4f}" for t in tasks)
        gpu = r["backbone"].get("gpu_hours")
        lines.append(f"| {rank(r)} | {label(r)} | {r['backbone']['params'] / 1e6:.1f} | "
                     f"{'-' if gpu is None else gpu} | {cells} | **{g:+.4f}** |")
    for t in tasks:
        rows = sorted([r for r in runs if t in r["tasks"] and valid(r, t)], key=lambda r: -r["tasks"][t]["score"])
        metric = rows[0]["tasks"][t]["metric"] if rows else ""
        lines += ["", f"## {t} ({metric})", "", "| # | team | score | vs baseline | head params |", "|---|---|---|---|---|"]
        rank = ranker()
        for r in rows:
            s = r["tasks"][t]["score"]
            lines.append(f"| {rank(r)} | {label(r)} | {s:.4f} | {s - base_score.get(t, s):+.4f} | "
                         f"{r['tasks'][t]['head_params']:,} |")
    rejected = [(r["team"], v) for r in runs for v in r["violations"]]
    rejected += [(r["team"], f"{t}: {v}") for r in runs for t in r["tasks"] for v in r["tasks"][t]["violations"]]
    if rejected:
        lines += ["", "## Rejected", ""] + [f"- {team}: {v}" for team, v in rejected]
    md = "\n".join(lines) + "\n"
    pathlib.Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
