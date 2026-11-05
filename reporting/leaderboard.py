"""Leaderboard à partir des résultats de tools/eval.py (leaderboard.py de l'arène).

Chaque tâche est classée par sa métrique. Le classement général est le gain moyen sur la
baseline (`--baseline`, défaut "baseline") sur toutes les tâches ; une tâche non soumise
compte comme un gain nul. Les références (poids externes) sont en italique et sans rang.
Les soumissions en violation sont listées à part.

    python reporting/leaderboard.py --split val
    python reporting/leaderboard.py --split test

Lit, pour chaque nom, le dernier results/eval/<nom>/<split>/<date>/results.json, et écrit
results/eval/leaderboard_<split>.md.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def latest_results(results_root: Path, split: str) -> list[dict]:
    """Le dernier résultat de chaque nom pour ce split (dates triables par nom de dossier)."""
    runs = []
    for team_dir in sorted((results_root / "eval").glob("*")):
        files = sorted(team_dir.glob(f"{split}/*/results.json"))
        if files:
            runs.append(json.loads(files[-1].read_text(encoding="utf-8")))
    return runs


def ranker():
    """Rang des équipes en compétition ; les références prennent 'ref' et aucune place."""
    count = [0]

    def rank(r):
        if r.get("reference"):
            return "ref"
        count[0] += 1
        return str(count[0])

    return rank


def label(r):
    return f"*{r['team']} (reference)*" if r.get("reference") else r["team"]


def render(runs: list[dict], split: str, baseline: str = "baseline") -> str:
    tasks = sorted({t for r in runs for t in r["tasks"]})
    base = next((r for r in runs if r["team"] == baseline), None)
    base_score = {t: base["tasks"][t]["score"] for t in base["tasks"]} if base else {}

    def valid(r, t=None):
        if r["violations"]:
            return False
        return t is None or not r["tasks"][t]["violations"]

    lines = [f"# Leaderboard ({split})", ""]
    overall = []
    for r in runs:
        if not valid(r):
            continue
        gains = [
            r["tasks"][t]["score"] - base_score.get(t, 0.0)
            if t in r["tasks"] and valid(r, t)
            else 0.0
            for t in tasks
        ]
        overall.append((sum(gains) / max(1, len(tasks)), r))
    lines += [
        "## Overall: mean gain over baseline",
        "",
        "| # | team | backbone (M params) | GPU-h | " + " | ".join(tasks) + " | mean gain |",
        "|---|---|---|---|" + "---|" * len(tasks) + "---|",
    ]
    rank = ranker()
    for g, r in sorted(overall, key=lambda x: -x[0]):
        cells = " | ".join(
            "-"
            if t not in r["tasks"]
            else "rejected"
            if not valid(r, t)
            else f"{r['tasks'][t]['score']:.4f}"
            for t in tasks
        )
        gpu = r["backbone"].get("gpu_hours")
        lines.append(
            f"| {rank(r)} | {label(r)} | {r['backbone']['params'] / 1e6:.1f} | "
            f"{'-' if gpu is None else gpu} | {cells} | **{g:+.4f}** |"
        )
    for t in tasks:
        rows = sorted(
            [r for r in runs if t in r["tasks"] and valid(r, t)],
            key=lambda r: -r["tasks"][t]["score"],
        )
        metric = rows[0]["tasks"][t]["metric"] if rows else ""
        lines += [
            "",
            f"## {t} ({metric})",
            "",
            "| # | team | score | vs baseline | head params |",
            "|---|---|---|---|---|",
        ]
        rank = ranker()
        for r in rows:
            s = r["tasks"][t]["score"]
            lines.append(
                f"| {rank(r)} | {label(r)} | {s:.4f} | {s - base_score.get(t, s):+.4f} | "
                f"{r['tasks'][t]['head_params']:,} |"
            )
    rejected = [(r["team"], v) for r in runs for v in r["violations"]]
    rejected += [
        (r["team"], f"{t}: {v}")
        for r in runs
        for t in r["tasks"]
        for v in r["tasks"][t]["violations"]
    ]
    if rejected:
        lines += ["", "## Rejected", ""] + [f"- {team}: {v}" for team, v in rejected]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--baseline", default="baseline", help="nom de référence des gains")
    p.add_argument("--results", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    md = render(latest_results(args.results, args.split), args.split, args.baseline)
    out = args.results / "eval" / f"leaderboard_{args.split}.md"
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
