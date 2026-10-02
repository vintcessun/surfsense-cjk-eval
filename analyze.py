"""Turn results/<embedder>/*.jsonl into results/summary.md.

    python analyze.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
CJK_SLICES = {"miracl-ja", "miracl-zh", "japanese", "chinese", "cross-lingual"}
SYNTHETIC = {"fullwidth"}
ORDER = ["today", "abstain-cjk", "today-idf", "today-nfkc", "trigram", "bigram", "bigram-idf", "uni-bigram",
         "uni-bigram-idf", "no-hiragana", "no-hiragana-idf", "no-hiragana-nfkc", "no-hiragana-nfkc-idf", "dictionary"]


def load(embedder: str) -> dict[str, dict]:
    runs = {}
    for name in ORDER:
        path = RESULTS / embedder / f"{name}.jsonl"
        if path.is_file():
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            meta = json.loads((RESULTS / embedder / f"{name}.meta.json").read_text())
            runs[name] = {"by_query": {r["query"]: r for r in records}, "meta": meta}
    return runs


def rows(run, slice_):
    return [r for r in run["by_query"].values() if r["slice"] == slice_]


def top5(run, slice_):
    return sum(r["in_top_5"] for r in rows(run, slice_))


def mrr(run, slice_):
    rs = rows(run, slice_)
    return sum(1 / r["rank"] for r in rs if r["rank"]) / len(rs) if rs else 0.0


def recall(run, slice_):
    marked = [r["keyword_recall"] for r in rows(run, slice_) if r.get("keyword_recall") is not None]
    return 100 * sum(marked) / len(marked) if marked else None


def terms(run, slice_):
    ts = [r["question_terms"] for r in rows(run, slice_) if "question_terms" in r]
    return sum(ts) / len(ts) if ts else None


def section(embedder: str, runs: dict) -> list[str]:
    names = list(runs)
    today = runs["today"]
    slices = list(dict.fromkeys(r["slice"] for r in today["by_query"].values()))
    out = [f"## {embedder}", "", "Top 5 / MRR per slice.", "",
           "| slice | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for s in slices:
        n = len(rows(today, s))
        out.append(f"| {s} ({n}) | " + " | ".join(f"{top5(runs[m], s)} / {mrr(runs[m], s):.3f}" for m in names) + " |")
    real = [s for s in slices if s not in SYNTHETIC]
    out.append("| **all real** | " + " | ".join(str(sum(top5(runs[m], s) for s in real)) for m in names) + " |")
    out += ["", "Keyword leg: recall of an answer among its 20 candidates, and terms per question.", "",
            "| slice | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for s in ("miracl-ja", "miracl-zh", "japanese", "chinese", "fullwidth"):
        cells = []
        for m in names:
            r, t = recall(runs[m], s), terms(runs[m], s)
            cells.append("-" if r is None else f"{r:.0f}% / {t:.1f}")
        out.append(f"| {s} | " + " | ".join(cells) + " |")
    out += ["", "| cost | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|",
            "| keyword index (MB) | " + " | ".join(f"{runs[m]['meta']['index_bytes'] / 1e6:.2f}" for m in names) + " |",
            "| index build (s) | " + " | ".join(str(runs[m]["meta"]["build_seconds"]) for m in names) + " |",
            "| median query (ms) | " + " | ".join(str(runs[m]["meta"]["median_query_ms"]) for m in names) + " |",
            "", "Queries in slices without CJK text whose rank differs from `today`:", ""]
    for m in names[1:]:
        moved = [
            f"`{q}` {today['by_query'][q]['rank']}→{r['rank']}"
            for q, r in runs[m]["by_query"].items()
            if r["slice"] not in CJK_SLICES | SYNTHETIC and r["rank"] != today["by_query"][q]["rank"]
        ]
        out.append(f"- **{m}**: " + (", ".join(moved) if moved else "none"))
    return out + [""]


def main() -> None:
    lines = ["# Results", ""]
    for embedder in ("bge", "granite"):
        if (RESULTS / embedder).is_dir():
            lines += section(embedder, load(embedder))
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
