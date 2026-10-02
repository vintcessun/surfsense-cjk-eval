"""What segmenting costs: per question, per chunk, and for a migration.

    uv run --project $SURFSENSE/surfsense_local/backend python perf.py
"""

import json
import sqlite3
import time
from pathlib import Path

from common import RESULTS, WORK
from variants import UNICODE61, no_hiragana, splitter, uni_bigram


def clock(fn, items, repeat=3) -> float:
    best = float("inf")
    for _ in range(repeat):
        started = time.perf_counter()
        for item in items:
            fn(item)
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    db = sqlite3.connect(WORK / "bge" / "surfsense.db")
    texts = [row[0] for row in db.execute("SELECT content FROM chunks")]
    db.close()
    chars = sum(len(t) for t in texts)
    questions = [q["text"] for q in json.loads((Path(__file__).parent / "data" / "miracl" / "queries.json").read_text(encoding="utf-8"))]

    report = {"chunks": len(texts), "chars": chars}
    for name, fn in (("no-hiragana", no_hiragana), ("uni-bigram", uni_bigram)):
        seconds = clock(fn, texts)
        report[f"{name}_segment_mb_per_s"] = round(chars / seconds / 1e6, 2)
        terms_today = splitter(None, None)
        terms_new = splitter("x", fn)
        report[f"{name}_question_terms_us"] = round(1e6 * clock(terms_new, questions) / len(questions))
        report["today_question_terms_us"] = round(1e6 * clock(terms_today, questions) / len(questions))
        # A migration segments every chunk and fills the index once.
        target = sqlite3.connect(":memory:")
        target.execute(f'CREATE VIRTUAL TABLE f USING fts5(content, content=\'\', contentless_delete=1, tokenize="{UNICODE61}")')
        started = time.perf_counter()
        target.executemany("INSERT INTO f(rowid, content) VALUES (?, ?)", ((i, fn(t)) for i, t in enumerate(texts)))
        target.commit()
        per_chunk = (time.perf_counter() - started) / len(texts)
        report[f"{name}_migration_s_per_100k_chunks"] = round(per_chunk * 100_000, 1)
    (RESULTS / "perf.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
