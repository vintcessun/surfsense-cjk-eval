"""Fetch the datasets: MIRACL Japanese and Chinese, LIMIT-small, and the granite model.

    uv run --project $SURFSENSE/surfsense_local/backend --with pyarrow python fetch.py

MIRACL: 100 questions per language from mteb/MIRACLRetrievalHardNegatives,
evenly spaced over the answerable ones sorted by id, with every passage judged
for them, hard negatives included (CC-BY-SA-4.0).
LIMIT-small: every fifth query of orionweller/LIMIT-small, the same 200 that
SurfSense's scripts/fetch_limit_small.py picks (CC-BY-4.0).
granite: ibm-granite/granite-embedding-97m-multilingual-r2's int8 ONNX export,
laid out the way SurfSense loads an embedder (Apache-2.0).
"""

import io
import json
import shutil
import sys

import httpx

from common import DATA, EMBEDDERS, MODELS

PARQUET = "https://datasets-server.huggingface.co/parquet"
ROWS = "https://datasets-server.huggingface.co/rows"
MIRACL = "mteb/MIRACLRetrievalHardNegatives"
LIMIT = "orionweller/LIMIT-small"
GRANITE = "ibm-granite/granite-embedding-97m-multilingual-r2"
QUERIES_PER_LANGUAGE = 100
EXPECT_CHARS = 40
ATTEMPTS = 5


def get(url: str, **kwargs) -> httpx.Response:
    """A GET that survives a dropped connection, which large CDN files do see."""
    for attempt in range(ATTEMPTS):
        try:
            reply = httpx.get(url, timeout=600.0, follow_redirects=True, **kwargs)
            reply.raise_for_status()
            return reply
        except httpx.TransportError:
            if attempt == ATTEMPTS - 1:
                raise
    raise AssertionError("unreachable")


def parquet(dataset: str, config: str) -> list[dict]:
    import pyarrow.parquet as pq

    listing = get(PARQUET, params={"dataset": dataset}).json()["parquet_files"]
    url = next(item["url"] for item in listing if item["config"] == config)
    return pq.read_table(io.BytesIO(get(url).content)).to_pylist()


def slug(value: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in value.lower()).strip("-")


def fetch_miracl() -> None:
    into = DATA / "miracl"
    shutil.rmtree(into, ignore_errors=True)
    (into / "corpus").mkdir(parents=True)
    queries, judged = [], {}
    for lang in ("ja", "zh"):
        judgements: dict[str, dict[str, int]] = {}
        for row in parquet(MIRACL, f"{lang}-qrels"):
            judgements.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
        questions = {str(r["id"]): str(r["text"]) for r in parquet(MIRACL, f"{lang}-queries")}
        answerable = sorted(q for q, p in judgements.items() if q in questions and any(p.values()))
        kept = answerable[:: max(1, len(answerable) // QUERIES_PER_LANGUAGE)][:QUERIES_PER_LANGUAGE]
        wanted = {p for q in kept for p in judgements[q]}
        passages = {str(r["id"]): r for r in parquet(MIRACL, f"{lang}-corpus") if str(r["id"]) in wanted}
        for pid, row in passages.items():
            title = str(row.get("title") or "").strip() or pid
            (into / "corpus" / f"miracl-{lang}-{slug(pid)}.md").write_text(
                f"# {title}\n\n{str(row['text']).strip()}\n", encoding="utf-8"
            )
        for q in kept:
            positives = [p for p, s in judgements[q].items() if s > 0 and p in passages]
            if not positives:
                continue
            query_id = f"miracl-{lang}-{q}"
            queries.append(
                {
                    "id": query_id,
                    "slice": f"miracl-{lang}",
                    "text": questions[q],
                    "expect": [str(passages[p]["text"]).strip().splitlines()[0][:EXPECT_CHARS] for p in positives],
                }
            )
            judged[query_id] = {
                f"miracl-{lang}-{slug(p)}": s for p, s in judgements[q].items() if p in passages
            }
        print(f"MIRACL {lang}: {sum(q['slice'] == f'miracl-{lang}' for q in queries)} queries, {len(passages)} passages")
    (into / "queries.json").write_text(json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")
    (into / "judged.json").write_text(json.dumps(judged, ensure_ascii=False), encoding="utf-8")


def rows(config: str, split: str) -> list[dict]:
    out, total = [], None
    while total is None or len(out) < total:
        page = get(ROWS, params={"dataset": LIMIT, "config": config, "split": split, "offset": len(out), "length": 100}).json()
        total = page["num_rows_total"]
        out += [item["row"] for item in page["rows"]]
    return out


def fetch_limit() -> None:
    """As SurfSense's scripts/fetch_limit_small.py does, into this repo's data."""
    into = DATA / "limit"
    shutil.rmtree(into, ignore_errors=True)
    (into / "corpus").mkdir(parents=True)
    for document in rows("corpus", "corpus"):
        name = str(document["_id"])
        (into / "corpus" / f"limit-{slug(name)}.md").write_text(
            f"# {name}\n\n{str(document.get('text') or '').strip()}\n", encoding="utf-8"
        )
    questions = {str(r["_id"]): str(r["text"]) for r in rows("queries", "queries")}
    relevant: dict[str, list[str]] = {}
    for row in rows("default", "test"):
        if int(row.get("score", 0)) > 0:
            relevant.setdefault(str(row["query-id"]), []).append(str(row["corpus-id"]))
    kept = sorted(relevant)[::5]
    queries = [
        {"id": f"limit-{q}", "slice": "limit", "text": questions[q], "expect": relevant[q]}
        for q in kept
        if q in questions
    ]
    (into / "queries.json").write_text(json.dumps(queries, indent=1), encoding="utf-8")
    print(f"LIMIT-small: {len(queries)} queries")


def fetch_granite() -> None:
    into = MODELS / EMBEDDERS["granite"]
    into.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{GRANITE}/resolve/main"
    for source, target in (
        ("onnx/model_quint8_avx2.onnx", "model_optimized.onnx"),
        ("tokenizer.json", "tokenizer.json"),
        ("config.json", "config.json"),
    ):
        if not (into / target).is_file():
            (into / target).write_bytes(get(f"{base}/{source}").content)
    print(f"granite: {into}")


if __name__ == "__main__":
    fetch_miracl()
    fetch_limit()
    fetch_granite()
    sys.exit(0)
