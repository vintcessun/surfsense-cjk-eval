"""Ask every query of every variant, on a copy of one embedder's index.

    uv run --project $SURFSENSE/surfsense_local/backend \
        --with jieba --with sudachipy --with sudachidict_core python variants.py bge

A variant changes only what the keyword leg indexes, how a question is split,
or how its coverage is scored; ranking goes through SurfSense's retrieve().
"""

import json
import math
import re
import sqlite3
import sys
import time
import unicodedata

from common import RESULTS, SEMANTIC_WEIGHTS, WORK, use_index

UNICODE61 = "unicode61 categories 'L* N* Co M*'"
HIRAGANA = "぀-ゟ"
KATAKANA = "゠-ヿㇰ-ㇿｦ-ﾟ"
HAN = "㐀-䶿一-鿿豈-﫿"
CJK = re.compile(f"[{HIRAGANA}{KATAKANA}{HAN}]")
CJK_RUN = re.compile(f"[{HIRAGANA}{KATAKANA}{HAN}]+")
CONTENT_RUN = re.compile(f"[{KATAKANA}{HAN}]+")
KANA = re.compile(f"[{HIRAGANA}{KATAKANA}]")


# --- how a text is split before unicode61 sees it -------------------------------

def pairs(run: str, unigrams: bool = False) -> str:
    if len(run) == 1:
        return f" {run} "
    grams = [run[i : i + 2] for i in range(len(run) - 1)]
    return " " + " ".join([*run, *grams] if unigrams else grams) + " "


def bigram(text, lang=None):
    return CJK_RUN.sub(lambda m: pairs(m.group()), text)


def uni_bigram(text, lang=None):
    return CJK_RUN.sub(lambda m: pairs(m.group(), unigrams=True), text)


def no_hiragana(text, lang=None):
    return CONTENT_RUN.sub(lambda m: pairs(m.group()), re.sub(f"[{HIRAGANA}]+", " ", text))


def nfkc(text, lang=None):
    return unicodedata.normalize("NFKC", text)


def nfkc_no_hiragana(text, lang=None):
    return no_hiragana(unicodedata.normalize("NFKC", text))


_segmenters = {}


def dictionary(text, lang=None):
    """jieba or Sudachi by the text's language, which the app could not know."""
    if lang is None:
        return text
    if lang not in _segmenters:
        if lang == "zh":
            import jieba

            jieba.setLogLevel(60)
            _segmenters[lang] = lambda run: jieba.cut_for_search(run)
        else:
            from sudachipy import Dictionary, SplitMode

            tok = Dictionary().create()
            _segmenters[lang] = lambda run: (m.surface() for m in tok.tokenize(run, SplitMode.A))
    split = _segmenters[lang]
    return CJK_RUN.sub(lambda m: " " + " ".join(w for w in split(m.group()) if w.strip()) + " ", text)


def doc_lang(title: str):
    if title.startswith("miracl-ja") or title.endswith("-ja"):
        return "ja"
    if title.startswith("miracl-zh") or title.endswith("-zh"):
        return "zh"
    return None


def query_lang(text: str):
    if KANA.search(text):
        return "ja"
    return "zh" if CJK.search(text) else None


# --- variants -------------------------------------------------------------------
# (keyword table, question splitter, coverage scoring, abstain on CJK questions)

TRIGRAM = "trigram"
VARIANTS = {
    "today": (None, None, "coverage", False),
    "abstain-cjk": (None, None, "coverage", True),
    "today-idf": (None, None, "idf", False),
    "today-nfkc": (nfkc, nfkc, "coverage", False),
    "trigram": (TRIGRAM, None, "coverage", False),
    "bigram": (bigram, bigram, "coverage", False),
    "bigram-idf": (bigram, bigram, "idf", False),
    "uni-bigram": (uni_bigram, uni_bigram, "coverage", False),
    "uni-bigram-idf": (uni_bigram, uni_bigram, "idf", False),
    "no-hiragana": (no_hiragana, no_hiragana, "coverage", False),
    "no-hiragana-idf": (no_hiragana, no_hiragana, "idf", False),
    "no-hiragana-nfkc": (nfkc_no_hiragana, nfkc_no_hiragana, "coverage", False),
    "no-hiragana-nfkc-idf": (nfkc_no_hiragana, nfkc_no_hiragana, "idf", False),
    "dictionary": (dictionary, dictionary, "coverage", False),
}


def build_table(db: sqlite3.Connection, keyword) -> None:
    if keyword is None:
        pass
    elif keyword == TRIGRAM:
        db.execute("DROP TABLE chunks_fts")
        db.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content='chunks', "
            "content_rowid='id', tokenize=\"trigram\")"
        )
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    else:
        rows = db.execute(
            "SELECT c.id, c.content, d.title FROM chunks c JOIN documents d ON d.id = c.document_id"
        ).fetchall()
        db.execute("DROP TABLE chunks_fts")
        db.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content='', "
            f"contentless_delete=1, tokenize=\"{UNICODE61}\")"
        )
        db.executemany(
            "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
            [(cid, keyword(content, doc_lang(title))) for cid, content, title in rows],
        )
    db.execute("CREATE VIRTUAL TABLE chunks_vocab USING fts5vocab(chunks_fts, row)")
    db.commit()


def splitter(keyword, prepare):
    tokenize = "trigram" if keyword == TRIGRAM else UNICODE61

    def terms(text: str) -> list[str]:
        body = prepare(text, query_lang(text)) if prepare else text
        db = sqlite3.connect(":memory:")
        db.execute(f'CREATE VIRTUAL TABLE q USING fts5(body, tokenize="{tokenize}")')
        db.execute("CREATE VIRTUAL TABLE v USING fts5vocab(q, row)")
        db.execute("INSERT INTO q(body) VALUES (?)", (body,))
        # FTS5's string escape: trigram keeps punctuation, quotes included.
        return [t.replace('"', '""') for (t,) in db.execute("SELECT term FROM v")]

    return terms


def fullwidth(text: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else c for c in text)


def queries():
    from build_index import corpus

    _, asked = corpus()
    # The same questions typed with a full-width IME: what NFKC is for.
    extra = [
        q.model_copy(update={"id": f"{q.id}-fullwidth", "slice": "fullwidth", "text": fullwidth(q.text)})
        for q in asked
        if q.slice in ("identifier", "english")
    ]
    return asked + extra


def main(embedder: str) -> None:
    use_index(embedder)
    from sqlalchemy import text as sql

    import shared.search as search
    from build_index import use_embedder
    from retrieval_eval.score import score
    from shared.db import create_db_engine, create_session_factory

    use_embedder(embedder)
    search.SEMANTIC_WEIGHT = SEMANTIC_WEIGHTS[embedder]
    original_terms, original_leg = search.terms, search._keyword_leg
    asked = queries()
    judged = json.loads((RESULTS.parent / "data" / "miracl" / "judged.json").read_text(encoding="utf-8"))
    out_dir = RESULTS / embedder
    out_dir.mkdir(parents=True, exist_ok=True)
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(VARIANTS)

    source = WORK / embedder / "surfsense.db"
    for name in only:
        keyword, prepare, scoring, abstain = VARIANTS[name]
        path = WORK / f"{embedder}-variants" / f"{name}.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        src, dst = sqlite3.connect(source), sqlite3.connect(path)
        src.backup(dst)
        src.close()
        built = time.monotonic()
        build_table(dst, keyword)
        built = time.monotonic() - built
        index_bytes = dst.execute("SELECT sum(length(block)) FROM chunks_fts_data").fetchone()[0]
        dst.close()

        engine = create_db_engine(path)
        session = create_session_factory(engine)()
        workspace = session.execute(sql("SELECT id FROM workspaces LIMIT 1")).scalar_one()
        chunks = session.execute(sql("SELECT count(*) FROM chunks")).scalar_one()
        df = dict(session.execute(sql("SELECT term, doc FROM chunks_vocab")).all())

        terms = splitter(keyword, prepare) if keyword else original_terms
        search.terms = terms

        def leg(session_, workspace_id, query, document_ids):
            if abstain and CJK.search(query):
                return []
            found = original_leg(session_, workspace_id, query, document_ids)
            if scoring == "coverage" or not found:
                return found
            # Each term weighted by BM25's idf: a rare term counts for more of the
            # question than a common one, and a term the index never saw counts in full.
            weight = {t: math.log(1 + (chunks - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in terms(query)}
            total = sum(weight.values())
            ids = [chunk_id for chunk_id, _ in found]
            matched = {chunk_id: 0.0 for chunk_id in ids}
            for t, w in weight.items():
                for chunk_id in search._matching(session_, workspace_id, f'"{t}"', document_ids, within=ids):
                    matched[chunk_id] += w
            return [(chunk_id, matched[chunk_id] / total) for chunk_id in ids]

        search._keyword_leg = leg
        titles = dict(session.execute(sql("SELECT c.id, d.title FROM chunks c JOIN documents d ON d.id = c.document_id")).all())
        contents = dict(session.execute(sql("SELECT id, content FROM chunks")).all())
        timings = []
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as out:
            for query in asked:
                started = time.monotonic()
                hits = search.retrieve(session, workspace, query.text, top_k=10)
                timings.append(time.monotonic() - started)
                ranking = score(query, [hit.content for hit in hits])
                record = {
                    "query": query.id,
                    "slice": query.slice,
                    "rank": ranking.rank,
                    "in_top_5": ranking.in_top_5,
                    "titles": [hit.title for hit in hits],
                }
                if CJK.search(query.text) or query.slice == "fullwidth":
                    candidates = {c for c, _ in leg(session, workspace, query.text, None)}
                    marked = judged.get(query.id)
                    positive = (
                        {c for c, t in titles.items() if marked.get(t) == 1}
                        if marked
                        else {c for c, body in contents.items() if any(e.casefold() in body.casefold() for e in query.expect)}
                    )
                    record["keyword_recall"] = bool(positive & candidates) if positive else None
                    record["question_terms"] = len(terms(query.text))
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta = {"index_bytes": index_bytes, "build_seconds": round(built, 2), "chunks": chunks,
                "median_query_ms": round(1000 * sorted(timings)[len(timings) // 2], 1)}
        (out_dir / f"{name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
        session.close()
        engine.dispose()
        search.terms, search._keyword_leg = original_terms, original_leg
        print(f"{embedder}/{name}: done {meta}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
