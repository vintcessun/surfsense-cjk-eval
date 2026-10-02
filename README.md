# surfsense-cjk-eval

Measurements behind SurfSense's proposal for keyword search in Japanese and Chinese ([MODSetter/SurfSense#1997](https://github.com/MODSetter/SurfSense/issues/1997)). They are kept here so anyone can reproduce them against a SurfSense checkout, without adding datasets or scripts to SurfSense itself.

Nothing here changes SurfSense. The scripts import a checkout's own retrieval, ingest and database code: `shared/`, `worker/` and `modules/`, and the eval's `retrieval_eval.cases` and `retrieval_eval.score`. They swap only what a variant names: the keyword table, the rule that splits a question, or the coverage score.

## Run it

```bash
export SURFSENSE=/path/to/SurfSense            # default: ../SurfSense
P="--project $SURFSENSE/surfsense_local/backend"
uv run $P python $SURFSENSE/surfsense_local/backend/scripts/fetch_embedding_model.py
uv run $P --with pyarrow python fetch.py                 # MIRACL ja/zh, LIMIT-small, granite
uv run $P python build_index.py bge                      # ~15 min on a laptop CPU
uv run $P python build_index.py granite                  # slower: a 97M-parameter model
uv run $P --with jieba --with sudachipy --with sudachidict_core python variants.py bge
uv run $P --with jieba --with sudachipy --with sudachidict_core python variants.py granite
uv run $P python perf.py
python analyze.py                                        # writes results/summary.md
```

Measured against SurfSense `dev` at `cddfd05f9`.

## The data

- **SurfSense's authored eval corpus**: 17 documents and 66 queries, read from the checkout.
- **LIMIT-small**: the same 200 queries SurfSense's `scripts/fetch_limit_small.py` picks (CC-BY-4.0).
- **MIRACL**: 100 Japanese and 100 Chinese questions from `mteb/MIRACLRetrievalHardNegatives`, evenly spaced over the answerable ones, with all 1,757 passages judged for them, hard negatives included (CC-BY-SA-4.0). These are native questions over Wikipedia, the natural-language case.
- **fullwidth** (synthetic, 20 queries): the authored English and identifier queries retyped in full-width characters (`Ｘ４００`), as a Japanese or Chinese IME produces them. It measures normalization and is left out of every "all" row.

Everything is indexed into one workspace through SurfSense's ingest pipeline, as a library would be.

## Variants

| Variant | What changes |
|---|---|
| `today` | nothing |
| `abstain-cjk` | the keyword leg proposes and scores nothing for a question containing CJK |
| `today-idf` | coverage weights each term by BM25's idf instead of counting terms |
| `trigram` | FTS5's `trigram` tokenizer for index and question |
| `bigram` | Han and kana runs as overlapping pairs before `unicode61` (the Lucene/Elasticsearch `cjk` baseline) |
| `uni-bigram` | as `bigram`, plus each character |
| `no-hiragana` | hiragana as a separator, Han and katakana runs as pairs |
| `*-idf` | the same split, idf-weighted coverage |
| `no-hiragana-nfkc` | NFKC before splitting, so full-width and half-width forms fold |
| `today-nfkc` | NFKC alone, no segmenting: added after the granite run showed NFKC to be the one gain that holds under both embedders |
| `no-hiragana-nfkc-idf` | both, added after the first bge run showed NFKC free and idf worth four Chinese answers |
| `dictionary` | jieba (search mode) for Chinese, Sudachi mode A for Japanese, chosen by the text's known language: an upper bound the app cannot reach, since it cannot tell Han-only text's language |

Each runs on both embedders: SurfSense's `bge-small-en-v1.5` at its semantic weight 0.65, and `granite-embedding-97m-multilingual-r2` at 0.85, the weight `docs/proposals/retrieval.md` measured for it.

## How the results are read

This was written before any variant was run on these indexes.

1. **The deciding measure is the natural-language one**: top 5 and MRR on `miracl-ja` and `miracl-zh`.
2. **No slice without CJK text may lose a top-5 answer**, LIMIT included. Every query whose rank moves there is listed, with its cause.
3. **The keyword leg's own recall** (an answer among its 20 candidates) shows whether it measures anything. It is a diagnostic, not the decision: the blend decides what a person sees.
4. **Between variants within a point or two on (1)**, the one with fewer moving parts wins: no index change over an index change, no new dependency over one, the smaller index over the larger.
5. **Both embedders count.** A variant that helps under bge but not under granite is worth less, since SurfSense may move to granite.

An earlier round also scored coverage against MIRACL's judged hard negatives. That turned out to be the wrong test, because those negatives were pooled partly from BM25, so they were chosen for sharing words with the question. It is not used here.

## Results

Every slice, MRR, keyword recall and each moved query are in [`results/summary.md`](results/summary.md). Costs are in [`results/perf.json`](results/perf.json), and per-query rankings in `results/<embedder>/*.jsonl`.

- **The embedder decides natural-language questions.** Today's MIRACL top 5 is 56 (ja) and 47 (zh) out of 100 with bge, and 92 and 91 with granite.
- **Under bge, segmenting CJK is a large gain.** NFKC plus bigrams over Han and katakana, with hiragana as a separator, gives 77 and 75, and 356 → 411 over all 466 queries. No query outside CJK loses rank; one improves.
- **Under granite, segmenting is neutral for natural-language questions** (92 and 91, within noise), with no query outside CJK losing rank. It still takes the keyword leg's recall from 34% and 1% to 99%, and the authored Japanese questions from 4 to 5 of 5.
- **NFKC alone** changes no real query under either embedder and takes full-width input from 0 to 15 (bge) and 19 (granite) of 20.
- **Rejected:**
  - abstaining on CJK questions (worse under bge);
  - idf-weighted coverage (nothing alone, mixed on top);
  - `trigram` (moves 14 queries without CJK under bge and 45 under granite);
  - dictionaries (not better, ~100 MB, need language identification).
- **Cost of the recommended split:**
  - 8.75 MB/s to segment;
  - +31 µs per question;
  - 6.5 s per 100,000 chunks to migrate;
  - a keyword index 5% larger than today's.

## Licences

The code here is offered under the same licence as SurfSense. The data is fetched, not committed: MIRACL (CC-BY-SA-4.0), LIMIT-small (CC-BY-4.0), granite (Apache-2.0).
