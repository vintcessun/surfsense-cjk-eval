"""Index the whole corpus through SurfSense's own ingest pipeline, with one embedder.

    uv run --project $SURFSENSE/surfsense_local/backend python build_index.py bge
    uv run --project $SURFSENSE/surfsense_local/backend python build_index.py granite

The corpus is SurfSense's authored eval corpus plus LIMIT-small and MIRACL from
data/, in one workspace, as one library would be. Each embedder gets its own
fresh database under ~/.surfsense-cjk-eval/<embedder>/: building over an old
one would keep its workspace, and the nearest-neighbour leg searches every
workspace before filtering.
"""

import sys
import time

from common import BACKEND, DATA, EMBEDDERS, use_index


def corpus():
    from retrieval_eval.cases import load_corpus, load_queries

    documents = load_corpus(BACKEND / "scripts" / "retrieval_eval" / "corpus")
    queries = load_queries(BACKEND / "scripts" / "retrieval_eval" / "queries.json")
    for name in ("limit", "miracl"):
        documents += load_corpus(DATA / name / "corpus")
        queries += load_queries(DATA / name / "queries.json")
    return documents, queries


def use_embedder(name: str) -> None:
    """Point SurfSense's encoder at one embedder's folder, as the app's constants would."""
    from worker.ingestion import embedding

    embedding.MODEL_DIR_NAME = EMBEDDERS[name]
    for cached in (embedding.tokenizer, embedding._encoder, embedding._session):
        cached.cache_clear()
    missing = embedding.missing_embedding_files()
    if missing:
        sys.exit(f"{name} is missing {missing}; run fetch.py")


def main(name: str) -> None:
    directory = use_index(name)
    for leftover in directory.glob("*.db*"):
        leftover.unlink()

    from modules.documents.models import Document, DocumentType
    from modules.workspaces.models import Workspace
    from shared.config import get_storage_settings
    from shared.db import create_db_engine, create_session_factory
    from shared.migrations import upgrade_to_head

    use_embedder(name)
    from worker.ingestion import run as ingest

    documents, _ = corpus()
    engine = create_db_engine(get_storage_settings().database_path)
    upgrade_to_head(engine)
    session = create_session_factory(engine)()
    workspace = Workspace(name="CJK eval")
    session.add(workspace)
    session.commit()
    started = time.monotonic()
    for count, document in enumerate(documents, 1):
        row = Document(
            workspace_id=workspace.id,
            title=document.title,
            document_type=DocumentType.NOTE,
            content=document.markdown,
        )
        session.add(row)
        session.commit()
        ingest(row.id)
        if count % 200 == 0:
            print(f"{count}/{len(documents)} in {time.monotonic() - started:.0f}s", flush=True)
    session.close()
    engine.dispose()
    print(f"{name}: {len(documents)} documents in {time.monotonic() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
