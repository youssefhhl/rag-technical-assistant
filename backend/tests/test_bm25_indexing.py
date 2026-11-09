"""Tests for BM25 index loading + build/persistence (no Qdrant, no model).

All cases run against temporary ``data_dir`` layouts created by the Phase 2
confactory-style helper here; disk writes are confined to ``tmp_path``.
"""

import json
import pickle

import pytest

from app.core.config import Settings
from app.retrieval import bm25_indexing as bm25i
from app.retrieval.bm25_indexing import (
    Bm25Document,
    Bm25IndexBundle,
    build_bm25_index,
    chunks_path_for,
    discover_sources,
    index_path,
    load_bm25_index,
    load_chunks,
    save_bm25_index,
    tokenize,
)


def make_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        scrape_request_delay_seconds=0,
    )


def write_chunks(tmp_path, source: str, count: int) -> None:
    path = chunks_path_for(source, make_settings(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for i in range(count):
            handle.write(
                json.dumps(
                    {
                        "page_content": f"chunk {i}",
                        "metadata": {
                            "source": f"https://docs.test/{source}#chunk{i}",
                            "title": f"{source} doc",
                            "source_name": source,
                            "document_type": "website",
                            "heading_hierarchy": [f"{source} doc"],
                            "chunk_index": i,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# -- tokenization ---------------------------------------------------------------


def test_tokenize_is_lowercase_word_extraction():
    assert tokenize("FastAPI  Routers!") == ["fastapi", "routers"]


def test_tokenize_keeps_accented_words():
    assert tokenize("Configuration du Label") == ["configuration", "du", "label"]
    assert tokenize("Dépendances") == ["dépendances"]


def test_tokenize_empty_text_is_safe():
    assert tokenize("") == []
    assert tokenize("   \n\t ") == []


# -- loading --------------------------------------------------------------------


def test_load_chunks_preserves_content_and_metadata(tmp_path):
    write_chunks(tmp_path, "alpha", 2)
    path = chunks_path_for("alpha", make_settings(tmp_path))

    documents = load_chunks(path)

    assert len(documents) == 2
    assert documents[0].content == "chunk 0"
    assert documents[0].metadata["source_name"] == "alpha"
    assert documents[1].metadata["chunk_index"] == 1


def test_load_chunks_skips_blank_lines_and_pinpoints_malformed(tmp_path):
    write_chunks(tmp_path, "alpha", 2)
    path = chunks_path_for("alpha", make_settings(tmp_path))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n\nnot json\n")

    with pytest.raises(ValueError, match=r"chunks\.jsonl:5"):
        load_chunks(path)


def test_chunks_path_matches_phase2_layout(tmp_path):
    assert chunks_path_for("fastapi", make_settings(tmp_path)) == (
        tmp_path / "data" / "processed" / "fastapi" / "chunks.jsonl"
    )


def test_discover_sources_lists_sources_with_chunks(tmp_path):
    write_chunks(tmp_path, "beta", 1)
    write_chunks(tmp_path, "alpha", 1)
    orphan = make_settings(tmp_path).data_dir / "processed" / "gamma"
    orphan.mkdir(parents=True)  # no chunks.jsonl inside

    assert discover_sources(make_settings(tmp_path)) == ["alpha", "beta"]


# -- build / persistence -----------------------------------------------------------


def test_build_indexes_all_discovered_sources_and_persists(tmp_path):
    write_chunks(tmp_path, "alpha", 3)
    write_chunks(tmp_path, "beta", 2)

    bundle = build_bm25_index(make_settings(tmp_path))

    assert isinstance(bundle, Bm25IndexBundle)
    assert bundle.source_names == ["alpha", "beta"]
    assert len(bundle.documents) == 5
    assert bundle.built_at  # timestamped
    assert bundle.parameters["corpus_size"] == 5
    # persisted exactly once under data/indexes/bm25/
    persisted = index_path(make_settings(tmp_path))
    assert persisted.is_file()
    assert persisted == tmp_path / "data" / "indexes" / "bm25" / "bm25_index.pkl"


def test_build_raises_when_no_processed_chunks(tmp_path):
    with pytest.raises(FileNotFoundError, match="Phase 2 ingestion first"):
        build_bm25_index(make_settings(tmp_path))


def test_build_raises_when_requested_source_missing(tmp_path):
    write_chunks(tmp_path, "alpha", 1)
    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        build_bm25_index(make_settings(tmp_path), source_names=["does-not-exist"])


def test_build_accepts_explicit_source_subset(tmp_path):
    write_chunks(tmp_path, "alpha", 2)
    write_chunks(tmp_path, "beta", 2)

    bundle = build_bm25_index(make_settings(tmp_path), source_names=["beta"])

    assert len(bundle.documents) == 2
    assert {document.metadata["source_name"] for document in bundle.documents} == {"beta"}


def test_save_and_load_roundtrip_preserves_documents(tmp_path):
    write_chunks(tmp_path, "alpha", 2)
    settings = make_settings(tmp_path)
    bundle = build_bm25_index(settings)

    reloaded = load_bm25_index(settings)

    assert len(reloaded.documents) == len(bundle.documents) == 2
    assert [document.content for document in reloaded.documents] == ["chunk 0", "chunk 1"]
    assert reloaded.documents[0].metadata["chunk_index"] == 0
    # identical query behaviour between the built and the reloaded bundle
    scores_built = bundle.bm25.get_scores(tokenize("chunk"))
    scores_loaded = reloaded.bm25.get_scores(tokenize("chunk"))
    assert list(scores_built) == list(scores_loaded)


def test_load_missing_index_raises_with_build_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_bm25_index"):
        load_bm25_index(make_settings(tmp_path))


def test_load_rejects_incompatible_bundle_version(tmp_path):
    settings = make_settings(tmp_path)
    target = index_path(settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        old = Bm25Document(content="legacy", metadata={})
        pickle.dump(
            Bm25IndexBundle(
                documents=[old],
                bm25=None,
                tokenized_corpus=[],
                source_names=["x"],
                built_at="now",
                version=999,  # future/foreign format
            ),
            handle,
        )

    with pytest.raises(ValueError, match="rebuild"):
        load_bm25_index(settings)


def test_save_bm25_index_writes_pickle_under_indexes_dir(tmp_path):
    write_chunks(tmp_path, "alpha", 1)
    settings = make_settings(tmp_path)
    bundle = build_bm25_index(settings)

    path = save_bm25_index(bundle, settings)

    with path.open("rb") as handle:
        raw = pickle.load(handle)
    assert isinstance(raw, Bm25IndexBundle)
    assert len(raw.documents) == 1
