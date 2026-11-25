"""Optional PostgreSQL candidate generation for FAQ and handbook retrieval."""
import json
import logging
import os
from typing import Dict, Iterable, Optional, Set


logger = logging.getLogger("pac_helpdesk.hybrid_search")
EMBEDDING_DIMENSIONS = 3072


def _hybrid_search_enabled() -> bool:
    return os.getenv("PGVECTOR_HYBRID_SEARCH", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _embedding_literal(embedding) -> Optional[str]:
    if not embedding or len(embedding) != EMBEDDING_DIMENSIONS:
        return None
    return json.dumps(embedding, separators=(",", ":"))


def get_hybrid_search_status(get_db_fn) -> Dict:
    if not _hybrid_search_enabled():
        return {"enabled": False, "available": False}

    conn = get_db_fn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
                to_regclass('public.idx_faq_embedding_hnsw') IS NOT NULL,
                to_regclass('public.idx_faq_search_vector') IS NOT NULL,
                to_regclass('public.idx_faq_search_trgm') IS NOT NULL,
                to_regclass('public.idx_handbook_embedding_hnsw') IS NOT NULL
        """)
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Could not read pgvector index status")
        extension_version, faq_vector, faq_fts, faq_trigram, handbook_vector = row
        indexes = {
            "faq_vector": bool(faq_vector),
            "faq_full_text": bool(faq_fts),
            "faq_trigram": bool(faq_trigram),
            "handbook_vector": bool(handbook_vector),
        }
        return {
            "enabled": True,
            "available": bool(extension_version) and all(indexes.values()),
            "extension_version": extension_version,
            "indexes": indexes,
        }
    finally:
        cur.close()
        conn.close()


def search_faq_candidate_ids(
    get_db_fn,
    query_embedding,
    query_texts: Iterable[str],
    limit: int = 40,
) -> Optional[Set[int]]:
    """Return vector/FTS/trigram FAQ candidates, or None if migration is absent."""
    vector = _embedding_literal(query_embedding)
    queries = list(dict.fromkeys(
        str(query).strip() for query in query_texts if str(query or "").strip()
    ))
    if not _hybrid_search_enabled() or vector is None or not queries:
        return None

    conn = None
    cur = None
    try:
        conn = get_db_fn()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                to_regclass('public.idx_faq_embedding_hnsw') IS NOT NULL,
                to_regclass('public.idx_faq_search_vector') IS NOT NULL,
                to_regclass('public.idx_faq_search_trgm') IS NOT NULL
        """)
        indexes = cur.fetchone()
        if not indexes or not all(indexes):
            return None

        cur.execute("""
            WITH query_texts AS (
                SELECT DISTINCT unnest(%s::text[]) AS query
            ),
            vector_hits AS (
                SELECT id
                FROM faq
                WHERE embedding_vector IS NOT NULL
                ORDER BY embedding_vector <=> %s::halfvec(3072)
                LIMIT %s
            ),
            fts_hits AS (
                SELECT f.id,
                       MAX(ts_rank_cd(
                           f.search_vector,
                           plainto_tsquery('english', q.query)
                       )) AS rank_score
                FROM faq f
                CROSS JOIN query_texts q
                WHERE f.search_vector @@ plainto_tsquery('english', q.query)
                GROUP BY f.id
                ORDER BY rank_score DESC
                LIMIT %s
            ),
            trigram_hits AS (
                SELECT f.id,
                       MAX(similarity(
                           coalesce(f.question, '') || ' ' ||
                           coalesce(f.related_questions, '') || ' ' ||
                           coalesce(f.keywords, ''),
                           q.query
                       )) AS rank_score
                FROM faq f
                CROSS JOIN query_texts q
                WHERE (
                    coalesce(f.question, '') || ' ' ||
                    coalesce(f.related_questions, '') || ' ' ||
                    coalesce(f.keywords, '')
                ) % q.query
                GROUP BY f.id
                ORDER BY rank_score DESC
                LIMIT %s
            )
            SELECT id FROM vector_hits
            UNION
            SELECT id FROM fts_hits
            UNION
            SELECT id FROM trigram_hits
        """, (queries, vector, limit, limit, limit))
        return {int(row[0]) for row in cur.fetchall()}
    except Exception as exc:
        if conn is not None:
            conn.rollback()
        logger.warning(
            "pgvector FAQ candidate search failed; using in-process retrieval (%s)",
            type(exc).__name__,
        )
        return None
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def search_handbook_candidates(
    cur,
    query_embedding,
    limit: int = 80,
) -> Optional[Dict[int, float]]:
    """Return indexed handbook IDs and cosine similarities, or None if unavailable."""
    vector = _embedding_literal(query_embedding)
    if not _hybrid_search_enabled() or vector is None:
        return None

    cur.execute("""
        SELECT to_regclass('public.idx_handbook_embedding_hnsw') IS NOT NULL
    """)
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    cur.execute("""
        SELECT id, 1 - (embedding_vector <=> %s::halfvec(3072)) AS similarity
        FROM handbook_chunks
        WHERE is_active = 1 AND embedding_vector IS NOT NULL
        ORDER BY embedding_vector <=> %s::halfvec(3072)
        LIMIT %s
    """, (vector, vector, limit))
    return {
        int(chunk_id): max(0.0, min(1.0, float(similarity or 0)))
        for chunk_id, similarity in cur.fetchall()
    }
