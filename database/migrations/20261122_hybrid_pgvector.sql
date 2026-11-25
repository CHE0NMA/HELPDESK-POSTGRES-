-- Optional hybrid retrieval migration.
-- Run this manually after confirming that the PostgreSQL provider supports
-- pgvector 0.7+ and after taking a database backup.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

ALTER TABLE faq
    ADD COLUMN IF NOT EXISTS embedding_vector halfvec(3072);

ALTER TABLE faq
    ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(question, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(related_questions, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(keywords, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(topic, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(intent, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(answer, '')), 'C')
    ) STORED;

ALTER TABLE handbook_chunks
    ADD COLUMN IF NOT EXISTS embedding_vector halfvec(3072);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM faq
        WHERE embedding IS NOT NULL
          AND CASE
              WHEN jsonb_typeof(embedding) = 'array'
                  THEN jsonb_array_length(embedding)
              ELSE 0
          END <> 3072
    ) OR EXISTS (
        SELECT 1
        FROM handbook_chunks
        WHERE embedding IS NOT NULL
          AND CASE
              WHEN jsonb_typeof(embedding) = 'array'
                  THEN jsonb_array_length(embedding)
              ELSE 0
          END <> 3072
    ) THEN
        RAISE EXCEPTION 'Existing JSONB embeddings must be 3072-dimensional before pgvector migration';
    END IF;
END $$;

UPDATE faq
SET embedding_vector = embedding::text::halfvec(3072)
WHERE embedding IS NOT NULL AND embedding_vector IS NULL;

UPDATE handbook_chunks
SET embedding_vector = embedding::text::halfvec(3072)
WHERE embedding IS NOT NULL AND embedding_vector IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_faq_embedding_hnsw
    ON faq USING hnsw (embedding_vector halfvec_cosine_ops);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_faq_search_vector
    ON faq USING gin (search_vector);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_faq_search_trgm
    ON faq USING gin (
        (
            coalesce(question, '') || ' ' ||
            coalesce(related_questions, '') || ' ' ||
            coalesce(keywords, '')
        ) gin_trgm_ops
    );
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_handbook_embedding_hnsw
    ON handbook_chunks USING hnsw (embedding_vector halfvec_cosine_ops)
    WHERE is_active = 1;
