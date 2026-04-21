-- Guideline records table for sourced IEQ threshold citations
CREATE TABLE IF NOT EXISTS env_guideline_records (
    id                  SERIAL PRIMARY KEY,

    -- Source identification
    source_key          TEXT NOT NULL,
    source_label        TEXT NOT NULL,
    source_url          TEXT,
    section_ref         TEXT,
    publication_year    INTEGER,
    effective_date      DATE,

    -- Classification
    metric              TEXT NOT NULL
                        CHECK (metric IN (
                            'co2', 'pm25', 'tvoc', 'temperature',
                            'humidity', 'light', 'sound', 'ieq', 'general'
                        )),
    citation_tier       TEXT NOT NULL
                        CHECK (citation_tier IN (
                            'regulatory', 'research', 'internal'
                        )),

    -- The claim
    claim_text          TEXT NOT NULL,
    embed_text          TEXT NOT NULL,

    -- Structured threshold values
    threshold_value     NUMERIC,
    threshold_type      TEXT
                        CHECK (threshold_type IN (
                            'max', 'min', 'target',
                            'range_min', 'range_max', NULL
                        )),
    threshold_unit      TEXT,
    threshold_condition TEXT,

    -- Caveat
    caveat_text         TEXT,

    -- Embedding
    embedding           vector(1536),

    -- Metadata
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),
    is_active           BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_guideline_metric
    ON env_guideline_records(metric);
CREATE INDEX IF NOT EXISTS idx_guideline_tier
    ON env_guideline_records(citation_tier);
CREATE INDEX IF NOT EXISTS idx_guideline_source
    ON env_guideline_records(source_key);
CREATE INDEX IF NOT EXISTS idx_guideline_metric_tier
    ON env_guideline_records(metric, citation_tier);
CREATE INDEX IF NOT EXISTS idx_guideline_embedding
    ON env_guideline_records
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
