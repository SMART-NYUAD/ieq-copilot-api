-- 004: remove the ivfflat indexes that were silently destroying retrieval recall.
--
-- SYMPTOM: "what is CO2?" returned three cards, none about CO2 — while a correct card
-- ("What a high CO2 reading implies for occupants") sat in the corpus at cosine similarity
-- 0.629, higher than anything actually returned.
--
-- CAUSE: env_knowledge_card_embeddings_embedding_idx was built WITH (lists = 100) over a
-- corpus of 55 rows — roughly 0.55 vectors per list, so most lists are empty. An ivfflat
-- scan only visits `probes` lists, so it can only ever see a fraction of the corpus, and
-- it reports that as fewer rows rather than as an error. Measured on the live store:
--
--     probes=1   (postgres default)  ->  0 rows returned for a LIMIT 5 query
--     probes=10  (what the code set) ->  5 rows, top similarity 0.483, WRONG cards
--     probes=100 (== lists)          ->  5 rows, top similarity 0.629, CORRECT cards
--
-- The application set `SET LOCAL ivfflat.probes = max(10, k*3)`, which lands squarely in
-- the broken middle: enough rows to look like a working search, not enough to be right.
--
-- FIX: drop the indexes. pgvector's own guidance is lists ~= rows/1000, i.e. lists=1 at
-- this scale; an approximate-nearest-neighbour index over tens of rows buys nothing a
-- sequential scan does not already give in microseconds, and costs exactness. Postgres
-- will now do an exact scan and return the true top-k every time.
--
-- WHEN TO REINTRODUCE ONE: not until this corpus is in the tens of thousands of rows, and
-- then prefer HNSW (pgvector >= 0.5), which does not have this failure mode and does not
-- need a row-count-dependent `lists`. If ivfflat is used anyway, `lists` must be derived
-- from the row count at build time and the index rebuilt as the corpus grows — a fixed
-- literal is what created this bug.
--
-- Recall is now measurable: `python tests/retrieval_eval.py` scores golden cases and
-- `--probe-sweep` reprints the diagnostic above against whatever indexes exist.

DROP INDEX IF EXISTS env_knowledge_card_embeddings_embedding_idx;

-- Same defect, smaller corpus: lists=10 over ~40 guideline records. search_guideline_records
-- never set probes at all, so it ran at the default of 1 list — about a tenth of the corpus.
DROP INDEX IF EXISTS idx_guideline_embedding;

-- Orphan from an earlier embedding model: rag_cards is vector(768) while everything in use
-- is 1024. Its index also declares no opclass, so it defaults to vector_l2_ops while every
-- query in this codebase uses `<=>` (cosine) — the index could never serve them anyway.
-- The table is left in place (dropping data is not this migration's job); only the
-- misleading index goes.
DROP INDEX IF EXISTS idx_rag_cards_embedding_ivfflat;
