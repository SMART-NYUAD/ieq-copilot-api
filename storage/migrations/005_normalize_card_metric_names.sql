-- 005: align knowledge-card metric names with the metric registry and guideline records.
--
-- The card store used `tvoc` and `noise` while `executors/metric_registry.py` canonicalises
-- those to `voc` and `sound`, and migration 003 already normalised the guideline records
-- from `tvoc` to `voc` for exactly this reason. Cards were left behind, so the two evidence
-- stores disagreed about what a metric is called: a card carrying metric_name='tvoc' and a
-- guideline record carrying metric='voc' describe the same reading and cannot be joined,
-- filtered, or reasoned about together.
--
-- This was visible in retrieval: a case asking for the `voc` explanation card matched
-- "What TVOC means" — the right card under the wrong key.
--
-- The embeddings are NOT invalidated: metric_name is a filter/label column, not part of the
-- embedded text, so the stored vectors stay correct.

UPDATE env_knowledge_cards SET metric_name = 'voc'   WHERE metric_name = 'tvoc';
UPDATE env_knowledge_cards SET metric_name = 'sound' WHERE metric_name = 'noise';

UPDATE env_knowledge_card_embeddings SET topic = 'voc'   WHERE topic = 'tvoc';
UPDATE env_knowledge_card_embeddings SET topic = 'sound' WHERE topic = 'noise';

UPDATE env_knowledge_cards SET topic = 'voc'   WHERE topic = 'tvoc';
UPDATE env_knowledge_cards SET topic = 'sound' WHERE topic = 'noise';
