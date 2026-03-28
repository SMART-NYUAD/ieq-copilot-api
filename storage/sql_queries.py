"""Local SQL constants for standalone API operation."""

ENV_KNOWLEDGE_QUERY_SEMANTIC_SQL = """
SELECT
    c.knowledge_card_id,
    c.card_type,
    c.topic,
    c.title,
    c.summary,
    c.content,
    c.audience,
    c.severity_level,
    c.metric_name,
    c.source_label,
    c.source_url_key,
    c.source_metadata,
    1 - (e.embedding <=> %s::vector) AS distance
FROM env_knowledge_card_embeddings e
JOIN env_knowledge_cards c
    ON c.knowledge_card_id = e.knowledge_card_id
ORDER BY e.embedding <=> %s::vector
LIMIT %s
"""

