#!/usr/bin/env python3
"""Seed knowledge cards and their embeddings.

    python -m storage.seed_knowledge_cards          # insert missing cards
    python -m storage.seed_knowledge_cards --list   # show what would be inserted

The card corpus previously had no seeder in this repository — cards were loaded out of
band, which is why gaps in it were invisible until retrieval was measured. Cards added here
are the ones ``tests/retrieval_eval.py`` showed the corpus could not answer:

* the system reports illumination (IIL) but held **no light cards at all**, so
  "why is it so dim in here" had nothing to ground against;
* ``humidity`` and ``sound`` had interpretation cards (what a *reading* means) but no
  explanation card (what the *metric* is), so definition questions fell through to a
  status card — the failure mode the retrieval eval's `definition` group exists to catch;
* the illumination sub-index had no ``ieq_subindex`` card while the other three did.

Embedding side note: ``embed_documents`` is used deliberately — passages must NOT carry the
bge query instruction prefix that ``embed_query`` applies. See ``storage/embeddings.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from storage.embeddings import embed_documents
from storage.postgres_client import get_cursor


def _card(
    *,
    card_type: str,
    topic: str,
    title: str,
    summary: str,
    content: str,
    metric_name: str | None,
    source_label: str,
    source_url_key: str,
    audience: str = "general",
    severity_level: str | None = None,
) -> Dict[str, Any]:
    return {
        "card_type": card_type,
        "topic": topic,
        "title": title,
        "summary": summary,
        "content": content,
        "metric_name": metric_name,
        "audience": audience,
        "severity_level": severity_level,
        "source_label": source_label,
        "source_url_key": source_url_key,
        # What actually gets embedded. Title + summary + content together, because a
        # title alone is too short to embed well and content alone loses the topic word
        # a short question keys off.
        "embed_text": f"{title}. {summary} {content}",
    }


KNOWLEDGE_CARDS: List[Dict[str, Any]] = [
    _card(
        card_type="explanation",
        topic="humidity",
        metric_name="humidity",
        title="What relative humidity means",
        summary=(
            "Relative humidity is how much moisture the air holds compared with the most "
            "it could hold at that temperature, shown as a percentage."
        ),
        content=(
            "Relative humidity (RH) describes how saturated the air is with water vapour, "
            "as a percentage of the maximum the air could hold at its current temperature. "
            "Because that maximum rises with temperature, the same amount of moisture reads "
            "as a lower RH in a warm room than a cool one. Most people are comfortable "
            "between roughly 30% and 60% RH. Below that, air feels dry and can cause static, "
            "dry eyes, irritated throats and cracked skin. Above it, the room feels clammy, "
            "sweat evaporates poorly so it feels warmer than the thermometer suggests, and "
            "sustained high humidity encourages mould growth and dust-mite activity. Humidity "
            "is a comfort and building-health metric rather than a pollutant."
        ),
        source_label="EPA Indoor Humidity Guidance",
        source_url_key="EPA_HUMIDITY",
    ),
    _card(
        card_type="explanation",
        topic="sound",
        metric_name="sound",
        title="What the sound level measurement means",
        summary=(
            "Sound level is the loudness of background noise in the space, measured in "
            "decibels (dB) on a logarithmic scale."
        ),
        content=(
            "The sound sensor reports background noise in A-weighted decibels, which "
            "approximate how the human ear responds across frequencies. The scale is "
            "logarithmic: an increase of about 10 dB is perceived as roughly twice as loud, "
            "so a change from 45 dB to 55 dB is a large difference, not a small one. A quiet "
            "office typically sits around 40-50 dB; normal conversation is about 60 dB. "
            "Sustained levels above roughly 55 dB make concentration and conversation harder "
            "and are a common source of complaints in open-plan spaces. This measures ambient "
            "level over time, not brief peaks such as a door closing."
        ),
        source_label="WHO Noise Guidance",
        source_url_key="WHO_NOISE",
    ),
    _card(
        card_type="explanation",
        topic="light",
        metric_name="light",
        title="What the light level measurement means",
        summary=(
            "Light level is how much illumination reaches the working surface, measured in "
            "lux."
        ),
        content=(
            "Illuminance is measured in lux — lumens falling on a square metre of surface. "
            "It describes how well-lit the space is where work actually happens, not how "
            "bright a lamp is. General office and study areas usually target somewhere around "
            "300-500 lux on the desk; detailed or paper-based tasks want more, and circulation "
            "areas need less. Too little light causes eye strain, fatigue and headaches, and "
            "pushes people to lean toward screens. Too much, or poorly placed light, causes "
            "glare and reflections that are just as tiring. Readings vary a lot with time of "
            "day, blind position and where the sensor sits relative to windows."
        ),
        source_label="Internal Combined Interpretation Policy",
        source_url_key="INTERNAL_COMBINED_POLICY",
    ),
    _card(
        card_type="ieq_subindex",
        topic="illumination_subindex",
        metric_name="iil",
        title="What the illumination subindex (IIL) means",
        summary=(
            "IIL scores how well-lit the space is on a 0-100 scale where higher is better."
        ),
        content=(
            "The illumination subindex (IIL) is one of the four dimensions behind the overall "
            "IEQ score, alongside air quality (IAQ), thermal comfort (ITC) and acoustic "
            "comfort (IAC). It converts measured light levels into a 0-100 score where HIGHER "
            "IS BETTER: a high IIL means lighting is adequate for the space's use, and a low "
            "IIL means it is dim or poorly distributed. It is an internal composite score, not "
            "an external standard, so it should never be presented as a compliance threshold. "
            "A low IIL is a real finding — it commonly shows up as eye strain, fatigue and "
            "reduced concentration — and usually points at lamp failure, reduced luminaire "
            "output, blinds, or daylight falling off later in the day."
        ),
        source_label="Internal Combined Interpretation Policy",
        source_url_key="INTERNAL_COMBINED_POLICY",
    ),
    _card(
        card_type="explanation",
        topic="voc",
        metric_name="voc",
        title="What VOCs are and where they come from",
        summary=(
            "Volatile organic compounds are carbon-based chemicals that evaporate into the "
            "air at room temperature."
        ),
        content=(
            "Volatile organic compounds (VOCs) are carbon-based chemicals that readily "
            "evaporate indoors. Common sources are cleaning products, solvents, adhesives, "
            "paints and coatings, printers, new furniture and flooring, personal-care "
            "products, and people themselves. The sensor reports TVOC — a single summed "
            "indicator of total VOC load rather than a measurement of any one compound — so "
            "it shows whether chemical load is rising, not which chemical is responsible. "
            "This sensor is ethanol-calibrated and reports in ppm, while most published TVOC "
            "guidelines are written in µg/m³; converting between them requires assuming a "
            "compound mix, so readings and mass-based limits are not directly comparable. "
            "Elevated TVOC is commonly noticed as a chemical or 'new' smell, and at higher "
            "levels as eye, nose and throat irritation or headache."
        ),
        source_label="RESET Air Standard",
        source_url_key="RESET_AIR",
    ),
]


_INSERT_CARD = """
INSERT INTO env_knowledge_cards (
    card_type, topic, title, summary, content, audience,
    severity_level, metric_name, condition_json, recommendation_json,
    tags, source_label, source_url_key, source_metadata
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
RETURNING knowledge_card_id
"""

_INSERT_EMBEDDING = """
INSERT INTO env_knowledge_card_embeddings (
    knowledge_card_id, topic, card_type, embedding, model_name
) VALUES (%s,%s,%s,%s,%s)
"""


def _existing_titles(cur) -> set:
    cur.execute("SELECT title FROM env_knowledge_cards")
    return {row["title"] for row in cur.fetchall()}


def seed(dry_run: bool = False) -> int:
    """Insert any card in KNOWLEDGE_CARDS whose title is not already present."""
    with get_cursor() as cur:
        present = _existing_titles(cur)
        pending = [c for c in KNOWLEDGE_CARDS if c["title"] not in present]

        if not pending:
            print(f"All {len(KNOWLEDGE_CARDS)} cards already present — nothing to do.")
            return 0
        print(f"{len(pending)} card(s) to insert:")
        for card in pending:
            print(f"  [{card['card_type']:<14}] {card['metric_name'] or '-':<10} {card['title']}")
        if dry_run:
            return 0

        vectors = embed_documents([c["embed_text"] for c in pending])
        if len(vectors) != len(pending):
            print(f"ERROR: expected {len(pending)} embeddings, got {len(vectors)}")
            return 1

        for card, vector in zip(pending, vectors):
            cur.execute(
                _INSERT_CARD,
                (
                    card["card_type"], card["topic"], card["title"], card["summary"],
                    card["content"], card["audience"], card["severity_level"],
                    card["metric_name"], json.dumps({}), json.dumps({}), json.dumps([]),
                    card["source_label"], card["source_url_key"], json.dumps({}),
                ),
            )
            card_id = cur.fetchone()["knowledge_card_id"]
            cur.execute(
                _INSERT_EMBEDDING,
                (card_id, card["topic"], card["card_type"], vector, "BAAI/bge-large-en-v1.5"),
            )
        print(f"Inserted {len(pending)} card(s).")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="show pending cards, insert nothing")
    args = parser.parse_args()
    return seed(dry_run=args.list)


if __name__ == "__main__":
    sys.exit(main())
