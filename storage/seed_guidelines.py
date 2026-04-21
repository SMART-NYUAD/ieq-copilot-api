"""
Seed script for env_guideline_records table.
Run once: python -m storage.seed_guidelines
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

try:
    from storage.postgres_client import get_cursor
    from storage.embeddings import embed_texts
except ImportError:
    from .postgres_client import get_cursor
    from .embeddings import embed_texts


_GUIDELINE_EMBED_DIM = 1536


def _normalize_embedding_dim(embedding: List[float], target_dim: int = _GUIDELINE_EMBED_DIM) -> List[float]:
    values = list(embedding or [])
    if len(values) == target_dim:
        return values
    if len(values) > target_dim:
        return values[:target_dim]
    return values + [0.0] * (target_dim - len(values))


GUIDELINE_RECORDS: List[Dict[str, Any]] = [
    # -- CO2 -----------------------------------------------------------
    {
        "source_key": "ASHRAE_62_1_2022",
        "source_label": "ANSI/ASHRAE Standard 62.1-2022",
        "source_url": "https://www.ashrae.org/technical-resources/bookstore/standards-62-1-62-2",
        "section_ref": "Section 6.2.2 and User's Manual Commentary",
        "publication_year": 2022,
        "metric": "co2",
        "citation_tier": "regulatory",
        "claim_text": (
            "ASHRAE 62.1-2022 does NOT set a CO2 concentration limit. "
            "CO2 is used as a proxy indicator for adequate ventilation "
            "relative to outdoor air (~420 ppm). The commonly cited "
            "1000 ppm indoor target represents approximately 700 ppm "
            "above outdoor levels and is used as a ventilation adequacy "
            "indicator in practice, but this value does not appear as a "
            "normative limit in ASHRAE 62.1. The standard specifies "
            "ventilation rates by space type, not CO2 thresholds."
        ),
        "embed_text": (
            "ASHRAE 62.1 CO2 carbon dioxide ventilation limit "
            "1000 ppm concentration indoor air quality standard"
        ),
        "threshold_value": None,
        "threshold_type": None,
        "threshold_unit": "ppm",
        "threshold_condition": None,
        "caveat_text": (
            "Any CO2 ppm classification in this system is derived from "
            "industry practice and research, not ASHRAE 62.1 normative text. "
            "Never cite ASHRAE 62.1 as a CO2 concentration requirement."
        ),
    },
    {
        "source_key": "RESET_AIR_V2",
        "source_label": "RESET Air Standard v2.1 — Commercial Interiors",
        "source_url": "https://reset.build/standard/air",
        "section_ref": "Section 4: Performance Thresholds",
        "publication_year": 2021,
        "metric": "co2",
        "citation_tier": "regulatory",
        "claim_text": (
            "RESET Air Standard v2 requires CO2 at or below 1000 ppm "
            "for Grade A certification and at or below 1200 ppm for "
            "Grade B certification during occupied hours, measured by "
            "continuous monitoring sensors."
        ),
        "embed_text": (
            "RESET Air CO2 carbon dioxide 1000 ppm Grade A 1200 ppm "
            "Grade B certification commercial interiors monitoring"
        ),
        "threshold_value": 1000,
        "threshold_type": "max",
        "threshold_unit": "ppm",
        "threshold_condition": "Grade A, occupied hours",
        "caveat_text": (
            "RESET Air certification requires sensor calibration and "
            "continuous monitoring. Spot readings alone do not determine "
            "certification status."
        ),
    },
    {
        "source_key": "ALLEN_ET_AL_2016",
        "source_label": "Allen et al. 2016, Environmental Health Perspectives",
        "source_url": "https://doi.org/10.1289/ehp.1510037",
        "section_ref": "Table 3: Cognitive Function Scores by CO2 Condition",
        "publication_year": 2016,
        "metric": "co2",
        "citation_tier": "research",
        "claim_text": (
            "A Harvard study (Allen et al. 2016) found that CO2 at "
            "approximately 1000 ppm was associated with a 15% decline "
            "in cognitive performance scores compared to 550 ppm. "
            "At 2500 ppm, scores declined by 50% across most domains "
            "including crisis response and strategy. "
            "This is a research finding, not a regulatory standard."
        ),
        "embed_text": (
            "CO2 cognitive performance concentration 1000 ppm 2500 ppm "
            "Harvard study research brain function decision making "
            "indoor air quality"
        ),
        "threshold_value": 1000,
        "threshold_type": "max",
        "threshold_unit": "ppm",
        "threshold_condition": "cognitive performance research threshold",
        "caveat_text": (
            "This is a single study finding. No regulatory body has "
            "adopted these values as enforceable limits. "
            "Frame as: 'research suggests' not 'the standard requires'."
        ),
    },
    # -- PM2.5 ---------------------------------------------------------
    {
        "source_key": "EPA_PM25_NAAQS_2024",
        "source_label": "EPA National Ambient Air Quality Standards for PM2.5 (revised 2024)",
        "source_url": "https://www.epa.gov/pm-pollution/national-ambient-air-quality-standards-naaqs-pm",
        "section_ref": "40 CFR Part 50, February 2024 revision",
        "publication_year": 2024,
        "metric": "pm25",
        "citation_tier": "regulatory",
        "claim_text": (
            "The EPA 24-hour PM2.5 NAAQS is 35 µg/m³ "
            "(98th percentile, averaged over 3 years). "
            "The annual PM2.5 NAAQS was revised in February 2024 "
            "to 9 µg/m³, reduced from the prior 12 µg/m³. "
            "These are outdoor ambient standards. No US federal "
            "indoor PM2.5 limit exists. AQI Good category: 0–9.0 µg/m³; "
            "Moderate: 9.1–35.4; Unhealthy for Sensitive Groups: 35.5–55.4; "
            "Unhealthy: 55.5–125.4; Very Unhealthy: 125.5–225.4; "
            "Hazardous: >225.4 µg/m³."
        ),
        "embed_text": (
            "EPA PM2.5 NAAQS 35 micrograms annual 9 AQI classification "
            "good moderate unhealthy hazardous particulate matter "
            "air quality standard 2024"
        ),
        "threshold_value": 35,
        "threshold_type": "max",
        "threshold_unit": "µg/m³",
        "threshold_condition": "24-hour average",
        "caveat_text": (
            "These are outdoor ambient standards. Indoor PM2.5 has no "
            "separate US federal limit. Indoor sources and filtration "
            "mean indoor levels can differ substantially from outdoor."
        ),
    },
    {
        "source_key": "WHO_AQG_2021",
        "source_label": "WHO Global Air Quality Guidelines 2021",
        "source_url": "https://www.who.int/publications/i/item/9789240034228",
        "section_ref": "Chapter 6.1, Table 6.1",
        "publication_year": 2021,
        "metric": "pm25",
        "citation_tier": "regulatory",
        "claim_text": (
            "WHO 2021 guidelines recommend annual mean PM2.5 below "
            "5 µg/m³ and 24-hour mean below 15 µg/m³. "
            "These are stricter than EPA NAAQS and represent "
            "health-protective targets rather than regulatory limits. "
            "WHO interim targets: IT-1 35/15, IT-2 25/10, IT-3 15/5 "
            "(24h/annual µg/m³)."
        ),
        "embed_text": (
            "WHO PM2.5 air quality guideline 15 micrograms 24 hour "
            "5 annual health target 2021 stricter than EPA"
        ),
        "threshold_value": 15,
        "threshold_type": "max",
        "threshold_unit": "µg/m³",
        "threshold_condition": "24-hour mean guideline",
        "caveat_text": (
            "WHO guidelines are health-protective targets, not legally "
            "enforceable limits. They are stricter than most national "
            "regulations including US EPA NAAQS."
        ),
    },
    # -- TVOC ----------------------------------------------------------
    {
        "source_key": "RESET_AIR_V2_TVOC",
        "source_label": "RESET Air Standard v2.1 — Commercial Interiors",
        "source_url": "https://reset.build/standard/air",
        "section_ref": "Section 4: Performance Thresholds",
        "publication_year": 2021,
        "metric": "tvoc",
        "citation_tier": "regulatory",
        "claim_text": (
            "RESET Air Standard v2 requires TVOC at or below 500 µg/m³ "
            "for Grade A certification and at or below 1000 µg/m³ for "
            "Grade B certification during occupied hours."
        ),
        "embed_text": (
            "RESET Air TVOC VOC total volatile organic compounds "
            "500 micrograms Grade A 1000 Grade B certification"
        ),
        "threshold_value": 500,
        "threshold_type": "max",
        "threshold_unit": "µg/m³",
        "threshold_condition": "Grade A, occupied hours",
        "caveat_text": (
            "TVOC readings vary significantly by sensor technology. "
            "Different sensors respond differently to VOC compounds. "
            "RESET requires sensor-specific calibration for certification."
        ),
    },
    {
        "source_key": "WELL_V2_A04",
        "source_label": "WELL Building Standard v2, Feature A04",
        "source_url": "https://v2.wellcertified.com/v/en/air/feature/4",
        "section_ref": "Feature A04: Volatile Compounds",
        "publication_year": 2020,
        "metric": "tvoc",
        "citation_tier": "regulatory",
        "claim_text": (
            "WELL Building Standard v2 Feature A04 requires TVOC levels "
            "not to exceed 500 µg/m³ as a long-term occupancy average "
            "in occupied spaces seeking WELL certification."
        ),
        "embed_text": (
            "WELL TVOC volatile organic compounds 500 micrograms "
            "building standard certification indoor air"
        ),
        "threshold_value": 500,
        "threshold_type": "max",
        "threshold_unit": "µg/m³",
        "threshold_condition": "long-term occupancy average",
        "caveat_text": None,
    },
    {
        "source_key": "WHO_IAQ_TVOC_2010",
        "source_label": "WHO Indoor Air Quality Guidelines: Selected Pollutants 2010",
        "source_url": "https://www.who.int/publications/i/item/9789289002134",
        "section_ref": "Chapter 7: Total VOCs",
        "publication_year": 2010,
        "metric": "tvoc",
        "citation_tier": "research",
        "claim_text": (
            "WHO 2010 indoor air quality guidelines indicate TVOC below "
            "300 µg/m³ as a comfort range, 300–3000 µg/m³ as a multifactorial "
            "exposure range requiring investigation of sources, and above "
            "3000 µg/m³ as discomfort range with potential health effects. "
            "These are guidance values, not enforceable regulatory limits."
        ),
        "embed_text": (
            "WHO TVOC total volatile organic compounds 300 3000 "
            "comfort discomfort indoor air quality guidance 2010"
        ),
        "threshold_value": 300,
        "threshold_type": "max",
        "threshold_unit": "µg/m³",
        "threshold_condition": "comfort range upper boundary",
        "caveat_text": (
            "WHO TVOC guidance is research-based, not a regulatory "
            "standard. Frame as: 'WHO guidance suggests' not "
            "'the WHO standard requires'."
        ),
    },
    # -- TEMPERATURE ---------------------------------------------------
    {
        "source_key": "ASHRAE_55_2023_COMFORT",
        "source_label": "ANSI/ASHRAE Standard 55-2023",
        "source_url": "https://www.ashrae.org/technical-resources/bookstore/standard-55",
        "section_ref": "Section 5.3.2",
        "publication_year": 2023,
        "metric": "temperature",
        "citation_tier": "regulatory",
        "claim_text": (
            "ASHRAE 55-2023 defines thermal comfort compliance as PMV "
            "between -0.5 and +0.5. This is NOT a fixed temperature range. "
            "For typical office conditions (1.0–1.1 met, 0.5–0.65 clo, "
            "still air, 50% RH), this approximately corresponds to "
            "20–23.5°C in heating season and 23–26°C in cooling season. "
            "These ranges shift with clothing, activity, humidity, and "
            "air speed. ASHRAE 55 explicitly does not address CO2, "
            "PM2.5, acoustics, or illumination."
        ),
        "embed_text": (
            "ASHRAE 55 temperature thermal comfort PMV comfort zone "
            "20 23 26 degrees office heating cooling season occupant"
        ),
        "threshold_value": None,
        "threshold_type": None,
        "threshold_unit": "degC",
        "threshold_condition": "PMV -0.5 to +0.5, context-dependent",
        "caveat_text": (
            "A single air temperature reading is insufficient for "
            "ASHRAE 55 compliance assessment. Full PMV calculation "
            "requires mean radiant temperature, air speed, humidity, "
            "clothing insulation, and metabolic rate."
        ),
    },
    {
        "source_key": "RESET_AIR_V2_TEMP",
        "source_label": "RESET Air Standard v2.1 — Commercial Interiors",
        "source_url": "https://reset.build/standard/air",
        "section_ref": "Section 4: Performance Thresholds",
        "publication_year": 2021,
        "metric": "temperature",
        "citation_tier": "regulatory",
        "claim_text": (
            "RESET Air Standard v2 requires indoor temperature between "
            "18°C and 28°C during occupied hours for certified "
            "commercial interiors."
        ),
        "embed_text": (
            "RESET Air temperature 18 28 degrees Celsius occupied hours "
            "commercial interiors certification range"
        ),
        "threshold_value": 28,
        "threshold_type": "range_max",
        "threshold_unit": "degC",
        "threshold_condition": "occupied hours",
        "caveat_text": None,
    },
    # -- HUMIDITY ------------------------------------------------------
    {
        "source_key": "ASHRAE_62_1_2022_HUM",
        "source_label": "ANSI/ASHRAE Standard 62.1-2022",
        "source_url": "https://www.ashrae.org/technical-resources/bookstore/standards-62-1-62-2",
        "section_ref": "Section 5.10",
        "publication_year": 2022,
        "metric": "humidity",
        "citation_tier": "regulatory",
        "claim_text": (
            "ASHRAE 62.1-2022 Section 5.10 requires indoor relative "
            "humidity in occupied spaces to be maintained below 65% RH "
            "to inhibit microbial growth. The standard does not set a "
            "lower humidity limit for IAQ purposes."
        ),
        "embed_text": (
            "ASHRAE 62.1 humidity relative humidity 65 percent mold "
            "microbial growth indoor air quality limit maximum"
        ),
        "threshold_value": 65,
        "threshold_type": "max",
        "threshold_unit": "percent RH",
        "threshold_condition": "occupied spaces, IAQ requirement",
        "caveat_text": (
            "ASHRAE 55 sets no humidity limits for thermal comfort. "
            "The 65% limit is an IAQ/mold prevention requirement "
            "from 62.1, not a comfort guideline."
        ),
    },
    {
        "source_key": "EPA_INDOOR_HUMIDITY",
        "source_label": "EPA: The Inside Story — A Guide to Indoor Air Quality",
        "source_url": "https://www.epa.gov/indoor-air-quality-iaq/inside-story-guide-indoor-air-quality",
        "section_ref": "Humidity and Comfort section",
        "publication_year": 2022,
        "metric": "humidity",
        "citation_tier": "regulatory",
        "claim_text": (
            "EPA indoor air quality guidance recommends maintaining "
            "relative humidity between 30% and 50% for optimal health "
            "and comfort. Below 30% RH increases respiratory irritation "
            "and static electricity. Above 50% promotes dust mites "
            "and mold growth."
        ),
        "embed_text": (
            "EPA humidity 30 50 percent optimal health comfort "
            "dry low high indoor air quality guidance"
        ),
        "threshold_value": 50,
        "threshold_type": "range_max",
        "threshold_unit": "percent RH",
        "threshold_condition": "optimal comfort and health range",
        "caveat_text": None,
    },
    {
        "source_key": "RESET_AIR_V2_HUM",
        "source_label": "RESET Air Standard v2.1 — Commercial Interiors",
        "source_url": "https://reset.build/standard/air",
        "section_ref": "Section 4: Performance Thresholds",
        "publication_year": 2021,
        "metric": "humidity",
        "citation_tier": "regulatory",
        "claim_text": (
            "RESET Air Standard v2 requires indoor relative humidity "
            "between 30% and 60% RH during occupied hours for certified "
            "commercial interiors."
        ),
        "embed_text": (
            "RESET Air humidity relative humidity 30 60 percent "
            "occupied hours commercial certification range"
        ),
        "threshold_value": 60,
        "threshold_type": "range_max",
        "threshold_unit": "percent RH",
        "threshold_condition": "occupied hours",
        "caveat_text": None,
    },
    # -- LIGHT ---------------------------------------------------------
    {
        "source_key": "WELL_V2_L07",
        "source_label": "WELL Building Standard v2, Feature L07",
        "source_url": "https://v2.wellcertified.com/v/en/light/feature/7",
        "section_ref": "Feature L07: Visual Lighting Design",
        "publication_year": 2020,
        "metric": "light",
        "citation_tier": "regulatory",
        "claim_text": (
            "WELL v2 Feature L07 requires minimum 300 lux measured on "
            "the workplane (0.8m height) for general office tasks, and "
            "minimum 500 lux for detailed work areas. "
            "Values above 1000 lux may cause glare discomfort. "
            "These are workplane illuminance targets for occupied hours."
        ),
        "embed_text": (
            "WELL light lux illuminance 300 500 workplane office "
            "task lighting minimum glare 1000 certification"
        ),
        "threshold_value": 300,
        "threshold_type": "min",
        "threshold_unit": "lux",
        "threshold_condition": "workplane at 0.8m, general office tasks",
        "caveat_text": (
            "Your sensors may not measure at workplane height. "
            "Ceiling-mounted sensors typically read higher than "
            "workplane illuminance. Apply appropriate correction "
            "before comparing to WELL targets."
        ),
    },
    # -- SOUND ---------------------------------------------------------
    {
        "source_key": "WELL_V2_S05",
        "source_label": "WELL Building Standard v2, Feature S05",
        "source_url": "https://v2.wellcertified.com/v/en/sound/feature/5",
        "section_ref": "Feature S05: Noise Levels",
        "publication_year": 2020,
        "metric": "sound",
        "citation_tier": "regulatory",
        "claim_text": (
            "WELL v2 Feature S05 requires background noise in open "
            "offices not to exceed 45 dBA, in private offices not to "
            "exceed 40 dBA, and in conference/focus rooms not to exceed "
            "35 dBA. These are background noise level limits, "
            "not instantaneous peak measurements."
        ),
        "embed_text": (
            "WELL noise sound 45 dBA open office 40 private 35 "
            "conference room background acoustic certification"
        ),
        "threshold_value": 45,
        "threshold_type": "max",
        "threshold_unit": "dBA",
        "threshold_condition": "background noise level, open office",
        "caveat_text": (
            "These limits apply to background noise levels (LAeq), "
            "not instantaneous readings. A single sensor reading above "
            "45 dB does not constitute a WELL exceedance."
        ),
    },
    {
        "source_key": "WHO_NOISE_2018",
        "source_label": "WHO Environmental Noise Guidelines for the European Region 2018",
        "source_url": "https://www.who.int/publications/i/item/9789289053563",
        "section_ref": "Chapter 4, Table of recommendations",
        "publication_year": 2018,
        "metric": "sound",
        "citation_tier": "regulatory",
        "claim_text": (
            "WHO 2018 Environmental Noise Guidelines recommend indoor "
            "noise below 35 dB LAeq for concentration tasks and "
            "sleep protection. Noise above 55 dB LAeq is associated "
            "with cardiovascular health effects with long-term exposure. "
            "These apply to environmental noise, not occupant-generated noise."
        ),
        "embed_text": (
            "WHO noise 35 dB concentration tasks 55 cardiovascular "
            "health effects indoor LAeq environmental noise guidelines"
        ),
        "threshold_value": 35,
        "threshold_type": "max",
        "threshold_unit": "dB LAeq",
        "threshold_condition": "concentration tasks, indoor",
        "caveat_text": (
            "WHO guidelines use time-averaged LAeq measurements. "
            "Instantaneous dB readings differ from LAeq. "
            "Short peaks above threshold do not constitute exceedance."
        ),
    },
    # -- IEQ INDEX -----------------------------------------------------
    {
        "source_key": "INTERNAL_IEQ_INDEX",
        "source_label": "Internal IEQ Composite Model",
        "source_url": None,
        "section_ref": "System-internal methodology",
        "publication_year": None,
        "metric": "ieq",
        "citation_tier": "internal",
        "claim_text": (
            "The IEQ index is an internal composite score combining "
            "sub-indices for indoor air quality (IIAQ), thermal comfort "
            "(ITC), acoustic comfort (IAC), and illumination (IIL). "
            "Score bands: >75 high quality, 51-75 medium quality, "
            "26-50 moderate quality, ≤25 low quality. "
            "This scoring methodology has no direct equivalent in "
            "any external published standard."
        ),
        "embed_text": (
            "IEQ index composite score internal IIAQ ITC IAC IIL "
            "sub-index high medium moderate low quality indoor "
            "environmental quality"
        ),
        "threshold_value": 75,
        "threshold_type": "range_min",
        "threshold_unit": "index",
        "threshold_condition": "high quality band lower boundary",
        "caveat_text": (
            "Always label IEQ index scores as internal system metrics. "
            "Never present them as externally standardized ratings. "
            "Users should not compare IEQ scores across different "
            "systems that may use different methodologies."
        ),
    },
]


def seed_guidelines() -> None:
    """Insert all guideline records with embeddings."""
    print(f"Seeding {len(GUIDELINE_RECORDS)} guideline records...")

    embed_texts_list = [r["embed_text"] for r in GUIDELINE_RECORDS]
    print("Generating embeddings...")
    embeddings = embed_texts(embed_texts_list)

    if len(embeddings) != len(GUIDELINE_RECORDS):
        print(
            f"ERROR: Expected {len(GUIDELINE_RECORDS)} embeddings, "
            f"got {len(embeddings)}"
        )
        sys.exit(1)

    inserted = 0
    skipped = 0

    with get_cursor() as cur:
        for record, embedding in zip(GUIDELINE_RECORDS, embeddings):
            normalized_embedding = _normalize_embedding_dim(embedding)
            try:
                cur.execute(
                    """
                    INSERT INTO env_guideline_records (
                        source_key, source_label, source_url,
                        section_ref, publication_year, metric,
                        citation_tier, claim_text, embed_text,
                        threshold_value, threshold_type,
                        threshold_unit, threshold_condition,
                        caveat_text, embedding
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT DO NOTHING
                """,
                    (
                        record["source_key"],
                        record["source_label"],
                        record.get("source_url"),
                        record.get("section_ref"),
                        record.get("publication_year"),
                        record["metric"],
                        record["citation_tier"],
                        record["claim_text"],
                        record["embed_text"],
                        record.get("threshold_value"),
                        record.get("threshold_type"),
                        record.get("threshold_unit"),
                        record.get("threshold_condition"),
                        record.get("caveat_text"),
                        normalized_embedding,
                    ),
                )
                inserted += 1
            except Exception as e:
                print(f"  SKIP {record['source_key']}: {e}")
                skipped += 1

    print(f"Done. Inserted: {inserted}, Skipped: {skipped}")


if __name__ == "__main__":
    seed_guidelines()
