-- Retire the two TVOC records attributed to WHO. WHO publishes no TVOC guideline.
--
-- These were seeded as "WHO Indoor Air Quality Guidelines: Selected Pollutants 2010,
-- Chapter 7: Total VOCs" with 300 µg/m³ / 0.061 ppm. That document covers nine named
-- pollutants -- benzene, carbon monoxide, formaldehyde, naphthalene, nitrogen dioxide,
-- PAHs, radon, trichloroethylene, tetrachloroethylene -- and states no TVOC guideline
-- value at all. Its Chapter 7 is radon. The 300 / 300-1000 µg/m³ banding is Seifert's
-- five-level TVOC scheme, recommended by the German Committee on Indoor Air Guide Values
-- (UBA), which this seed already cites correctly at the 950 µg/m³ precautionary level.
--
-- This was not a cosmetic mislabel. VOC reads in ppm and the ppm twin carried the lowest
-- threshold of any VOC record, so strictest-applicable made it the GOVERNING threshold for
-- every VOC verdict the system produced -- each one citing a body that never published the
-- number. Replacements are seeded as UBA_TVOC_HYGIENIC / UBA_TVOC_HYGIENIC_PPM, same
-- figures, correct attribution.
--
-- Deactivated rather than deleted: every read path filters on is_active, so this removes
-- them from retrieval and from the citable set while leaving the rows for audit. Anything
-- already persisted that cites them keeps a resolvable record of what was claimed.

UPDATE env_guideline_records
   SET is_active = FALSE,
       updated_at = NOW()
 WHERE source_key IN ('WHO_IAQ_VOC_2010', 'WHO_IAQ_VOC_2010_PPM')
   AND metric = 'voc';
