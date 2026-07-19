-- Materialize approved candidate features from MEMBER_FEATURES_BASE.
-- One column per whitelist feature; formulas mirror src/feature_catalog.py.
-- Only accepted candidates are used downstream — this view exposes all four so
-- the challenger loop can select whichever the gate validated.
USE WAREHOUSE CARECOST_WH;
USE SCHEMA CARECOST_DEMO.ANALYTICS;

CREATE OR REPLACE VIEW MEMBER_FEATURES_CANDIDATES AS
SELECT
    b.*,
    b.COST_30D / GREATEST(b.COST_90D / 3, 1)                          AS COST_ACCELERATION,
    b.DISTINCT_PROVIDER_COUNT_90D / GREATEST(b.CLAIM_COUNT_90D, 1)    AS PROVIDER_FRAGMENTATION,
    b.ED_COUNT_30D / GREATEST(b.ED_COUNT_90D / 3, 1)                  AS ED_ACCELERATION,
    b.INPATIENT_COST_90D / GREATEST(b.COST_90D, 1)                    AS INPATIENT_COST_SHARE
FROM MEMBER_FEATURES_BASE b;
