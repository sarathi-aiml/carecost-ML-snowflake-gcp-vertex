-- Reference residual segmentation in pure SQL.
-- The notebook discovers the segment with a shallow decision tree and writes the
-- aggregate summary via write_pandas; this query is the Snowflake-native
-- equivalent for inspecting segment-level residuals directly in the warehouse.
USE WAREHOUSE CARECOST_WH;
USE SCHEMA CARECOST_DEMO.ANALYTICS;

SELECT
    p.RUN_ID,
    IFF(p.ED_COUNT_BUCKET = 'HIGH_ED', 'segment_high_ed', 'segment_other') AS SEGMENT_ID,
    COUNT(*)                       AS MEMBER_COUNT,
    ROUND(AVG(p.ACTUAL_COST), 2)   AS MEAN_ACTUAL_COST,
    ROUND(AVG(p.PREDICTED_COST), 2) AS MEAN_PREDICTED_COST,
    ROUND(AVG(p.RESIDUAL), 2)      AS MEAN_RESIDUAL,
    ROUND(AVG(p.ABSOLUTE_ERROR), 2) AS MEAN_ABSOLUTE_ERROR
FROM (
    SELECT
        pr.*,
        IFF(f.ED_COUNT_30D >= 2, 'HIGH_ED', 'OTHER') AS ED_COUNT_BUCKET
    FROM MODEL_PREDICTIONS pr
    JOIN MEMBER_FEATURES_BASE f
      ON f.MEMBER_ID = pr.MEMBER_ID AND f.INDEX_DATE = pr.INDEX_DATE
    WHERE pr.MODEL_TYPE = 'XGB_BASELINE'
) p
GROUP BY p.RUN_ID, SEGMENT_ID
ORDER BY MEAN_RESIDUAL DESC;
