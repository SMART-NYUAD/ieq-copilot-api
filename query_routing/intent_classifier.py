"""Intent types for query routing."""

from enum import Enum


class IntentType(str, Enum):
    DEFINITION_EXPLANATION = "definition_explanation"
    CURRENT_STATUS_DB = "current_status_db"
    POINT_LOOKUP_DB = "point_lookup_db"
    AGGREGATION_DB = "aggregation_db"
    COMPARISON_DB = "comparison_db"
    ANOMALY_ANALYSIS_DB = "anomaly_analysis_db"
    FORECAST_DB = "forecast_db"
    VIEWER_CONTROL = "viewer_control"
    HEATMAP_CONTROL = "heatmap_control"
    DOWNLOAD_DATA = "download_data"
    IFC_MODEL_QA = "ifc_model_qa"
    UNKNOWN_FALLBACK = "unknown_fallback"
