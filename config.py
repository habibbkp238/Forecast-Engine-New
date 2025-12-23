"""
Configuration file for Demand Forecast Engine - TimeGPT Only Version
Contains all constants, settings, and configuration parameters
"""

from typing import Dict, List, Tuple
import streamlit as st

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_TITLE = "🚀 Demand Forecast Engine (TimeGPT Edition)"
APP_LAYOUT = "wide"
PAGE_ICON = "📊"

# ============================================================================
# DATA REQUIREMENTS
# ============================================================================

# Required columns in historical data
REQUIRED_COLUMNS = ['ds', 'item_name', 'wh', 'channel', 'y']

# Optional columns
OPTIONAL_COLUMNS = ['category', 'class', 'extention', 'promo_flag']

# Static attribute columns (will be forward-filled)
STATIC_COLUMNS = ['item_name', 'wh', 'channel', 'category', 'class', 'extention']

# ============================================================================
# GRANULARITY OPTIONS
# ============================================================================

class GranularityLevel:
    """Defines granularity levels for forecasting"""
    # Warehouse options
    BY_WAREHOUSE = "by_warehouse"
    ALL_WAREHOUSES = "all_warehouses"
    
    # Channel options
    BY_CHANNEL = "by_channel"
    ALL_CHANNELS = "all_channels"

GRANULARITY_OPTIONS = {
    'warehouse': {
        'by_warehouse': 'By Origin Warehouse (separate forecast per warehouse)',
        'all_warehouses': 'All Warehouses Combined (aggregate across warehouses)'
    },
    'channel': {
        'by_channel': 'By Channel (separate forecast per channel)',
        'all_channels': 'All Channels Combined (aggregate across channels)'
    }
}

# ============================================================================
# FORECAST TIERS (Based on data availability)
# ============================================================================

class ForecastTier:
    """Defines different forecasting approaches based on data availability"""
    SKIP = "SKIP"
    NAIVE = "NAIVE"
    MOVING_AVERAGE = "MOVING_AVERAGE"
    TIMEGPT_BASIC = "TIMEGPT_BASIC"
    TIMEGPT_FULL = "TIMEGPT_FULL"

# Minimum data points required for each tier
TIER_THRESHOLDS = {
    'skip_below': 1,           # Skip if less than 1 point (cold start)
    'naive_range': (1, 3),     # 1-3 points: use naive or MA
    'timegpt_basic': 8,        # 4-7 points: TimeGPT without intervals
    'timegpt_full_weekly': 8,  # 8+ points: TimeGPT with intervals (weekly)
    'timegpt_full_monthly': 8  # 8+ points: TimeGPT with intervals (monthly)
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# TimeGPT settings
TIMEGPT_CONFIG = {
    'confidence_levels': [80, 95],
    'min_samples_for_intervals_weekly': 8,
    'min_samples_for_intervals_monthly': 8,
}

# ============================================================================
# DATA QUALITY THRESHOLDS
# ============================================================================

DATA_QUALITY = {
    'max_outlier_ratio': 0.3,      # Warn if >30% outliers
    'intermittent_zero_ratio': 0.75,  # >75% zeros = intermittent
    'lumpy_zero_ratio': 0.50,      # >50% zeros = lumpy
}

# ============================================================================
# FREQUENCY MAPPINGS
# ============================================================================

FREQUENCY_MAP = {
    'Weekly': {
        'days': ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN'],
        'default': 'W',
        'expected_diff_days': 7,
    },
    'Monthly': {
        'pandas_freq': 'MS',  # Month start
        'expected_diff_days': 30,
    }
}

# ============================================================================
# OUTLIER HANDLING
# ============================================================================

OUTLIER_CONFIG = {
    'method': 'iqr',
    'iqr_multiplier': 1.5,
    'cap_upper': True,
    'cap_lower': False,
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

CHART_CONFIG = {
    'height': 500,
    'template': 'plotly_white',
    'colors': {
        'actual': '#1f77b4',
        'timegpt': '#ff7f0e',
        'forecast': '#2ca02c',
        'interval_80': 'rgba(0,176,246,0.2)',
        'interval_95': 'rgba(0,100,80,0.2)',
    }
}

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_MESSAGES = {
    'no_file': "⚠️ Please upload your historical data file.",
    'missing_columns': "❌ Missing required columns: {columns}",
    'no_promo_file': "⚠️ Promotional features are enabled but no promo file uploaded.",
    'invalid_frequency': "❌ Could not detect valid frequency in your data.",
    'timegpt_api_error': "❌ TimeGPT API error: {error}",
    'insufficient_data': "⚠️ SKU has insufficient data for forecasting.",
    'cold_start': "ℹ️ Cold start items (0 history) are not handled by this tool.",
}

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_CONFIG = {
    'file_name': 'demand_forecast_output.xlsx',
    'sheet_names': {
        'forecast': 'TimeGPT_Forecast',
        'metadata': 'Forecast_Metadata',
        'skipped': 'Skipped_Items',
    }
}

# ============================================================================
# UI TEXT & LABELS
# ============================================================================

UI_TEXT = {
    'data_guidelines': """
    ### 📋 Data Preparation Guidelines
    
    #### Required Columns:
    - `ds`: Date column (must match your selected frequency)
    - `item_name`: Product/SKU identifier
    - `wh`: Warehouse code
    - `channel`: Sales channel
    - `y`: Sales quantity (numeric)
    
    #### Optional Columns:
    - `category`, `class`, `extention`: Product attributes
    - `promo_flag`: Promotional indicator (1 = promo, 0 = no promo)
    
    #### Frequency Requirements:
    - **Weekly**: Dates should be 7 days apart (e.g., every Monday)
    - **Monthly**: Dates should be month-start (e.g., 2024-01-01, 2024-02-01)
    
    #### File Format:
    - Accepted: `.xlsx` or `.csv`
    - One row per item-date-warehouse-channel combination
    """,
    
    'promo_notice': """
    ⚠️ **Important**: Your future promo calendar must match your selected frequency.
    - If you selected **Weekly**, upload weekly promo data
    - If you selected **Monthly**, upload monthly promo data
    """,
    
    'granularity_info': """
    ℹ️ **Granularity Settings** determine the level at which forecasts are generated:
    
    - **By Origin Warehouse**: Separate forecast for each warehouse
    - **All Warehouses Combined**: Single forecast aggregated across all warehouses
    
    - **By Channel**: Separate forecast for each channel (Online, Retail, etc.)
    - **All Channels Combined**: Single forecast aggregated across all channels
    
    Choose the level that matches your planning needs. Module 2 can handle allocation/breakdown later.
    """,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_frequency_config(frequency: str) -> Dict:
    """Get frequency configuration based on user selection"""
    return FREQUENCY_MAP.get(frequency, {})

def get_tier_threshold(frequency: str, tier: str) -> int:
    """Get data threshold for a specific forecasting tier"""
    if tier == 'timegpt_full':
        key = f'timegpt_full_{frequency.lower()}'
        return TIER_THRESHOLDS.get(key, 8)
    return TIER_THRESHOLDS.get(tier, 0)

def get_unique_id_columns(warehouse_level: str, channel_level: str) -> List[str]:
    """
    Get columns to use for creating unique_id based on granularity
    
    Args:
        warehouse_level: 'by_warehouse' or 'all_warehouses'
        channel_level: 'by_channel' or 'all_channels'
        
    Returns:
        List of columns to use for unique_id
    """
    columns = ['item_name']
    
    if warehouse_level == GranularityLevel.BY_WAREHOUSE:
        columns.append('wh')
    
    if channel_level == GranularityLevel.BY_CHANNEL:
        columns.append('channel')
    
    return columns

def get_groupby_columns(warehouse_level: str, channel_level: str) -> List[str]:
    """
    Get columns to group by for aggregation
    
    Args:
        warehouse_level: 'by_warehouse' or 'all_warehouses'
        channel_level: 'by_channel' or 'all_channels'
        
    Returns:
        List of columns to group by
    """
    columns = ['item_name', 'ds']
    
    if warehouse_level == GranularityLevel.BY_WAREHOUSE:
        columns.insert(1, 'wh')  # Add after item_name
    
    if channel_level == GranularityLevel.BY_CHANNEL:
        if 'wh' in columns:
            columns.insert(2, 'channel')  # Add after wh
        else:
            columns.insert(1, 'channel')  # Add after item_name
    
    return columns

def get_granularity_preview(warehouse_level: str, channel_level: str) -> str:
    """
    Get preview text showing forecast level
    
    Args:
        warehouse_level: 'by_warehouse' or 'all_warehouses'
        channel_level: 'by_channel' or 'all_channels'
        
    Returns:
        Preview string
    """
    parts = ['item']
    
    if warehouse_level == GranularityLevel.BY_WAREHOUSE:
        parts.append('wh')
    
    if channel_level == GranularityLevel.BY_CHANNEL:
        parts.append('channel')
    
    return f"Forecasting at **[{' + '.join(parts)}]** level"

# ============================================================================
# SESSION STATE KEYS
# ============================================================================

SESSION_KEYS = {
    'uploaded_data': 'uploaded_data',
    'processed_data': 'processed_data',
    'forecast_results': 'forecast_results',
    'skipped_items': 'skipped_items',
    'id_lookup': 'id_lookup',
    'config': 'user_config',
}

def init_session_state():
    """Initialize all session state variables"""
    for key in SESSION_KEYS.values():
        if key not in st.session_state:
            st.session_state[key] = None
