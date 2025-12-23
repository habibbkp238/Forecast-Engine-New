"""
Data Processing Module for Demand Forecast Engine - TimeGPT Only Version
Handles data validation, cleaning, preprocessing, aggregation by granularity
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
from utilsforecast.preprocessing import fill_gaps
from config import *

# ============================================================================
# FILE READING
# ============================================================================

def read_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Read uploaded Excel or CSV file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        DataFrame or None if error
    """
    try:
        if uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload .xlsx or .csv")
            return None
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None

# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_required_columns(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Check if all required columns are present
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing_cols) == 0, missing_cols

def validate_data_types(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate data types of key columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check if 'y' is numeric
    try:
        pd.to_numeric(df['y'])
    except:
        errors.append("Column 'y' (sales) must be numeric")
    
    # Check if 'ds' can be converted to datetime
    try:
        pd.to_datetime(df['ds'])
    except:
        errors.append("Column 'ds' (date) must be in valid date format")
    
    return len(errors) == 0, errors

def check_negative_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for negative sales values and flag them
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with flagged issues
    """
    negative_count = (df['y'] < 0).sum()
    if negative_count > 0:
        st.warning(f"⚠️ Found {negative_count} rows with negative sales. These will be set to 0.")
        df.loc[df['y'] < 0, 'y'] = 0
    return df

# ============================================================================
# GRANULARITY AGGREGATION
# ============================================================================

def aggregate_by_granularity(
    df: pd.DataFrame,
    warehouse_level: str,
    channel_level: str,
    use_promo: bool = False
) -> pd.DataFrame:
    """
    Aggregate data based on selected granularity levels
    
    Args:
        df: Input DataFrame
        warehouse_level: 'by_warehouse' or 'all_warehouses'
        channel_level: 'by_channel' or 'all_channels'
        use_promo: Whether promotional features are used
        
    Returns:
        Aggregated DataFrame
    """
    # Get groupby columns based on granularity
    groupby_cols = get_groupby_columns(warehouse_level, channel_level)
    
    # Define aggregation rules
    agg_dict = {'y': 'sum'}
    
    if use_promo and 'promo_flag' in df.columns:
        agg_dict['promo_flag'] = 'max'  # Use max to preserve promo indicator
    
    # Add static columns (take first value)
    static_cols = ['category', 'class', 'extention']
    for col in static_cols:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    # Keep warehouse and channel if needed (for later reference)
    if warehouse_level == GranularityLevel.ALL_WAREHOUSES and 'wh' in df.columns:
        # Store original warehouse info for reference but don't group by it
        pass
    
    if channel_level == GranularityLevel.ALL_CHANNELS and 'channel' in df.columns:
        # Store original channel info for reference but don't group by it
        pass
    
    # Aggregate
    df_agg = df.groupby(groupby_cols, as_index=False).agg(agg_dict)
    
    # Add back warehouse column if aggregated (for tracking)
    if warehouse_level == GranularityLevel.ALL_WAREHOUSES:
        df_agg['wh'] = 'ALL_WH'
    
    # Add back channel column if aggregated (for tracking)
    if channel_level == GranularityLevel.ALL_CHANNELS:
        df_agg['channel'] = 'ALL_CHANNEL'
    
    return df_agg

def create_unique_id(
    df: pd.DataFrame,
    warehouse_level: str,
    channel_level: str
) -> pd.DataFrame:
    """
    Create unique identifier based on granularity
    
    Args:
        df: Input DataFrame
        warehouse_level: 'by_warehouse' or 'all_warehouses'
        channel_level: 'by_channel' or 'all_channels'
        
    Returns:
        DataFrame with unique_id column added
    """
    try:
        id_columns = get_unique_id_columns(warehouse_level, channel_level)
        
        # Create unique_id by joining selected columns
        df['unique_id'] = df[id_columns[0]].astype(str)
        for col in id_columns[1:]:
            df['unique_id'] = df['unique_id'] + '_' + df[col].astype(str)
        
        return df
    except KeyError as e:
        st.error(f"❌ Missing required column for unique ID: {e}")
        raise

# ============================================================================
# DATA CLEANING
# ============================================================================

def adjust_outliers_iqr(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Detect and cap outliers using IQR method
    
    Args:
        df: Input DataFrame with 'unique_id' and 'y' columns
        
    Returns:
        Tuple of (cleaned DataFrame, number of outliers adjusted)
    """
    def cap_outliers_group(series: pd.Series) -> pd.Series:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (OUTLIER_CONFIG['iqr_multiplier'] * IQR)
        
        if IQR > 0 and upper_bound > Q1:
            return series.clip(upper=upper_bound)
        return series
    
    df['y_adjusted'] = df.groupby('unique_id')['y'].transform(cap_outliers_group)
    cap_count = (df['y_adjusted'] < df['y']).sum()
    
    df['y'] = df['y_adjusted']
    df = df.drop(columns=['y_adjusted'])
    
    return df, cap_count

# ============================================================================
# FREQUENCY DETECTION & GAP FILLING
# ============================================================================

def detect_frequency(df: pd.DataFrame, selected_frequency: str) -> str:
    """
    Detect or validate frequency of time series data
    
    Args:
        df: Input DataFrame with 'ds' column
        selected_frequency: User-selected frequency ('Weekly' or 'Monthly')
        
    Returns:
        Pandas frequency string (e.g., 'W-MON', 'MS')
    """
    if selected_frequency == 'Monthly':
        return 'MS'
    
    # For weekly, detect the day of week
    time_diffs = df.sort_values(by=['unique_id', 'ds']).groupby('unique_id')['ds'].diff().dt.days
    inferred_freq_days = int(time_diffs.mode().iloc[0]) if len(time_diffs.mode()) > 0 else 7
    
    if inferred_freq_days % 7 != 0:
        st.warning(f"⚠️ Inferred frequency is {inferred_freq_days} days (not exactly 7). Assuming weekly.")
        return 'W'
    
    # Detect day of week
    start_day_code = df['ds'].min().dayofweek
    days = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']
    freq = days[start_day_code]
    
    st.info(f"✅ Detected frequency: {freq}")
    return freq

def fill_missing_dates(df: pd.DataFrame, freq: str, use_promo: bool = False) -> pd.DataFrame:
    """
    Fill missing dates in time series using utilsforecast
    
    Args:
        df: Input DataFrame
        freq: Pandas frequency string
        use_promo: Whether promotional features are used
        
    Returns:
        DataFrame with complete date range
    """
    # Use utilsforecast to fill gaps
    df_filled = fill_gaps(df, freq=freq, id_col='unique_id', time_col='ds')
    
    # Fill NaN values created by gap filling
    fill_values = {'y': 0}
    if use_promo and 'promo_flag' in df_filled.columns:
        fill_values['promo_flag'] = 0
    
    df_filled = df_filled.fillna(value=fill_values)
    
    # Forward-fill static columns
    static_cols = ['item_name', 'wh', 'channel', 'category', 'class', 'extention']
    static_cols_to_ffill = [col for col in static_cols if col in df_filled.columns]
    df_filled[static_cols_to_ffill] = df_filled.groupby('unique_id')[static_cols_to_ffill].ffill()
    
    return df_filled

# ============================================================================
# DATA QUALITY ANALYSIS
# ============================================================================

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Perform comprehensive data quality analysis
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'total_skus': df['unique_id'].nunique(),
        'total_records': len(df),
        'date_range': (df['ds'].min(), df['ds'].max()),
        'missing_values': df.isnull().sum().to_dict(),
        'negative_sales': (df['y'] < 0).sum(),
        'zero_sales_ratio': (df['y'] == 0).sum() / len(df),
    }
    
    # Analyze by SKU
    sku_stats = df.groupby('unique_id').agg({
        'y': ['count', 'sum', 'mean', 'std'],
        'ds': ['min', 'max']
    }).reset_index()
    
    sku_stats.columns = ['unique_id', 'record_count', 'total_sales', 'avg_sales', 'std_sales', 'start_date', 'end_date']
    
    quality_report['sku_statistics'] = sku_stats
    quality_report['skus_with_zeros_only'] = (sku_stats['total_sales'] == 0).sum()
    quality_report['skus_no_variance'] = (sku_stats['std_sales'] == 0).sum()
    
    return quality_report

def categorize_sku_by_data_availability(sku_length: int, frequency: str) -> str:
    """
    Determine forecast tier based on data availability
    
    Args:
        sku_length: Number of historical data points
        frequency: 'Weekly' or 'Monthly'
        
    Returns:
        Forecast tier string
    """
    if sku_length == 0:
        return ForecastTier.SKIP
    elif sku_length <= 3:
        return ForecastTier.NAIVE
    elif sku_length < TIER_THRESHOLDS['timegpt_basic']:
        return ForecastTier.TIMEGPT_BASIC
    else:
        return ForecastTier.TIMEGPT_FULL

# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_historical_data(
    df_raw: pd.DataFrame,
    frequency: str,
    warehouse_level: str,
    channel_level: str,
    handle_outliers: bool = True,
    use_promo: bool = False
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """
    Complete preprocessing pipeline for historical data
    
    Args:
        df_raw: Raw uploaded DataFrame
        frequency: User-selected frequency ('Weekly' or 'Monthly')
        warehouse_level: 'by_warehouse' or 'all_warehouses'
        channel_level: 'by_channel' or 'all_channels'
        handle_outliers: Whether to adjust outliers
        use_promo: Whether promotional features are used
        
    Returns:
        Tuple of (processed_df, quality_report, id_lookup_table)
    """
    # Step 1: Validate columns
    is_valid, missing_cols = validate_required_columns(df_raw)
    if not is_valid:
        raise ValueError(ERROR_MESSAGES['missing_columns'].format(columns=', '.join(missing_cols)))
    
    # Step 2: Validate data types
    is_valid, errors = validate_data_types(df_raw)
    if not is_valid:
        raise ValueError('\n'.join(errors))
    
    # Step 3: Convert date column
    df = df_raw.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Step 4: Check for negative sales
    df = check_negative_sales(df)
    
    # Step 5: Aggregate by granularity BEFORE creating unique_id
    st.info(f"📊 Aggregating data by granularity...")
    df = aggregate_by_granularity(df, warehouse_level, channel_level, use_promo)
    
    # Step 6: Create unique_id based on granularity
    df = create_unique_id(df, warehouse_level, channel_level)
    
    # Step 7: Handle outliers
    outlier_count = 0
    if handle_outliers:
        df, outlier_count = adjust_outliers_iqr(df)
        if outlier_count > 0:
            st.info(f"✅ Adjusted {outlier_count} outlier data points using IQR method.")
    
    # Step 8: Detect frequency
    freq = detect_frequency(df, frequency)
    
    # Step 9: Fill missing dates
    df = fill_missing_dates(df, freq, use_promo)
    
    # Step 10: Create ID lookup table
    id_cols = ['unique_id', 'item_name']
    if warehouse_level == GranularityLevel.BY_WAREHOUSE:
        id_cols.append('wh')
    if channel_level == GranularityLevel.BY_CHANNEL:
        id_cols.append('channel')
    
    # Add optional columns if present
    for col in ['category', 'class', 'extention']:
        if col in df.columns:
            id_cols.append(col)
    
    id_lookup = df[id_cols].drop_duplicates()
    
    # Step 11: Prepare final columns
    base_cols = ['unique_id', 'ds', 'y']
    optional_cols = ['category', 'class', 'extention']
    if use_promo and 'promo_flag' in df.columns:
        base_cols.append('promo_flag')
    
    final_cols = base_cols + [col for col in optional_cols if col in df.columns]
    df_final = df[final_cols]
    
    # Step 12: Quality report
    quality_report = analyze_data_quality(df_final)
    quality_report['outliers_adjusted'] = outlier_count
    quality_report['frequency'] = freq
    quality_report['granularity'] = {
        'warehouse_level': warehouse_level,
        'channel_level': channel_level
    }
    
    return df_final, quality_report, id_lookup

def preprocess_future_promo(
    df_promo_raw: pd.DataFrame,
    id_lookup: pd.DataFrame,
    warehouse_level: str,
    channel_level: str
) -> pd.DataFrame:
    """
    Preprocess future promotional calendar
    
    Args:
        df_promo_raw: Raw promo calendar DataFrame
        id_lookup: ID lookup table from historical data
        warehouse_level: 'by_warehouse' or 'all_warehouses'
        channel_level: 'by_channel' or 'all_channels'
        
    Returns:
        Processed promo DataFrame
    """
    # Create unique_id
    df_promo = create_unique_id(df_promo_raw.copy(), warehouse_level, channel_level)
    
    # Convert date
    df_promo['ds'] = pd.to_datetime(df_promo['ds'])
    
    # Get groupby columns
    groupby_cols = get_groupby_columns(warehouse_level, channel_level)
    
    # Aggregate duplicates
    df_promo = df_promo.groupby(groupby_cols, as_index=False).agg({'promo_flag': 'max'})
    
    # Recreate unique_id after aggregation
    df_promo = create_unique_id(df_promo, warehouse_level, channel_level)
    
    # Only keep SKUs that exist in historical data
    valid_ids = id_lookup['unique_id'].unique()
    df_promo = df_promo[df_promo['unique_id'].isin(valid_ids)]
    
    return df_promo
