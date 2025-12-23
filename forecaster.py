"""
Forecasting Module for Demand Forecast Engine - TimeGPT Only Version
Handles TimeGPT forecasting and simple baseline methods
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st
from nixtla import NixtlaClient
from config import *
from data_processor import categorize_sku_by_data_availability

# ============================================================================
# NAIVE & SIMPLE FORECASTING
# ============================================================================

def forecast_naive(
    df: pd.DataFrame,
    horizon: int,
    freq: str
) -> pd.DataFrame:
    """
    Naive forecast: repeat last value for all future periods
    
    Args:
        df: Historical data (unique_id, ds, y)
        horizon: Number of periods to forecast
        freq: Pandas frequency string
        
    Returns:
        Forecast DataFrame
    """
    forecasts = []
    
    for uid in df['unique_id'].unique():
        sku_data = df[df['unique_id'] == uid].sort_values('ds')
        last_value = sku_data['y'].iloc[-1]
        last_date = sku_data['ds'].iloc[-1]
        
        # Generate future dates
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({
            'unique_id': uid,
            'ds': future_dates,
            'forecast': last_value
        })
        forecasts.append(forecast_df)
    
    return pd.concat(forecasts, ignore_index=True)

def forecast_moving_average(
    df: pd.DataFrame,
    horizon: int,
    freq: str,
    window: int = 3
) -> pd.DataFrame:
    """
    Moving average forecast: use average of last N periods
    
    Args:
        df: Historical data (unique_id, ds, y)
        horizon: Number of periods to forecast
        freq: Pandas frequency string
        window: Number of periods to average (default: 3)
        
    Returns:
        Forecast DataFrame
    """
    forecasts = []
    
    for uid in df['unique_id'].unique():
        sku_data = df[df['unique_id'] == uid].sort_values('ds')
        
        # Calculate moving average of last N periods
        ma_value = sku_data['y'].tail(window).mean()
        last_date = sku_data['ds'].iloc[-1]
        
        # Generate future dates
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({
            'unique_id': uid,
            'ds': future_dates,
            'forecast': ma_value
        })
        forecasts.append(forecast_df)
    
    return pd.concat(forecasts, ignore_index=True)

# ============================================================================
# TIMEGPT FORECASTING
# ============================================================================

def forecast_timegpt(
    df_history: pd.DataFrame,
    df_future: Optional[pd.DataFrame],
    horizon: int,
    freq: str,
    api_key: str,
    include_intervals: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Generate forecast using TimeGPT
    
    Args:
        df_history: Historical data
        df_future: Future exogenous variables (promo)
        horizon: Forecast horizon
        freq: Pandas frequency
        api_key: Nixtla API key
        include_intervals: Whether to include prediction intervals
        
    Returns:
        Tuple of (forecast_df, error_message)
    """
    try:
        # Initialize client
        client = NixtlaClient(api_key=api_key)
        
        # Determine confidence levels
        level_list = TIMEGPT_CONFIG['confidence_levels'] if include_intervals else None
        
        # Make forecast
        forecast_df = client.forecast(
            df=df_history,
            X_df=df_future,
            h=horizon,
            level=level_list,
            freq=freq
        )
        
        # Round and clip to ensure positive integer forecasts
        forecast_cols = [col for col in forecast_df.columns if 'TimeGPT' in col]
        for col in forecast_cols:
            forecast_df[col] = np.ceil(forecast_df[col]).clip(lower=0)
        
        return forecast_df, None
        
    except Exception as e:
        error_msg = f"TimeGPT API error: {str(e)}"
        return None, error_msg

# ============================================================================
# MAIN FORECAST ORCHESTRATOR
# ============================================================================

def generate_forecasts(
    df_history: pd.DataFrame,
    df_future: Optional[pd.DataFrame],
    horizon: int,
    freq: str,
    frequency_type: str,
    api_key: str,
    naive_method: str = 'naive',
    quality_report: Optional[Dict] = None
) -> Dict:
    """
    Main orchestrator for generating all forecasts based on data availability
    
    Args:
        df_history: Historical data
        df_future: Future exogenous variables
        horizon: Forecast horizon
        freq: Pandas frequency
        frequency_type: 'Weekly' or 'Monthly'
        api_key: Nixtla API key
        naive_method: 'naive' or 'moving_average' for limited data
        quality_report: Data quality report from preprocessing
        
    Returns:
        Dictionary containing all forecast results and metadata
    """
    results = {
        'forecasts': {},
        'methods_used': {},
        'skipped': {},
        'warnings': [],
        'model_availability': {
            'timegpt': False,
        }
    }
    
    # Analyze each SKU's data availability
    sku_lengths = df_history.groupby('unique_id').size()
    
    # Categorize SKUs by forecast tier
    tier_distribution = {}
    for uid, length in sku_lengths.items():
        tier = categorize_sku_by_data_availability(length, frequency_type)
        tier_distribution[uid] = (tier, length)
    
    # Separate SKUs by tier
    naive_skus = [uid for uid, (tier, _) in tier_distribution.items() if tier == ForecastTier.NAIVE]
    timegpt_basic_skus = [uid for uid, (tier, _) in tier_distribution.items() if tier == ForecastTier.TIMEGPT_BASIC]
    timegpt_full_skus = [uid for uid, (tier, _) in tier_distribution.items() if tier == ForecastTier.TIMEGPT_FULL]
    skip_skus = [uid for uid, (tier, _) in tier_distribution.items() if tier == ForecastTier.SKIP]
    
    # Store skipped items
    for uid in skip_skus:
        results['skipped'][uid] = "Cold start (0 history points)"
    
    # Check for all zeros or no variance
    for uid in df_history['unique_id'].unique():
        sku_data = df_history[df_history['unique_id'] == uid]['y']
        if sku_data.sum() == 0:
            results['skipped'][uid] = "All sales are zero"
            # Remove from forecast lists
            if uid in naive_skus:
                naive_skus.remove(uid)
            if uid in timegpt_basic_skus:
                timegpt_basic_skus.remove(uid)
            if uid in timegpt_full_skus:
                timegpt_full_skus.remove(uid)
        elif sku_data.std() == 0:
            results['skipped'][uid] = "No variance (all same value)"
            # Remove from forecast lists
            if uid in naive_skus:
                naive_skus.remove(uid)
            if uid in timegpt_basic_skus:
                timegpt_basic_skus.remove(uid)
            if uid in timegpt_full_skus:
                timegpt_full_skus.remove(uid)
    
    # Process Naive/MA forecasts (1-3 data points)
    if len(naive_skus) > 0:
        st.info(f"📊 Processing {len(naive_skus)} SKUs with limited data (1-3 points) using {naive_method}...")
        df_naive = df_history[df_history['unique_id'].isin(naive_skus)]
        
        if naive_method == 'moving_average':
            naive_forecast = forecast_moving_average(df_naive, horizon, freq)
        else:
            naive_forecast = forecast_naive(df_naive, horizon, freq)
        
        results['forecasts']['naive'] = naive_forecast
        for uid in naive_skus:
            results['methods_used'][uid] = naive_method.upper()
    
    # Process TimeGPT Basic (4-7 points, no intervals)
    if len(timegpt_basic_skus) > 0:
        st.info(f"📊 Processing {len(timegpt_basic_skus)} SKUs with TimeGPT (basic, no intervals)...")
        df_basic = df_history[df_history['unique_id'].isin(timegpt_basic_skus)]
        df_future_basic = df_future[df_future['unique_id'].isin(timegpt_basic_skus)] if df_future is not None else None
        
        timegpt_basic, error = forecast_timegpt(df_basic, df_future_basic, horizon, freq, api_key, include_intervals=False)
        
        if timegpt_basic is not None:
            results['forecasts']['timegpt_basic'] = timegpt_basic
            results['model_availability']['timegpt'] = True
            for uid in timegpt_basic_skus:
                results['methods_used'][uid] = 'TIMEGPT_BASIC'
        else:
            results['warnings'].append(f"TimeGPT basic failed: {error}")
            # Fallback to naive for these SKUs
            st.warning(f"⚠️ TimeGPT basic failed, using naive fallback for {len(timegpt_basic_skus)} SKUs")
            df_fallback = df_history[df_history['unique_id'].isin(timegpt_basic_skus)]
            naive_fallback = forecast_naive(df_fallback, horizon, freq)
            results['forecasts']['timegpt_basic_fallback'] = naive_fallback
            for uid in timegpt_basic_skus:
                results['methods_used'][uid] = 'NAIVE_FALLBACK'
                results['warnings'].append(f"SKU {uid}: Fallback to naive due to TimeGPT error")
    
    # Process TimeGPT Full (8+ points, with intervals)
    if len(timegpt_full_skus) > 0:
        st.info(f"📊 Processing {len(timegpt_full_skus)} SKUs with TimeGPT (with confidence intervals)...")
        df_full = df_history[df_history['unique_id'].isin(timegpt_full_skus)]
        df_future_full = df_future[df_future['unique_id'].isin(timegpt_full_skus)] if df_future is not None else None
        
        timegpt_full, error = forecast_timegpt(df_full, df_future_full, horizon, freq, api_key, include_intervals=True)
        
        if timegpt_full is not None:
            results['forecasts']['timegpt_full'] = timegpt_full
            results['model_availability']['timegpt'] = True
            for uid in timegpt_full_skus:
                results['methods_used'][uid] = 'TIMEGPT_FULL'
        else:
            results['warnings'].append(f"TimeGPT full failed: {error}")
            # Fallback to naive for these SKUs
            st.warning(f"⚠️ TimeGPT full failed, using naive fallback for {len(timegpt_full_skus)} SKUs")
            df_fallback = df_history[df_history['unique_id'].isin(timegpt_full_skus)]
            naive_fallback = forecast_naive(df_fallback, horizon, freq)
            results['forecasts']['timegpt_full_fallback'] = naive_fallback
            for uid in timegpt_full_skus:
                results['methods_used'][uid] = 'NAIVE_FALLBACK'
                results['warnings'].append(f"SKU {uid}: Fallback to naive due to TimeGPT error")
    
    # Summary statistics
    results['summary'] = {
        'total_skus': len(sku_lengths),
        'forecasted': len(results['methods_used']),
        'skipped': len(results['skipped']),
        'tier_distribution': {
            'naive': len(naive_skus),
            'timegpt_basic': len(timegpt_basic_skus),
            'timegpt_full': len(timegpt_full_skus),
            'skipped': len(skip_skus),
        }
    }
    
    return results

# ============================================================================
# FORECAST CONSOLIDATION
# ============================================================================

def consolidate_forecasts(forecast_results: Dict) -> pd.DataFrame:
    """
    Consolidate all forecasts into a single DataFrame for export
    
    Args:
        forecast_results: Results from generate_forecasts()
        
    Returns:
        Consolidated forecast DataFrame
    """
    all_forecasts = []
    
    forecasts_dict = forecast_results['forecasts']
    methods_used = forecast_results['methods_used']
    
    # Process each forecast type
    for forecast_type, df in forecasts_dict.items():
        if 'naive' in forecast_type or 'fallback' in forecast_type:
            # Simple forecast (just forecast column)
            df_copy = df.copy()
            df_copy['method'] = df_copy['unique_id'].map(methods_used)
            all_forecasts.append(df_copy[['unique_id', 'ds', 'forecast', 'method']])
            
        elif 'timegpt' in forecast_type:
            # TimeGPT forecast
            df_copy = df.copy()
            df_copy['forecast'] = df_copy['TimeGPT']
            df_copy['method'] = df_copy['unique_id'].map(methods_used)
            
            # Base columns
            cols = ['unique_id', 'ds', 'forecast', 'method']
            
            # Include intervals if available
            if 'TimeGPT-lo-80' in df_copy.columns:
                cols.extend(['TimeGPT-lo-80', 'TimeGPT-hi-80', 'TimeGPT-lo-95', 'TimeGPT-hi-95'])
            
            all_forecasts.append(df_copy[cols])
    
    # Concatenate all forecasts
    if len(all_forecasts) > 0:
        consolidated = pd.concat(all_forecasts, ignore_index=True)
        return consolidated
    else:
        return pd.DataFrame()
