"""
Visualization Module for Demand Forecast Engine - TimeGPT Only Version
Handles plotting and chart generation (simplified - no ML comparison)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
from config import *

# ============================================================================
# INDIVIDUAL SKU FORECAST PLOT
# ============================================================================

def plot_sku_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    unique_id: str,
    show_intervals: bool = True
) -> go.Figure:
    """
    Create interactive plot for individual SKU forecast
    
    Args:
        history_df: Historical data
        forecast_df: Forecast data
        unique_id: SKU identifier
        show_intervals: Whether to show prediction intervals
        
    Returns:
        Plotly figure
    """
    # Filter data for selected SKU
    hist_sku = history_df[history_df['unique_id'] == unique_id].sort_values('ds')
    fcst_sku = forecast_df[forecast_df['unique_id'] == unique_id].sort_values('ds')
    
    # Create figure
    fig = go.Figure()
    
    # Add actual history
    fig.add_trace(go.Scatter(
        x=hist_sku['ds'],
        y=hist_sku['y'],
        mode='lines+markers',
        name='Actual',
        line=dict(color=CHART_CONFIG['colors']['actual']),
        marker=dict(size=6)
    ))
    
    # Add forecast
    if 'forecast' in fcst_sku.columns:
        fig.add_trace(go.Scatter(
            x=fcst_sku['ds'],
            y=fcst_sku['forecast'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color=CHART_CONFIG['colors']['forecast'], dash='dash'),
            marker=dict(size=6)
        ))
    
    # Add prediction intervals if available
    if show_intervals and 'TimeGPT-lo-80' in fcst_sku.columns:
        # 95% interval
        fig.add_trace(go.Scatter(
            x=fcst_sku['ds'],
            y=fcst_sku['TimeGPT-hi-95'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=fcst_sku['ds'],
            y=fcst_sku['TimeGPT-lo-95'],
            mode='lines',
            line=dict(width=0),
            fillcolor=CHART_CONFIG['colors']['interval_95'],
            fill='tonexty',
            name='95% Interval',
            hoverinfo='skip'
        ))
        
        # 80% interval
        fig.add_trace(go.Scatter(
            x=fcst_sku['ds'],
            y=fcst_sku['TimeGPT-hi-80'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=fcst_sku['ds'],
            y=fcst_sku['TimeGPT-lo-80'],
            mode='lines',
            line=dict(width=0),
            fillcolor=CHART_CONFIG['colors']['interval_80'],
            fill='tonexty',
            name='80% Interval',
            hoverinfo='skip'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Forecast for: {unique_id}",
        xaxis_title="Date",
        yaxis_title="Sales Units",
        height=CHART_CONFIG['height'],
        template=CHART_CONFIG['template'],
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# ============================================================================
# SUMMARY VISUALIZATIONS
# ============================================================================

def plot_forecast_summary(
    forecast_df: pd.DataFrame,
    frequency: str
) -> go.Figure:
    """
    Plot total forecast across all SKUs over time
    
    Args:
        forecast_df: Consolidated forecast data
        frequency: 'Weekly' or 'Monthly'
        
    Returns:
        Plotly figure
    """
    # Aggregate forecast by date
    summary = forecast_df.groupby('ds')['forecast'].sum().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=summary['ds'],
        y=summary['forecast'],
        mode='lines+markers',
        name='Total Forecast',
        line=dict(color='purple', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(128,0,128,0.2)'
    ))
    
    fig.update_layout(
        title=f"Total Forecast Across All SKUs ({frequency})",
        xaxis_title="Date",
        yaxis_title="Total Units",
        height=400,
        template=CHART_CONFIG['template'],
        hovermode='x unified'
    )
    
    return fig

def plot_forecast_by_attribute(
    forecast_df: pd.DataFrame,
    id_lookup: pd.DataFrame,
    attribute_col: str = 'category'
) -> go.Figure:
    """
    Plot forecast breakdown by attribute (category, channel, etc.)
    
    Args:
        forecast_df: Forecast data
        id_lookup: ID lookup with attributes
        attribute_col: Column name for attribute to plot
        
    Returns:
        Plotly figure
    """
    # Merge with attribute
    df = pd.merge(forecast_df, id_lookup[['unique_id', attribute_col]], on='unique_id', how='left')
    
    # Aggregate by attribute and date
    summary = df.groupby(['ds', attribute_col])['forecast'].sum().reset_index()
    
    fig = px.line(
        summary,
        x='ds',
        y='forecast',
        color=attribute_col,
        title=f"Forecast by {attribute_col.title()}",
        labels={'forecast': 'Forecasted Units', 'ds': 'Date'}
    )
    
    fig.update_layout(
        height=450,
        template=CHART_CONFIG['template'],
        hovermode='x unified'
    )
    
    return fig

def plot_sku_distribution(
    forecast_results: Dict
) -> go.Figure:
    """
    Plot distribution of SKUs by forecast method used
    
    Args:
        forecast_results: Results from forecasting
        
    Returns:
        Plotly figure
    """
    tier_dist = forecast_results['summary']['tier_distribution']
    
    labels = []
    values = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFEAA7']
    
    for tier, count in tier_dist.items():
        if count > 0:
            labels.append(tier.replace('_', ' ').title())
            values.append(count)
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="SKU Distribution by Forecast Method",
        height=400,
        template=CHART_CONFIG['template']
    )
    
    return fig

# ============================================================================
# DATA QUALITY VISUALIZATIONS
# ============================================================================

def plot_data_quality_summary(quality_report: Dict) -> go.Figure:
    """
    Create visualization of data quality metrics
    
    Args:
        quality_report: Quality report from preprocessing
        
    Returns:
        Plotly figure
    """
    sku_stats = quality_report['sku_statistics']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Sales Distribution", "Data Coverage"),
        specs=[[{"type": "box"}, {"type": "histogram"}]]
    )
    
    # Box plot of average sales
    fig.add_trace(
        go.Box(
            y=sku_stats['avg_sales'],
            name='Avg Sales',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # Histogram of record counts
    fig.add_trace(
        go.Histogram(
            x=sku_stats['record_count'],
            name='Data Points',
            marker_color='lightgreen',
            nbinsx=20
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Average Sales per Period", row=1, col=1)
    fig.update_xaxes(title_text="Number of Data Points", row=1, col=2)
    fig.update_yaxes(title_text="Number of SKUs", row=1, col=2)
    
    fig.update_layout(
        title="Data Quality Overview",
        height=400,
        template=CHART_CONFIG['template'],
        showlegend=False
    )
    
    return fig

# ============================================================================
# FORECAST HEATMAP
# ============================================================================

def plot_forecast_heatmap(
    forecast_df: pd.DataFrame,
    id_lookup: pd.DataFrame,
    top_n: int = 20
) -> go.Figure:
    """
    Create heatmap of forecast for top N SKUs
    
    Args:
        forecast_df: Forecast data
        id_lookup: ID lookup table
        top_n: Number of top SKUs to show
        
    Returns:
        Plotly figure
    """
    # Calculate total forecast per SKU
    total_by_sku = forecast_df.groupby('unique_id')['forecast'].sum().sort_values(ascending=False)
    top_skus = total_by_sku.head(top_n).index
    
    # Filter for top SKUs
    df_top = forecast_df[forecast_df['unique_id'].isin(top_skus)].copy()
    
    # Pivot for heatmap
    pivot = df_top.pivot(index='unique_id', columns='ds', values='forecast')
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Blues',
        hovertemplate='SKU: %{y}<br>Date: %{x}<br>Forecast: %{z:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Forecast Heatmap - Top {top_n} SKUs by Volume",
        xaxis_title="Date",
        yaxis_title="SKU",
        height=600,
        template=CHART_CONFIG['template']
    )
    
    return fig
