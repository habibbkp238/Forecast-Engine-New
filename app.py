"""
Main Streamlit Application for Demand Forecast Engine - TimeGPT Only Version
Simplified 3-tab interface with granularity options
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Import custom modules
from config import *
from data_processor import *
from forecaster import *
from visualizer import *

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout=APP_LAYOUT
)

# Initialize session state
init_session_state()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_df_to_excel(dataframes_dict: Dict[str, pd.DataFrame]) -> bytes:
    """
    Convert multiple DataFrames to multi-sheet Excel file
    
    Args:
        dataframes_dict: Dictionary of {sheet_name: dataframe}
        
    Returns:
        Excel file as bytes
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

def display_quality_metrics(quality_report: Dict):
    """Display data quality metrics in a nice format"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total SKUs", quality_report['total_skus'])
    with col2:
        st.metric("Total Records", f"{quality_report['total_records']:,}")
    with col3:
        start_date = quality_report['date_range'][0].strftime('%Y-%m-%d')
        end_date = quality_report['date_range'][1].strftime('%Y-%m-%d')
        st.metric("Date Range", f"{start_date} to {end_date}")
    with col4:
        zero_pct = quality_report['zero_sales_ratio'] * 100
        st.metric("Zero Sales %", f"{zero_pct:.1f}%")

def display_forecast_summary(forecast_results: Dict):
    """Display forecast summary statistics"""
    summary = forecast_results['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total SKUs", summary['total_skus'])
    with col2:
        forecasted = summary['forecasted']
        pct = (forecasted / summary['total_skus']) * 100 if summary['total_skus'] > 0 else 0
        st.metric("Forecasted", f"{forecasted} ({pct:.1f}%)")
    with col3:
        timegpt_count = summary['tier_distribution'].get('timegpt_full', 0) + summary['tier_distribution'].get('timegpt_basic', 0)
        st.metric("TimeGPT Used", timegpt_count)
    with col4:
        skipped = summary['skipped']
        st.metric("Skipped", skipped, delta=None, delta_color="off")

# ============================================================================
# MAIN APP
# ============================================================================

st.title(APP_TITLE)
st.markdown("Professional demand forecasting powered by TimeGPT with flexible granularity options")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Data Upload & Preview",
    "⚙️ Configuration",
    "📈 Forecast & Results"
])

# ============================================================================
# TAB 1: DATA UPLOAD & PREVIEW
# ============================================================================

with tab1:
    st.header("Data Upload & Quality Check")
    
    # Guidelines
    with st.expander("📋 Data Preparation Guidelines", expanded=False):
        st.markdown(UI_TEXT['data_guidelines'])
    
    # File upload
    st.subheader("1. Upload Historical Data")
    uploaded_history = st.file_uploader(
        "Upload your historical sales data (.xlsx or .csv)",
        type=["xlsx", "csv"],
        key="history_upload"
    )
    
    if uploaded_history:
        # Read file
        with st.spinner("Reading file..."):
            df_raw = read_uploaded_file(uploaded_history)
        
        if df_raw is not None:
            st.success(f"✅ File loaded successfully! {len(df_raw):,} rows")
            
            # Store in session state
            st.session_state[SESSION_KEYS['uploaded_data']] = df_raw
            
            # Preview data
            st.subheader("2. Data Preview")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            # Quick validation
            st.subheader("3. Quick Validation")
            
            is_valid, missing_cols = validate_required_columns(df_raw)
            
            if not is_valid:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: " + ", ".join(REQUIRED_COLUMNS))
                st.stop()
            else:
                st.success("✅ All required columns present")
            
            # Check data types
            is_valid_types, type_errors = validate_data_types(df_raw)
            if not is_valid_types:
                st.error("❌ Data type issues found:")
                for error in type_errors:
                    st.write(f"  - {error}")
                st.stop()
            else:
                st.success("✅ Data types are valid")
            
            # Basic statistics
            st.subheader("4. Basic Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                unique_items = df_raw['item_name'].nunique() if 'item_name' in df_raw.columns else 0
                st.metric("Unique Items", unique_items)
            
            with col2:
                unique_wh = df_raw['wh'].nunique() if 'wh' in df_raw.columns else 0
                st.metric("Warehouses", unique_wh)
            
            with col3:
                unique_channel = df_raw['channel'].nunique() if 'channel' in df_raw.columns else 0
                st.metric("Channels", unique_channel)
            
            # Continue button
            st.markdown("---")
            if st.button("✅ Data looks good, continue to Configuration →", type="primary", use_container_width=True):
                st.success("Proceed to the Configuration tab to set forecast parameters")
    else:
        st.info("👆 Please upload your historical sales data to begin")

# ============================================================================
# TAB 2: CONFIGURATION
# ============================================================================

with tab2:
    st.header("Forecast Configuration")
    
    if SESSION_KEYS['uploaded_data'] not in st.session_state or st.session_state[SESSION_KEYS['uploaded_data']] is None:
        st.warning("⚠️ Please upload data in the 'Data Upload & Preview' tab first")
        st.stop()
    
    st.markdown("Configure your forecast parameters below:")
    
    # Time bucket selection
    st.subheader("1. Time Bucket")
    frequency = st.radio(
        "Select the frequency of your data:",
        ['Weekly', 'Monthly'],
        index=0,
        horizontal=True,
        help="Ensure your historical data matches this frequency"
    )
    
    # Forecast horizon
    st.subheader("2. Forecast Horizon")
    default_horizon = 12 if frequency == 'Monthly' else 26
    horizon = st.number_input(
        f"How many {frequency.lower()[:-2]}s ahead do you want to forecast?",
        min_value=4,
        max_value=104,
        value=default_horizon,
        step=1,
        help=f"Recommended: {default_horizon} {frequency.lower()[:-2]}s"
    )
    
    # Granularity settings
    st.subheader("3. Forecast Granularity")
    with st.expander("ℹ️ What is granularity?", expanded=False):
        st.markdown(UI_TEXT['granularity_info'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Warehouse Level:**")
        warehouse_level = st.radio(
            "Choose warehouse granularity:",
            options=list(GRANULARITY_OPTIONS['warehouse'].keys()),
            format_func=lambda x: GRANULARITY_OPTIONS['warehouse'][x],
            key="warehouse_level",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Channel Level:**")
        channel_level = st.radio(
            "Choose channel granularity:",
            options=list(GRANULARITY_OPTIONS['channel'].keys()),
            format_func=lambda x: GRANULARITY_OPTIONS['channel'][x],
            key="channel_level",
            label_visibility="collapsed"
        )
    
    # Show granularity preview
    preview_text = get_granularity_preview(warehouse_level, channel_level)
    st.info(f"📊 {preview_text}")
    
    # Data preprocessing
    st.subheader("4. Data Preprocessing")
    handle_outliers = st.toggle(
        "Adjust outliers using IQR method",
        value=True,
        help="Automatically cap extreme values to reduce noise in the forecast"
    )
    
    # Promotional features
    st.subheader("5. Promotional Features (Optional)")
    use_promo = st.toggle(
        "Include promotional features",
        value=False,
        help="Enable this if you have promotional data and want to model promo impact"
    )
    
    uploaded_future_promo = None
    if use_promo:
        st.info(UI_TEXT['promo_notice'])
        uploaded_future_promo = st.file_uploader(
            "Upload future promotional calendar",
            type=["xlsx", "csv"],
            key="promo_upload"
        )
        
        if uploaded_future_promo:
            st.success("✅ Promo calendar uploaded")
    
    # Naive forecast method
    st.subheader("6. Limited Data Handling")
    naive_method = st.radio(
        "For SKUs with only 1-3 data points, use:",
        ['naive', 'moving_average'],
        format_func=lambda x: "Last Value (Naive)" if x == 'naive' else "3-Period Moving Average",
        horizontal=True,
        help="This method will be used for SKUs that don't have enough history for TimeGPT"
    )
    
    # Save configuration
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Configuration Summary")
        config_summary = f"""
        - **Time Bucket**: {frequency}
        - **Horizon**: {horizon} {frequency.lower()[:-2]}s
        - **Warehouse Level**: {GRANULARITY_OPTIONS['warehouse'][warehouse_level]}
        - **Channel Level**: {GRANULARITY_OPTIONS['channel'][channel_level]}
        - **Outlier Handling**: {'Enabled' if handle_outliers else 'Disabled'}
        - **Promotional Features**: {'Enabled' if use_promo else 'Disabled'}
        - **Limited Data Method**: {naive_method.replace('_', ' ').title()}
        """
        st.markdown(config_summary)
    
    with col2:
        if st.button("Save & Continue →", type="primary", use_container_width=True):
            # Validate promo upload if needed
            if use_promo and uploaded_future_promo is None:
                st.error("❌ Please upload future promo calendar or disable promotional features")
            else:
                # Save config to session state
                st.session_state[SESSION_KEYS['config']] = {
                    'frequency': frequency,
                    'horizon': horizon,
                    'warehouse_level': warehouse_level,
                    'channel_level': channel_level,
                    'handle_outliers': handle_outliers,
                    'use_promo': use_promo,
                    'uploaded_future_promo': uploaded_future_promo,
                    'naive_method': naive_method,
                }
                st.success("✅ Configuration saved! Proceed to 'Forecast & Results' tab")

# ============================================================================
# TAB 3: FORECAST & RESULTS
# ============================================================================

with tab3:
    st.header("Run Forecast & View Results")
    
    # Check if config is set
    if SESSION_KEYS['config'] not in st.session_state or st.session_state[SESSION_KEYS['config']] is None:
        st.warning("⚠️ Please complete the Configuration tab first")
        st.stop()
    
    config = st.session_state[SESSION_KEYS['config']]
    
    # Display configuration
    st.subheader("Review Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Time Bucket:** {config['frequency']}")
        st.write(f"**Horizon:** {config['horizon']} periods")
    
    with col2:
        wh_label = "By Warehouse" if config['warehouse_level'] == GranularityLevel.BY_WAREHOUSE else "All Warehouses"
        ch_label = "By Channel" if config['channel_level'] == GranularityLevel.BY_CHANNEL else "All Channels"
        st.write(f"**Warehouse:** {wh_label}")
        st.write(f"**Channel:** {ch_label}")
    
    with col3:
        st.write(f"**Outliers:** {config['handle_outliers']}")
        st.write(f"**Promotions:** {config['use_promo']}")
    
    st.markdown("---")
    
    # Run forecast button
    run_forecast = st.button("🚀 Run Forecast", type="primary", use_container_width=True)
    
    if run_forecast:
        
        df_raw = st.session_state[SESSION_KEYS['uploaded_data']]
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # ================================================================
            # STEP 1: PREPROCESSING
            # ================================================================
            status_text.text("Step 1/3: Preprocessing data...")
            progress_bar.progress(10)
            
            df_history, quality_report, id_lookup = preprocess_historical_data(
                df_raw=df_raw,
                frequency=config['frequency'],
                warehouse_level=config['warehouse_level'],
                channel_level=config['channel_level'],
                handle_outliers=config['handle_outliers'],
                use_promo=config['use_promo']
            )
            
            # Store for later use
            st.session_state[SESSION_KEYS['id_lookup']] = id_lookup
            
            progress_bar.progress(30)
            st.success("✅ Step 1/3: Preprocessing complete")
            
            # Display quality metrics
            with st.expander("📊 Data Quality Report"):
                display_quality_metrics(quality_report)
            
            # Process future promo if needed
            df_future = None
            if config['use_promo'] and config['uploaded_future_promo'] is not None:
                df_promo_raw = read_uploaded_file(config['uploaded_future_promo'])
                if df_promo_raw is not None:
                    df_future = preprocess_future_promo(
                        df_promo_raw, 
                        id_lookup, 
                        config['warehouse_level'],
                        config['channel_level']
                    )
                    st.info(f"✅ Processed {len(df_future)} future promo records")
            
            # ================================================================
            # STEP 2: FORECASTING
            # ================================================================
            status_text.text("Step 2/3: Generating forecasts with TimeGPT...")
            progress_bar.progress(35)
            
            forecast_results = generate_forecasts(
                df_history=df_history,
                df_future=df_future,
                horizon=config['horizon'],
                freq=quality_report['frequency'],
                frequency_type=config['frequency'],
                api_key=st.secrets["NIXTLA_API_KEY"],
                naive_method=config['naive_method'],
                quality_report=quality_report
            )
            
            progress_bar.progress(85)
            st.success("✅ Step 2/3: Forecasting complete")
            
            # Display summary
            with st.expander("📊 Forecast Summary"):
                display_forecast_summary(forecast_results)
            
            # ================================================================
            # STEP 3: CONSOLIDATION
            # ================================================================
            status_text.text("Step 3/3: Consolidating results...")
            progress_bar.progress(90)
            
            # Consolidate forecasts
            consolidated_forecast = consolidate_forecasts(forecast_results)
            
            # Merge with historical data for visualization
            full_data = pd.merge(
                df_history[['unique_id', 'ds', 'y']],
                consolidated_forecast,
                on=['unique_id', 'ds'],
                how='outer'
            )
            
            # Store results
            st.session_state[SESSION_KEYS['forecast_results']] = {
                'forecast_data': full_data,
                'consolidated_forecast': consolidated_forecast,
                'forecast_results': forecast_results,
                'quality_report': quality_report,
            }
            
            progress_bar.progress(100)
            st.success("✅ Step 3/3: Complete!")
            
            status_text.text("Done! View results below.")
            
        except Exception as e:
            st.error(f"❌ Error during forecasting: {str(e)}")
            st.exception(e)
            progress_bar.progress(0)
            status_text.text("Forecast failed")
    
    # ========================================================================
    # RESULTS SECTION
    # ========================================================================
    
    if SESSION_KEYS['forecast_results'] in st.session_state and st.session_state[SESSION_KEYS['forecast_results']] is not None:
        
        st.markdown("---")
        st.header("📊 Forecast Results")
        
        # Get results from session state
        results = st.session_state[SESSION_KEYS['forecast_results']]
        forecast_data = results['forecast_data']
        consolidated_forecast = results['consolidated_forecast']
        forecast_results = results['forecast_results']
        quality_report = results['quality_report']
        id_lookup = st.session_state[SESSION_KEYS['id_lookup']]
        
        # Create sub-sections
        result_tab1, result_tab2, result_tab3 = st.tabs([
            "📊 Summary Dashboard",
            "🔍 SKU Detail View",
            "💾 Export Results"
        ])
        
        # ====================================================================
        # RESULT TAB 1: SUMMARY DASHBOARD
        # ====================================================================
        
        with result_tab1:
            st.subheader("Forecast Summary Dashboard")
            
            # KPI Cards
            st.markdown("### Key Metrics")
            display_forecast_summary(forecast_results)
            
            # Total forecast chart
            st.markdown("### Total Forecast Over Time")
            fig_summary = plot_forecast_summary(consolidated_forecast, config['frequency'])
            st.plotly_chart(fig_summary, use_container_width=True)
            
            # Forecast by attribute (if available)
            if 'category' in id_lookup.columns and id_lookup['category'].nunique() > 1:
                st.markdown("### Forecast by Category")
                fig_category = plot_forecast_by_attribute(consolidated_forecast, id_lookup, 'category')
                st.plotly_chart(fig_category, use_container_width=True)
            
            # SKU distribution
            st.markdown("### SKU Distribution by Forecast Method")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_dist = plot_sku_distribution(forecast_results)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.markdown("#### Method Breakdown")
                tier_dist = forecast_results['summary']['tier_distribution']
                for tier, count in tier_dist.items():
                    if count > 0:
                        pct = (count / forecast_results['summary']['total_skus']) * 100
                        st.write(f"**{tier.replace('_', ' ').title()}**: {count} ({pct:.1f}%)")
            
            # Data quality overview
            st.markdown("### Data Quality Overview")
            fig_quality = plot_data_quality_summary(quality_report)
            st.plotly_chart(fig_quality, use_container_width=True)
        
        # ====================================================================
        # RESULT TAB 2: SKU DETAIL VIEW
        # ====================================================================
        
        with result_tab2:
            st.subheader("Individual SKU Forecast")
            
            # Get all unique IDs
            all_skus = sorted(forecast_data['unique_id'].dropna().unique())
            
            if len(all_skus) == 0:
                st.warning("No forecasted SKUs available")
            else:
                # SKU selector
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    selected_sku = st.selectbox(
                        "Select SKU to view:",
                        options=all_skus,
                        index=0
                    )
                
                with col2:
                    # Get method used for this SKU
                    method_used = forecast_results['methods_used'].get(selected_sku, 'Unknown')
                    st.metric("Method Used", method_used)
                
                # Display SKU attributes
                sku_info = id_lookup[id_lookup['unique_id'] == selected_sku].iloc[0]
                
                st.markdown("#### SKU Information")
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.write(f"**Item:** {sku_info.get('item_name', 'N/A')}")
                with info_cols[1]:
                    if 'wh' in sku_info and config['warehouse_level'] == GranularityLevel.BY_WAREHOUSE:
                        st.write(f"**Warehouse:** {sku_info.get('wh', 'N/A')}")
                    else:
                        st.write(f"**Warehouse:** All Combined")
                with info_cols[2]:
                    if 'channel' in sku_info and config['channel_level'] == GranularityLevel.BY_CHANNEL:
                        st.write(f"**Channel:** {sku_info.get('channel', 'N/A')}")
                    else:
                        st.write(f"**Channel:** All Combined")
                with info_cols[3]:
                    if 'category' in sku_info:
                        st.write(f"**Category:** {sku_info.get('category', 'N/A')}")
                
                # Plot forecast
                st.markdown("#### Forecast Visualization")
                
                # Check if intervals are available
                has_intervals = 'TimeGPT-lo-80' in consolidated_forecast.columns
                
                fig_sku = plot_sku_forecast(
                    history_df=forecast_data[forecast_data['unique_id'] == selected_sku],
                    forecast_df=consolidated_forecast[consolidated_forecast['unique_id'] == selected_sku],
                    unique_id=selected_sku,
                    show_intervals=has_intervals
                )
                st.plotly_chart(fig_sku, use_container_width=True)
                
                # Show forecast table
                st.markdown("#### Forecast Data Table")
                sku_forecast = consolidated_forecast[consolidated_forecast['unique_id'] == selected_sku][
                    ['ds', 'forecast', 'method']
                ].sort_values('ds')
                
                # Rename for display
                sku_forecast_display = sku_forecast.rename(columns={
                    'ds': 'Date',
                    'forecast': 'Forecast',
                    'method': 'Method'
                })
                
                st.dataframe(sku_forecast_display, use_container_width=True, hide_index=True)
        
        # ====================================================================
        # RESULT TAB 3: EXPORT
        # ====================================================================
        
        with result_tab3:
            st.subheader("Export Forecast Results")
            
            st.markdown("""
            Download your forecast results as an Excel file with multiple sheets:
            - **TimeGPT Forecast**: Main forecast ready for Module 2
            - **Forecast Metadata**: Which method was used for each SKU
            - **Skipped Items**: SKUs that couldn't be forecasted with reasons
            """)
            
            st.markdown("---")
            
            # Prepare export data
            try:
                # Sheet 1: TimeGPT Forecast
                export_forecast = pd.merge(
                    consolidated_forecast[['unique_id', 'ds', 'forecast']],
                    id_lookup,
                    on='unique_id',
                    how='left'
                )
                export_forecast = export_forecast.rename(columns={'ds': 'date'})
                
                # Reorder columns based on granularity
                base_cols = ['date', 'item_name']
                if config['warehouse_level'] == GranularityLevel.BY_WAREHOUSE and 'wh' in export_forecast.columns:
                    base_cols.append('wh')
                if config['channel_level'] == GranularityLevel.BY_CHANNEL and 'channel' in export_forecast.columns:
                    base_cols.append('channel')
                base_cols.append('forecast')
                
                optional_cols = [col for col in ['category', 'class', 'extention'] if col in export_forecast.columns]
                export_forecast = export_forecast[base_cols + optional_cols]
                
                # Sheet 2: Forecast Metadata
                metadata_records = []
                for uid, method in forecast_results['methods_used'].items():
                    sku_info = id_lookup[id_lookup['unique_id'] == uid].iloc[0] if uid in id_lookup['unique_id'].values else {}
                    
                    history_length = len(forecast_data[forecast_data['unique_id'] == uid]['y'].dropna())
                    
                    record = {
                        'unique_id': uid,
                        'item_name': sku_info.get('item_name', ''),
                        'method_used': method,
                        'history_length': history_length,
                        'forecast_horizon': config['horizon']
                    }
                    
                    if config['warehouse_level'] == GranularityLevel.BY_WAREHOUSE:
                        record['wh'] = sku_info.get('wh', '')
                    
                    if config['channel_level'] == GranularityLevel.BY_CHANNEL:
                        record['channel'] = sku_info.get('channel', '')
                    
                    metadata_records.append(record)
                
                export_metadata = pd.DataFrame(metadata_records)
                
                # Sheet 3: Skipped Items
                skipped_records = []
                for uid, reason in forecast_results['skipped'].items():
                    sku_info = id_lookup[id_lookup['unique_id'] == uid].iloc[0] if uid in id_lookup['unique_id'].values else {}
                    
                    record = {
                        'unique_id': uid,
                        'item_name': sku_info.get('item_name', ''),
                        'skip_reason': reason
                    }
                    
                    if config['warehouse_level'] == GranularityLevel.BY_WAREHOUSE:
                        record['wh'] = sku_info.get('wh', '')
                    
                    if config['channel_level'] == GranularityLevel.BY_CHANNEL:
                        record['channel'] = sku_info.get('channel', '')
                    
                    skipped_records.append(record)
                
                export_skipped = pd.DataFrame(skipped_records) if len(skipped_records) > 0 else pd.DataFrame()
                
                # Create Excel file
                sheets_dict = {
                    EXPORT_CONFIG['sheet_names']['forecast']: export_forecast,
                    EXPORT_CONFIG['sheet_names']['metadata']: export_metadata,
                }
                
                if len(export_skipped) > 0:
                    sheets_dict[EXPORT_CONFIG['sheet_names']['skipped']] = export_skipped
                
                excel_data = convert_df_to_excel(sheets_dict)
                
                # Download button
                st.download_button(
                    label="📥 Download Forecast Results (Excel)",
                    data=excel_data,
                    file_name=EXPORT_CONFIG['file_name'],
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary",
                    use_container_width=True
                )
                
                # Show preview
                st.markdown("---")
                st.markdown("### Preview: Forecast Output (First 20 rows)")
                st.dataframe(export_forecast.head(20), use_container_width=True, hide_index=True)
                
                # Export statistics
                st.markdown("---")
                st.markdown("### Export Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Forecasted SKUs", len(export_forecast['item_name'].unique()))
                with col2:
                    st.metric("Total Forecast Records", len(export_forecast))
                with col3:
                    st.metric("Sheets in Export", len(sheets_dict))
                
            except Exception as e:
                st.error(f"Error preparing export: {str(e)}")
                st.exception(e)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Demand Forecast Engine - TimeGPT Edition | Powered by Nixtla TimeGPT
    </div>
    """,
    unsafe_allow_html=True
)
