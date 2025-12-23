# 🚀 Demand Forecast Engine - TimeGPT Edition v2.0

A streamlined, production-ready demand forecasting application powered exclusively by TimeGPT with flexible granularity options.

## ✨ What's New in V2.0

### Simplified Architecture
- ✅ **TimeGPT Only** - Single, reliable forecasting model
- ✅ **No ML Dependencies** - Compatible with Python 3.13+
- ✅ **No Backtesting** - Faster execution, simpler workflow
- ✅ **3-Tab Interface** - Streamlined user experience

### New Granularity Features
- ✅ **Warehouse Aggregation** - Forecast by warehouse OR aggregate all warehouses
- ✅ **Channel Aggregation** - Forecast by channel OR aggregate all channels
- ✅ **Flexible Combinations** - 4 granularity levels to match your planning needs

---

## 📊 Granularity Options Explained

### Four Forecasting Levels:

**Level 1: Most Granular (Item + Warehouse + Channel)**
- Separate forecast for each item-warehouse-channel combination
- Example: SKU001_WH_A_Online, SKU001_WH_B_Retail
- Use when: You need detailed forecasts per location and channel

**Level 2: Warehouse Aggregated (Item + Channel)**
- Forecast per item-channel, aggregated across all warehouses
- Example: SKU001_Online, SKU001_Retail  
- Use when: Module 2 will allocate to warehouses

**Level 3: Channel Aggregated (Item + Warehouse)**
- Forecast per item-warehouse, aggregated across all channels
- Example: SKU001_WH_A, SKU001_WH_B
- Use when: Module 2 will allocate to channels

**Level 4: Most Aggregated (Item Only)**
- Single forecast per item, aggregated across all warehouses and channels
- Example: SKU001
- Use when: Module 2 will handle all allocation

---

## 🎯 Key Features

### Core Capabilities
- **TimeGPT Forecasting** - Foundation model with zero-shot learning
- **Tiered Approach** - Automatic method selection based on data:
  - 1-3 points → Naive/Moving Average
  - 4-7 points → TimeGPT (no intervals)
  - 8+ points → TimeGPT (with 80% & 95% confidence intervals)
- **Edge Case Handling** - Graceful handling of limited data, zeros, outliers
- **Promotional Modeling** - Optional promo feature support
- **Professional Export** - Multi-sheet Excel ready for Module 2

### User Interface
- **Tab 1: Data Upload** - Upload, preview, validate
- **Tab 2: Configuration** - Set frequency, horizon, granularity
- **Tab 3: Results** - View, analyze, and export forecasts

---

## 📋 Requirements

### System Requirements
- Python 3.8+ (including 3.13 ✅)
- 2GB RAM minimum
- Internet connection for TimeGPT API

### API Keys
- Nixtla API key (for TimeGPT) - Get free trial at [nixtla.io](https://nixtla.io)

---

## 🛠️ Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.streamlit/secrets.toml`:
```toml
NIXTLA_API_KEY = "your-api-key-here"
```

### 3. Run Application
```bash
streamlit run app.py
```

---

## 📊 Data Requirements

### Historical Data File (Required)

**Required Columns:**
- `ds`: Date (must match frequency)
- `item_name`: Product identifier
- `wh`: Warehouse code
- `channel`: Sales channel
- `y`: Sales quantity

**Optional Columns:**
- `category`, `class`, `extention`: Product attributes
- `promo_flag`: Promotional indicator (0 or 1)

**Example:**
```csv
ds,item_name,wh,channel,y,category,promo_flag
2023-01-02,SKU001,WH_A,Online,150,Electronics,0
2023-01-02,SKU001,WH_B,Retail,200,Electronics,0
2023-01-09,SKU001,WH_A,Online,180,Electronics,1
```

### Future Promo Calendar (Optional)

Same columns as historical, but only `promo_flag` values are used.

---

## 🎯 Usage Guide

### Step 1: Upload Data
1. Upload CSV or Excel file
2. Review data preview
3. Check validation results

### Step 2: Configure Forecast
1. **Select Time Bucket**: Weekly or Monthly
2. **Set Horizon**: Number of periods to forecast
3. **Choose Granularity**:
   - Warehouse: By Origin OR All Combined
   - Channel: By Channel OR All Combined
4. **Enable/Disable Options**:
   - Outlier handling
   - Promotional features
5. **Choose Naive Method**: For SKUs with 1-3 data points

### Step 3: Run & Export
1. Review configuration
2. Click "Run Forecast"
3. View results in 3 sub-tabs:
   - Summary Dashboard
   - SKU Detail View
   - Export Results
4. Download Excel file

---

## 📤 Export Format

### Excel File Structure:

**Sheet 1: TimeGPT_Forecast**
- Main forecast output
- Columns adapt to your granularity selection:
  - `date`, `item_name`, `[wh]`, `[channel]`, `forecast`
  - Optional: `category`, `class`, `extention`
  
**Sheet 2: Forecast_Metadata**
- Method used per SKU
- History length
- Forecast horizon

**Sheet 3: Skipped_Items** (if any)
- SKUs that couldn't be forecasted
- Reasons for skipping

---

## 🔄 Granularity Examples

### Example Input Data:
```csv
ds,item_name,wh,channel,y
2023-01-02,SKU001,WH_A,Online,100
2023-01-02,SKU001,WH_A,Retail,150
2023-01-02,SKU001,WH_B,Online,120
2023-01-02,SKU001,WH_B,Retail,180
Total: 550
```

### Output by Granularity:

**By Warehouse + By Channel** (Level 1):
```csv
date,item_name,wh,channel,forecast
2024-01-01,SKU001,WH_A,Online,105
2024-01-01,SKU001,WH_A,Retail,158
2024-01-01,SKU001,WH_B,Online,126
2024-01-01,SKU001,WH_B,Retail,189
```

**All Warehouses + By Channel** (Level 2):
```csv
date,item_name,channel,forecast
2024-01-01,SKU001,Online,231
2024-01-01,SKU001,Retail,347
```

**By Warehouse + All Channels** (Level 3):
```csv
date,item_name,wh,forecast
2024-01-01,SKU001,WH_A,263
2024-01-01,SKU001,WH_B,315
```

**All Warehouses + All Channels** (Level 4):
```csv
date,item_name,forecast
2024-01-01,SKU001,578
```

---

## 🎓 Choosing the Right Granularity

### When to Use Each Level:

**Most Granular (Item + WH + Channel)**
- ✅ You have historical data at this level
- ✅ Each location-channel has distinct patterns
- ✅ No allocation needed in Module 2
- ❌ More SKUs = longer forecast time
- ❌ Limited data per combination

**Warehouse Aggregated (Item + Channel)**
- ✅ Warehouse allocation handled by Module 2
- ✅ More data per forecast = better accuracy
- ✅ Consistent channel behavior across locations
- Example: Online sales similar across all warehouses

**Channel Aggregated (Item + Warehouse)**
- ✅ Channel allocation handled by Module 2
- ✅ Location-specific patterns important
- ✅ Channels have similar patterns
- Example: Different demand per warehouse region

**Most Aggregated (Item Only)**
- ✅ Maximum data per forecast
- ✅ Both warehouse & channel allocated in Module 2
- ✅ Fastest forecasting
- ✅ Best for items with limited history
- Example: New products, slow movers

**Rule of Thumb**: Use the highest aggregation level that makes business sense!

---

## 🔧 Troubleshooting

### Common Issues:

**"Missing required columns"**
- Ensure your file has: `ds`, `item_name`, `wh`, `channel`, `y`

**"TimeGPT API error"**
- Check your API key in `.streamlit/secrets.toml`
- Verify internet connection
- Check API quota at nixtla.io

**"Cold start items skipped"**
- Normal behavior for items with 0 history
- Check the Skipped_Items sheet in export

**Slow performance**
- Expected for 1000+ SKUs (3-5 minutes)
- Consider using higher granularity aggregation
- Progress bar shows real-time status

---

## 📊 Performance

- **Small datasets** (<100 SKUs): ~30 seconds
- **Medium datasets** (100-500 SKUs): 1-2 minutes
- **Large datasets** (500-1000 SKUs): 3-5 minutes
- **Very large** (1000+ SKUs): 5-10 minutes

**Tip**: Higher granularity aggregation = fewer SKUs = faster forecasts

---

## 🆚 Differences from V1.0

| Feature | V1.0 | V2.0 (This Version) |
|---------|------|---------------------|
| Models | TimeGPT + ML + Ensemble | TimeGPT Only |
| Backtesting | ✅ Yes | ❌ No |
| Accuracy Metrics | ✅ MAPE, RMSE, MAE | ❌ None |
| Dependencies | Many (Numba, MLForecast) | Minimal |
| Python 3.13 | ❌ Not compatible | ✅ Compatible |
| Granularity Options | ❌ No | ✅ Yes (4 levels) |
| Tabs | 4 | 3 |
| Speed | Slower (ML training) | Faster |
| Complexity | High | Low |
| Best For | Advanced users, accuracy focus | Speed, simplicity, flexibility |

---

## 🚀 Module 2 Integration

The exported forecast is ready for Module 2 (allocation):

### Workflow:

**Level 1 (Most Granular)**:
```
Module 1 Output → Module 2 → Only weekly breakdown (if monthly)
```

**Level 2-4 (Aggregated)**:
```
Module 1 Output → Module 2 → Allocate by warehouse/channel → Weekly breakdown
```

### Export Format:
- Uses clean column names: `date`, `item_name`, `wh`, `channel`, `forecast`
- No technical columns (no `unique_id`, `method`, etc.) in main forecast sheet
- Ready to import directly into allocation engine

---

## 💡 Best Practices

1. **Start Simple**: Begin with highest aggregation, then drill down if needed
2. **Match Your Data**: Choose granularity that matches your historical data quality
3. **Consider Module 2**: If you have allocation logic in Module 2, use aggregated forecasts
4. **Test Both Frequencies**: Try both weekly and monthly to see which works better
5. **Review Skipped Items**: Check why items were skipped and fix data if possible
6. **Use Confidence Intervals**: Available for SKUs with 8+ history points

---

## 📞 Support

For issues:
1. Check this README
2. Review error messages (they're detailed!)
3. Check the Data Quality Report in Tab 1
4. Review Skipped_Items sheet in export

---

## 📄 License

[Your License]

---

## 🙏 Acknowledgments

- Powered by [Nixtla TimeGPT](https://nixtla.io)
- Built with [Streamlit](https://streamlit.io)
- Visualizations by [Plotly](https://plotly.com)

---

**Version**: 2.0 TimeGPT Edition  
**Last Updated**: December 2024  
**Status**: Production Ready ✅
