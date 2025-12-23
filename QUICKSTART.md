# 🚀 Quick Start Guide - V2.0

## Get Started in 3 Minutes!

### Step 1: Install (1 minute)
```bash
pip install -r requirements.txt
```

**Note**: Works with Python 3.8+ including Python 3.13! ✅

### Step 2: Add API Key (30 seconds)
Create `.streamlit/secrets.toml`:
```toml
NIXTLA_API_KEY = "your-key-here"
```

Get free API key at [nixtla.io](https://nixtla.io)

### Step 3: Run (30 seconds)
```bash
streamlit run app.py
```

### Step 4: Use (1 minute)
1. **Upload** your data (CSV/Excel)
2. **Configure** frequency + granularity
3. **Run** forecast
4. **Download** results

---

## 📊 Data Format

Your file needs:
```csv
ds,item_name,wh,channel,y
2023-01-02,SKU001,WH_A,Online,150
2023-01-09,SKU001,WH_A,Online,180
```

---

## 🎯 Granularity Quick Guide

### Choose Your Level:

**Option 1: Most Detailed**
- Warehouse: By Origin
- Channel: By Channel
- Result: Forecast per warehouse per channel
- Use: When you need full detail

**Option 2: Aggregate Warehouses**
- Warehouse: All Combined
- Channel: By Channel
- Result: Forecast per channel (all warehouses combined)
- Use: When Module 2 allocates warehouses

**Option 3: Aggregate Channels**
- Warehouse: By Origin
- Channel: All Combined
- Result: Forecast per warehouse (all channels combined)
- Use: When Module 2 allocates channels

**Option 4: Maximum Aggregation**
- Warehouse: All Combined
- Channel: All Combined
- Result: One forecast per item (total)
- Use: When Module 2 handles everything

---

## ⚡ Quick Test

Use the dummy data files provided:
1. Upload `historical_sales_weekly.xlsx`
2. Config:
   - Time Bucket: **Weekly**
   - Horizon: **26 weeks**
   - Warehouse: **By Origin** (to see all detail)
   - Channel: **By Channel**
3. Run Forecast
4. See results in ~2 minutes!

---

## 🎯 What You Get

✅ Professional forecasts powered by TimeGPT  
✅ Confidence intervals (80% & 95%)  
✅ Multi-sheet Excel export  
✅ Summary dashboard  
✅ Individual SKU details  
✅ Ready for Module 2  

---

## 💡 Pro Tips

1. **Start Aggregated**: Use higher aggregation first, then drill down if needed
2. **Check Quality**: Review the Data Quality Report
3. **Review Skipped**: Check why items were skipped
4. **Test Both**: Try weekly vs monthly to see which works better
5. **Use Confidence Intervals**: Helpful for risk assessment

---

## 🔧 Troubleshooting

**API Error?**
- Check your API key
- Verify internet connection
- Ensure you have API quota

**Slow?**
- Normal for 1000+ SKUs (3-5 min)
- Try higher aggregation for speed

**Items Skipped?**
- Check Skipped_Items sheet for reasons
- Usually: 0 points, all zeros, or no variance

---

## 🆚 V2.0 vs V1.0

**V2.0 Advantages**:
- ✅ Python 3.13 compatible
- ✅ Faster (no ML training)
- ✅ Simpler (fewer dependencies)
- ✅ Granularity options (NEW!)
- ✅ Easier to maintain

**V1.0 Advantages**:
- ✅ Model comparison
- ✅ Accuracy metrics (MAPE, RMSE)
- ✅ Backtesting

**Choose V2.0 if**: You want speed, simplicity, and granularity flexibility  
**Choose V1.0 if**: You need accuracy validation and model comparison

---

## 📚 Next Steps

1. Read full [README.md](README.md) for detailed docs
2. Try different granularity levels
3. Export and review results
4. Integrate with Module 2

---

**Ready to forecast!** 🎉
