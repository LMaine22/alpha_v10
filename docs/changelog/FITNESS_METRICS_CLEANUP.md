# Fitness Metrics Cleanup - COMPLETED ✅

## 🎯 **Goal**
Remove all old fitness metrics (Sortino, Expectancy, Support) and show ONLY the new robust fitness metrics (RET_over_CVaR5, MartinRatio, GPR) in all outputs.

## ✅ **Changes Made**

### **1. GA Generation Display Updates**

#### **`alpha_discovery/search/nsga.py`**
- **Before**: `Best Sortino: 100.000 | Best Expectancy: 16884.6`
- **After**: `Best RET/CVaR5: 1.42 | Best Martin: 1.10 | Best GPR: 2.30`

#### **`alpha_discovery/search/island_model.py`**
- **Before**: `Best Sortino: 100.000 | Best Expectancy: 16884.6`
- **After**: `Best RET/CVaR5: 1.42 | Best Martin: 1.10 | Best GPR: 2.30`

### **2. Artifacts & Summary Cleanup**

#### **`alpha_discovery/reporting/robust_metrics_helper.py`**
- **Updated `get_robust_metrics_column_order()`**: Now only includes fitness metrics
- **Updated `add_robust_metrics_to_summary()`**: Only adds RET_over_CVaR5, MartinRatio, GPR
- **Removed**: All old metrics (sortino_lb, expectancy, support, sharpe, omega_ratio, etc.)

### **3. GA Core Updates**

#### **`alpha_discovery/search/ga_core.py`**
- **Removed legacy fallback**: No more fallback to old Sortino/Expectancy/Support
- **Only fitness metrics**: Only stores RET_over_CVaR5, MartinRatio, GPR in perf dict
- **Clean objectives**: Always uses new robust metrics for objectives

### **4. One-Liner Reporting**

#### **`alpha_discovery/reporting/one_liner.py`**
- **Before**: `RET/ES=1.42 | Martin=1.10 | GPR=2.30 | WR_lb=0.57 | DD=-18.0% | UI=4.9 | N=23`
- **After**: `RET/ES=1.42 | Martin=1.10 | GPR=2.30`

## 🎯 **What You'll See Now**

### **Generation Summaries:**
```
Gen 1/20 | Best RET/CVaR5: 1.42 | Best Martin: 1.10 | Best GPR: 2.30 | Pop: 200
```

### **Setup One-Liners:**
```
SETUP_0001: RET/ES=1.42 | Martin=1.10 | GPR=2.30
```

### **Summary Files:**
- **Columns**: Only setup_id, ticker, direction, description, signal_ids, trades_count, pnl_dollars, nav info, and the 3 fitness metrics
- **No more**: sortino_lb, expectancy, support, sharpe, omega_ratio, etc.

## 🚀 **Benefits**

1. **Cleaner Output**: Only shows the metrics that matter for fitness evaluation
2. **Consistent**: All outputs use the same 3 fitness metrics
3. **Focused**: No confusion with old metrics that aren't being optimized
4. **Efficient**: Smaller file sizes and faster processing

## 📊 **Fitness Metrics Used**

- **`RET_over_CVaR5`**: Return over Expected Shortfall at 5%
- **`MartinRatio`**: Annualized return / Ulcer Index  
- **`GPR`**: Gain-to-Pain Ratio

These are the 3 objectives your GA is optimizing for with the "balanced" profile! 🎯

## ✅ **Status**
All changes completed and tested. The system now shows ONLY the new robust fitness metrics in all outputs.
