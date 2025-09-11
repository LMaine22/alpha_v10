# Strict OOS Gauntlet Files Explained

## 🎯 **Overview**

The Strict OOS Gauntlet is a 3-stage filtering system that validates strategies using **only out-of-sample data** (test periods that were never seen during training). This ensures true out-of-sample validation without look-ahead bias.

## 📁 **File Structure & Purpose**

### **Core Files:**

#### **1. `strict_oos_stitched_ledger.csv`**
- **What**: Raw stitched ledger of ALL OOS trades across all folds
- **Purpose**: Complete audit trail of every trade that happened in test periods
- **Columns**: Standard trade ledger columns (setup_id, ticker, pnl_dollars, dates, etc.)
- **When**: Created first, before any filtering

#### **2. `all_setups_rollup_full_oos.csv`**
- **What**: Summary of ALL setups before any gauntlet filtering
- **Purpose**: Shows performance of every setup that had OOS trades
- **Key Columns**: setup_id, ticker, trades_count, pnl_dollars_sum, description
- **When**: Created before Stage 1 filtering

### **Stage 1 Files (Health Check):**

#### **3. `stage1_oos.csv`**
- **What**: Ledger after Stage 1 filtering (recency + minimum trades)
- **Purpose**: Only setups that passed recency and activity checks
- **Filter**: Must have traded recently + minimum trade count
- **When**: After Stage 1, before Stage 2

#### **4. `stage1_summary.csv`**
- **What**: Summary of Stage 1 results
- **Purpose**: Shows which setups passed/failed Stage 1 and why
- **Key Columns**: setup_id, pass_recency, pass_min_trades, pass_stage1
- **When**: After Stage 1 filtering

#### **5. `stage1_oos_compat_summary.csv`**
- **What**: OOS-compatible Stage 1 results (if legacy Stage 1 failed)
- **Purpose**: Fallback when legacy Stage 1 doesn't work with OOS data
- **When**: Only if legacy Stage 1 returns no filtering

### **Stage 2 Files (Profitability):**

#### **6. `stage2_oos.csv`**
- **What**: Ledger after Stage 2 filtering (profitability checks)
- **Purpose**: Only profitable setups that passed Stage 1
- **Filter**: Positive NAV, positive PnL, reasonable drawdowns
- **When**: After Stage 2, before Stage 3

#### **7. `stage2_summary.csv`**
- **What**: Summary of Stage 2 results
- **Purpose**: Shows which setups passed/failed Stage 2 and why
- **Key Columns**: setup_id, pass_stage2, reasons
- **When**: After Stage 2 filtering

### **Stage 3 Files (Robustness):**

#### **8. `stage3_oos.csv`**
- **What**: Ledger after Stage 3 filtering (statistical robustness)
- **Purpose**: Only statistically robust setups that passed Stages 1 & 2
- **Filter**: DSR, confidence intervals, stability ratios
- **When**: After Stage 3, final survivors

#### **9. `stage3_summary.csv`**
- **What**: Summary of Stage 3 results
- **Purpose**: Shows which setups passed/failed Stage 3 and why
- **Key Columns**: setup_id, pass_stage3, reasons
- **When**: After Stage 3 filtering

### **Final Results:**

#### **10. `gauntlet_summary.csv`**
- **What**: Final survivors that passed all 3 stages
- **Purpose**: Clean list of deployable setups
- **Key Columns**: setup_fp, ticker, direction, trades_count
- **When**: After all filtering complete

#### **11. `gauntlet_ledger.csv`**
- **What**: Final ledger of surviving setups only
- **Purpose**: Complete trade history of deployable setups
- **When**: After all filtering complete

#### **12. `all_setups_summary.csv`**
- **What**: Summary of all setups (survivors + non-survivors)
- **Purpose**: Complete view of all setups with their final status
- **Key Columns**: setup_id, ticker, trades_count, pnl_dollars_sum, description
- **When**: After all filtering complete

## 🔄 **Data Flow**

```
Raw OOS Data
    ↓
strict_oos_stitched_ledger.csv (ALL trades)
    ↓
all_setups_rollup_full_oos.csv (ALL setups summary)
    ↓
Stage 1: Health Check
    ↓
stage1_oos.csv + stage1_summary.csv
    ↓
Stage 2: Profitability Check
    ↓
stage2_oos.csv + stage2_summary.csv
    ↓
Stage 3: Robustness Check
    ↓
stage3_oos.csv + stage3_summary.csv
    ↓
Final Results
    ↓
gauntlet_summary.csv + gauntlet_ledger.csv + all_setups_summary.csv
```

## 🎯 **Key Differences Between Files**

### **By Stage:**
- **Pre-filtering**: `strict_oos_stitched_ledger.csv`, `all_setups_rollup_full_oos.csv`
- **Stage 1**: `stage1_oos.csv`, `stage1_summary.csv`, `stage1_oos_compat_summary.csv`
- **Stage 2**: `stage2_oos.csv`, `stage2_summary.csv`
- **Stage 3**: `stage3_oos.csv`, `stage3_summary.csv`
- **Final**: `gauntlet_summary.csv`, `gauntlet_ledger.csv`, `all_setups_summary.csv`

### **By Content Type:**
- **Ledgers**: Trade-level data (individual trades)
- **Summaries**: Setup-level data (aggregated per setup)

### **By Scope:**
- **All setups**: Every setup that had OOS trades
- **Survivors only**: Only setups that passed all stages
- **Per-stage**: Only setups that passed up to that stage

## 🔍 **How to Use These Files**

### **For Analysis:**
1. **Start with** `all_setups_summary.csv` - see all setups
2. **Check** `gauntlet_summary.csv` - see final survivors
3. **Drill down** to `gauntlet_ledger.csv` - see individual trades

### **For Debugging:**
1. **Check** `stage1_summary.csv` - see recency/activity issues
2. **Check** `stage2_summary.csv` - see profitability issues
3. **Check** `stage3_summary.csv` - see robustness issues

### **For Monitoring:**
1. **Track** `gauntlet_summary.csv` - count of survivors
2. **Monitor** `all_setups_summary.csv` - overall performance
3. **Audit** `strict_oos_stitched_ledger.csv` - complete trade history

## 📊 **With Robust Metrics**

Now all these files will include the new robust metrics columns:
- `RET_over_CVaR5`, `MartinRatio`, `GPR`
- `RobustSharpe_MAD`, `WinRate_Wilson_LB`, `Expectancy_LB`
- `Calmar`, `UlcerIndex`, `PainIndex`
- `DD_max`, `DD_avg_duration`, `DD_max_duration`

This gives you much better visibility into the risk-adjusted performance of each setup at every stage of the gauntlet! 🎯
