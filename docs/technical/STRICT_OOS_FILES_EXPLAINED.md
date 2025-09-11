# Strict OOS Files Explained - Why So Many? 🤔

## 🎯 **The Problem You're Seeing**
You have **11 files** in `strict_oos/` which seems excessive, but each serves a specific purpose in the gauntlet pipeline.

## 📁 **File Breakdown - What Each One Does**

### **Core Pipeline Files (4 files)**
1. **`strict_oos_stitched_ledger.csv`** - Raw OOS data from all folds stitched together
2. **`stage1_oos.csv`** - After Stage 1 filtering (health checks, recency)
3. **`stage2_oos.csv`** - After Stage 2 filtering (profitability)  
4. **`stage3_oos.csv`** - After Stage 3 filtering (robustness)

### **Summary Files (4 files)**
5. **`stage1_summary.csv`** - Stage 1 pass/fail results per setup
6. **`stage2_summary.csv`** - Stage 2 profitability metrics per setup
7. **`stage3_summary.csv`** - Stage 3 robustness metrics per setup
8. **`gauntlet_summary.csv`** - Final survivors summary

### **Rollup Files (2 files)**
9. **`all_setups_rollup_full_oos.csv`** - ALL setups before any filtering
10. **`all_setups_summary.csv`** - Final survivors with metrics

### **Special Files (1 file)**
11. **`stage1_compat_note.txt`** - Explains why OOS-compat Stage 1 was used

## 🔄 **Why So Many Files?**

### **The Gauntlet Pipeline is a 3-Stage Filter:**
```
Raw OOS Data → Stage 1 → Stage 2 → Stage 3 → Final Survivors
     ↓           ↓         ↓         ↓           ↓
  stitched    stage1_   stage2_   stage3_   gauntlet_
  ledger      oos.csv   oos.csv   oos.csv   ledger.csv
```

### **Each Stage Creates 2 Files:**
- **`stageX_oos.csv`** - The actual trade data after filtering
- **`stageX_summary.csv`** - The pass/fail results and metrics

## 🎯 **What You Actually Need**

### **For Analysis:**
- **`gauntlet_summary.csv`** - Final survivors with all metrics
- **`all_setups_summary.csv`** - Same as above (duplicate)

### **For Debugging:**
- **`stage1_compat_note.txt`** - Why OOS-compat was used
- **`strict_oos_stitched_ledger.csv`** - Raw data before filtering

### **You Can Ignore:**
- All the intermediate `stage1_oos.csv`, `stage2_oos.csv`, `stage3_oos.csv`
- All the `stageX_summary.csv` files
- The rollup files

## 🚀 **Simplification Suggestion**

The system could be simplified to only create:
1. **`strict_oos_stitched_ledger.csv`** - Raw data
2. **`gauntlet_summary.csv`** - Final results
3. **`stage1_compat_note.txt`** - Debug info

**Total: 3 files instead of 11!**

## 🔍 **Why This Happened**

The strict OOS system was designed to be **auditable** - you can see exactly what happened at each stage. But for most users, this creates unnecessary complexity.

**The good news:** You only need to look at `gauntlet_summary.csv` for your analysis! 🎯
