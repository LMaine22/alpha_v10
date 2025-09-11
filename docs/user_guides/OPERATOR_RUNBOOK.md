# Operator Runbook - Robust Metrics System

## 🎯 **System Overview**

The robust metrics system replaces the problematic Sortino-based fitness with more reliable, options-friendly risk metrics that don't break with very profitable strategies.

## 📊 **Fitness Profiles**

### **Available Profiles:**
- **`safe`**: `["RET_over_CVaR5", "MartinRatio"]` - Conservative, drawdown-focused
- **`balanced`** (default): `["RET_over_CVaR5", "MartinRatio", "GPR"]` - Balanced risk/return
- **`aggressive`**: `["RET_over_CVaR5", "GPR", "RobustSharpe_MAD"]` - Higher risk tolerance
- **`quality`**: `["MartinRatio", "GPR", "WinRate_Wilson_LB"]` - Quality-focused
- **`legacy`**: `["sortino_lb", "expectancy", "support"]` - Fallback to old system

### **Configuration:**
```python
# Use preset profile
settings.ga.fitness_profile = "balanced"

# Or use custom objectives
settings.ga.objectives = ["RET_over_CVaR5", "MartinRatio", "GPR"]
```

## 🔍 **Sanity Checks**

### **1. Metric Finiteness**
- All metrics should be finite (no NaN/inf)
- GPR clamped at 10.0 to prevent infinite domination
- Fallbacks applied for invalid values

### **2. Expected Behavior**
- **More wins** → Higher GPR, WinRate_Wilson_LB
- **Less drawdown** → Higher MartinRatio, Calmar
- **Better risk-adjusted returns** → Higher RET_over_CVaR5, RobustSharpe_MAD

### **3. Stability Checks**
- Medians should be stable across folds
- No wild swings in metric values
- Evidence that gates are active (filtering working)

## 🚪 **Gauntlet Gates (Recommended)**

### **Lower Bounds:**
- `TradesCount_min = 12` - Minimum trade count
- `WinRate_Wilson_LB_min = 0.53` - Credible win rate
- `RET_over_CVaR5_min = 1.0` - Risk-adjusted return
- `MartinRatio_min = 0.8` - Drawdown-adjusted return

### **Upper Bounds:**
- `DD_max >= -0.25` - Max 25% drawdown
- `UlcerIndex <= 6.0` - Pain threshold
- `N_eff >= 10` - Effective sample size

### **Implementation:**
```python
# In gauntlet filtering
gates = {
    "TradesCount_min": 12,
    "WinRate_Wilson_LB_min": 0.53,
    "RET_over_CVaR5_min": 1.0,
    "DD_max_min": -0.25,  # Note: negative value
    "UlcerIndex_max": 6.0,
    "MartinRatio_min": 0.8,
}
```

## 🛡️ **Edge Case Handling**

### **Clamps & Fallbacks:**
- **GPR**: Clamped at 10.0 (prevents infinite domination)
- **NaN/inf values**: Fallback to 0.0
- **Empty ledgers**: All metrics return 0.0
- **Single trade**: Graceful degradation

### **Robust Calculations:**
- **CVaR fallback**: If 5% tail too thin, use 10% tail
- **Wilson bounds**: Conservative win rate estimates
- **MAD Sharpe**: No winsorization issues

## 🧪 **Experiments & Monitoring**

### **Profile Bake-offs:**
1. Run same data with different profiles
2. Compare Pareto fronts
3. Analyze metric correlations (ρ < 0.75 preferred)
4. Check dominance frequency

### **Stability Testing:**
- 100× Moving Block Bootstrap for confidence intervals
- Rolling 60-day `RET/ES` drift monitoring
- Regime-aware Sharpe consistency checks

### **Kill Switches:**
- `RET_over_CVaR5_lb < 0.8` twice in a row
- `Martin_lb < 0.5` twice in a row
- Tie-break by `WinRate_Wilson_LB`

## 📈 **One-Liner Reporting**

### **Format:**
```
SETUP_0001: RET/ES=1.42 | Martin=1.10 | GPR=2.30 | WR_lb=0.57 | DD=-18.0% | UI=4.9 | N=23
```

### **Enable/Disable:**
```python
# In config
settings.reporting.one_liners = True  # Enable compact summaries
```

### **Interpretation:**
- **RET/ES**: Return over Expected Shortfall (higher = better)
- **Martin**: Martin Ratio (higher = better)
- **GPR**: Gain-to-Pain Ratio (higher = better)
- **WR_lb**: Wilson lower-bound win rate (higher = better)
- **DD**: Max drawdown % (less negative = better)
- **UI**: Ulcer Index (lower = better)
- **N**: Trade count

## 🚀 **First Run Recommendations**

### **Settings:**
- **Profile**: `balanced`
- **Population**: 128-256 individuals
- **Generations**: 60-100
- **Epsilon dominance**: ~1e-4
- **Gates**: All recommended gates active

### **Parallel Testing:**
- Keep legacy run in parallel for 1-2 days
- Compare results and validate improvements
- Monitor for any unexpected behavior

## 🔧 **Troubleshooting**

### **Common Issues:**

1. **All metrics = 0.0**
   - Check ledger has required columns (`pnl_dollars`, `capital_allocated`)
   - Verify data is not empty

2. **GPR = 10.0 consistently**
   - Normal for very profitable strategies
   - Clamp prevents infinite domination

3. **One-liners not printing**
   - Check `settings.reporting.one_liners = True`
   - Verify no import errors

4. **Metrics seem wrong**
   - Check clamps and fallbacks are working
   - Verify input data quality

### **Debug Commands:**
```bash
# Test one-liner system
python scripts/one_liner_smoke.py

# Test robust metrics
python test_robust_metrics.py

# Check config
python -c "from alpha_discovery.config import settings; print(settings.ga.fitness_profile)"
```

## 📋 **Quick Reference**

### **Key Files:**
- `alpha_discovery/eval/robust_metrics.py` - Core metrics
- `alpha_discovery/ga/fitness_loader.py` - Fitness system
- `alpha_discovery/reporting/one_liner.py` - Compact reporting
- `scripts/one_liner_smoke.py` - Test script

### **Key Config:**
- `settings.ga.fitness_profile` - Active profile
- `settings.ga.objectives` - Custom objectives
- `settings.reporting.one_liners` - Enable summaries

### **Key Metrics:**
- `RET_over_CVaR5` - Risk-adjusted return
- `MartinRatio` - Drawdown-adjusted return
- `GPR` - Gain-to-pain ratio
- `WinRate_Wilson_LB` - Conservative win rate

This system should provide much more robust and meaningful fitness evaluation for your options trading strategies! 🎯
