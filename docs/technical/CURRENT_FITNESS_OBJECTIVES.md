# Your Current Fitness Objectives

## 🎯 **Active Configuration**

Based on your current config settings:

```python
# In alpha_discovery/config.py
settings.ga.fitness_profile = "balanced"  # Default
settings.ga.objectives = None  # Using profile, not custom
```

## 📊 **Your Current Fitness Objectives**

Since you're using the **"balanced"** profile, your GA is optimizing for these 3 objectives:

### **1. RET_over_CVaR5** (Return over Expected Shortfall)
- **What**: Total return divided by Expected Shortfall at 5%
- **Why**: Pain-aware, tail-robust risk metric
- **Higher = Better**: More return per unit of tail risk
- **Range**: 0.0 to ∞ (clamped at reasonable values)

### **2. MartinRatio** (Martin Ratio)
- **What**: Annualized return divided by Ulcer Index
- **Why**: Drawdown-shape aware risk metric
- **Higher = Better**: More return per unit of drawdown pain
- **Range**: 0.0 to ∞ (clamped at reasonable values)

### **3. GPR** (Gain-to-Pain Ratio)
- **What**: Sum of wins divided by absolute sum of losses
- **Why**: Simple, options-friendly risk metric
- **Higher = Better**: More gains relative to losses
- **Range**: 0.0 to 10.0 (clamped to prevent infinite domination)

## 🔄 **How NSGA-II Uses These**

Your GA will:
1. **Maximize all 3 objectives simultaneously**
2. **Find Pareto-optimal solutions** (trade-offs between objectives)
3. **Use crowding distance** for diversity in the Pareto front
4. **Apply clamps and fallbacks** to ensure numerical stability

## 🎛️ **Alternative Profiles Available**

If you want to change your fitness objectives, you can switch profiles:

### **Safe Profile** (Conservative):
```python
settings.ga.fitness_profile = "safe"
# Objectives: ["RET_over_CVaR5", "MartinRatio"]
```

### **Aggressive Profile** (Higher Risk):
```python
settings.ga.fitness_profile = "aggressive"
# Objectives: ["RET_over_CVaR5", "GPR", "RobustSharpe_MAD"]
```

### **Quality Profile** (Quality-Focused):
```python
settings.ga.fitness_profile = "quality"
# Objectives: ["MartinRatio", "GPR", "WinRate_Wilson_LB"]
```

### **Custom Objectives**:
```python
settings.ga.objectives = ["RET_over_CVaR5", "GPR", "Calmar"]
# This overrides the profile setting
```

## 📈 **What This Means for Your Evolution**

With the **"balanced"** profile, your GA will favor setups that:
- ✅ **Have good risk-adjusted returns** (RET_over_CVaR5)
- ✅ **Manage drawdowns well** (MartinRatio)
- ✅ **Have favorable win/loss ratios** (GPR)

This should give you a good balance of return, risk management, and consistency - perfect for options trading strategies! 🎯

## 🔍 **Monitoring Your Evolution**

During GA evolution, you'll see one-liner summaries like:
```
SETUP_0001: RET/ES=1.42 | Martin=1.10 | GPR=2.30 | WR_lb=0.57 | DD=-18.0% | UI=4.9 | N=23
```

The first three values (RET/ES, Martin, GPR) are your **fitness objectives** that the GA is optimizing for! 🚀
