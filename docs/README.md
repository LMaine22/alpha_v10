# Alpha Discovery v10 - Documentation

Welcome to the Alpha Discovery v10 documentation. This directory contains comprehensive documentation for the project.

## 📁 Documentation Structure

### User Guides
- **[Operator Runbook](user_guides/OPERATOR_RUNBOOK.md)** - Main operational guide for running experiments

### Technical Documentation
- **[8-Feature IV System](technical/COMPLETE_8_FEATURE_IV_SYSTEM.md)** - Complete IV pricing system documentation
- **[Robust Metrics Upgrade](technical/ROBUST_METRICS_UPGRADE.md)** - Metrics system improvements
- **[Strict OOS Gauntlet](technical/STRICT_OOS_GAUNTLET_EXPLAINED.md)** - Out-of-sample validation system
- **[Strict OOS Files](technical/STRICT_OOS_FILES_EXPLAINED.md)** - File structure and outputs
- **[Current Fitness Objectives](technical/CURRENT_FITNESS_OBJECTIVES.md)** - GA fitness configuration

### Changelog
- **[Fitness Metrics Cleanup](changelog/FITNESS_METRICS_CLEANUP.md)** - Metrics system cleanup notes
- **[Fitness Metrics Fixed](changelog/FITNESS_METRICS_FIXED.md)** - Metrics fixes and improvements
- **[Strict OOS Fixed](changelog/STRICT_OOS_FIXED.md)** - OOS system fixes
- **[Strict OOS Simplified](changelog/STRICT_OOS_SIMPLIFIED.md)** - OOS system simplification

## 🚀 Quick Start

1. See the [Operator Runbook](user_guides/OPERATOR_RUNBOOK.md) for getting started
2. Check [8-Feature IV System](technical/COMPLETE_8_FEATURE_IV_SYSTEM.md) for pricing details
3. Review [Strict OOS Gauntlet](technical/STRICT_OOS_GAUNTLET_EXPLAINED.md) for validation

## 📊 Project Overview

Alpha Discovery v10 is a comprehensive options trading strategy discovery system that uses:
- Genetic Algorithms (GA) with NSGA-II multi-objective optimization
- Walk-forward validation with strict out-of-sample testing
- Regime-aware exit strategies
- Economic event feature integration
- Advanced options pricing with IV smile interpolation

## 🔧 System Components

- **Data Pipeline**: Bloomberg data processing and economic event integration
- **Feature Engineering**: Technical indicators and event-based features
- **Signal Generation**: Primitive signal compilation and combination
- **Genetic Search**: Multi-objective optimization for strategy discovery
- **Backtesting**: Options simulation with regime-aware exits
- **Validation**: Multi-stage gauntlet with strict OOS testing
- **Reporting**: Comprehensive results analysis and visualization
