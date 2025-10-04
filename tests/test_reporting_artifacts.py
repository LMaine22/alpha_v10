import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_discovery.eval import metrics
from alpha_discovery.reporting.artifacts import save_results
from alpha_discovery.config import settings


def test_metrics_non_finite_handling():
    daily_returns = pd.Series([0.02, 0.015], index=pd.date_range("2020-01-01", periods=2, freq="B"))
    ledger = pd.DataFrame({
        "trigger_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "exit_date": pd.to_datetime(["2020-01-05", "2020-01-06"]),
        "pnl_dollars": [200.0, 150.0],
        "pnl_pct": [0.2, 0.15],
    })

    result = metrics.compute_portfolio_metrics_bundle(daily_returns, ledger)

    assert result.get("profit_factor") is None
    assert result.get("profit_factor_raw") in (None, np.inf)
    lb = result.get("bootstrap_profit_factor_lb")
    assert lb is None or np.isfinite(lb)


def test_save_results_produces_new_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    splits = {
        "splits": [
            {
                "split_id": "OUTER_001",
                "train_start": "2020-01-01",
                "train_end": "2020-12-31",
                "test_start": "2021-01-01",
                "test_end": "2021-02-01",
            }
        ]
    }

    ledger = pd.DataFrame({
        "trigger_date": pd.to_datetime(["2021-01-05"]),
        "exit_date": pd.to_datetime(["2021-01-12"]),
        "fold_id": ["OUTER_001_inner_00"],
        "outer_id": ["OUTER_001"],
        "ticker": ["AAPL US Equity"],
        "horizon_days": [5],
        "direction": ["long"],
        "exit_policy_id": ["policy"],
        "exit_reason": ["target"],
        "strike": [150.0],
        "entry_underlying": [150.0],
        "exit_underlying": [155.0],
        "entry_iv": [0.3],
        "exit_iv": [0.28],
        "entry_option_price": [5.0],
        "exit_option_price": [6.5],
        "contracts": [1],
        "capital_allocated": [500.0],
        "capital_allocated_used": [500.0],
        "pnl_dollars": [150.0],
        "pnl_pct": [0.3],
    })

    aggregated_df = pd.DataFrame([
        {
            "individual": ("AAPL US Equity", ["SIG_A", "SIG_B"]),
            "direction": "long",
            "metrics": {
                "dsr": 0.6,
                "bootstrap_calmar_lb": 0.4,
                "bootstrap_profit_factor_lb": 1.5,
                "expectancy": 0.15,
                "support": 260,
                "n_trades": 1,
                "fold_count": 1,
                "fold_coverage_ratio": 1.0,
                "outer_split_ids": ["OUTER_001"],
                "first_trigger": "2021-01-05",
                "last_trigger": "2021-01-12",
                "max_drawdown": -0.05,
                "cagr": 0.32,
                "eligible": True,
                "eligibility_reasons": [],
            },
            "trade_ledger": ledger,
            "objectives": [0.6, 0.4, 1.5],
        }
    ])

    signals_meta = [
        {"signal_id": "SIG_A", "description": "Signal A"},
        {"signal_id": "SIG_B", "description": "Signal B"},
    ]

    artifacts = save_results(
        aggregated_df,
        signals_meta,
        str(run_dir),
        splits,
        settings,
        data_end=pd.Timestamp("2021-02-01"),
    )

    summary_path = Path(artifacts['summary_path'])
    ledger_path = Path(artifacts['ledger_path'])
    split_path = Path(artifacts['split_path'])
    priors_csv_path = Path(artifacts['priors_csv'])
    priors_json_path = Path(artifacts['priors_json'])

    assert summary_path.exists()
    assert ledger_path.exists()
    assert split_path.exists()
    assert priors_csv_path.exists()
    assert priors_json_path.exists()

    summary = pd.read_csv(summary_path)
    assert "dsr" in summary.columns
    assert summary.loc[0, "ticker"] == "AAPL US Equity"

    with open(priors_json_path) as f:
        payload = json.load(f)
    assert isinstance(payload, list)
    assert payload[0]["setup_id"].startswith("SETUP_")
    assert payload[0]["trades_total"] == 1
