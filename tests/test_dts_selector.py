from alpha_discovery.dts.selector import DailyTradeSelector, SelectorConfig


def make_priors():
    return [
        {
            'setup_id': 'SETUP_GOOD',
            'ticker': 'AAPL',
            'signals_list': ['SIG_A', 'SIG_B'],
            'dsr_score': 0.8,
            'bootstrap_calmar_lb_score': 0.6,
            'bootstrap_profit_factor_lb_score': 1.4,
            'dsr_source': 'dsr',
            'bootstrap_calmar_lb_source': 'bootstrap_calmar_lb',
            'bootstrap_profit_factor_lb_source': 'bootstrap_profit_factor_lb',
            'trades_total': 40,
            'trades_12m': 6,
            'trades_6m': 3,
            'flags': {'eligible': False, 'psr_ok': True, 'dd_ok': True},
            'penalty_scalar': 1.2,
            'last_trigger': '2025-09-25',
        },
        {
            'setup_id': 'SETUP_SPARSE',
            'ticker': 'MSFT',
            'signals_list': ['SIG_C', 'SIG_D'],
            'dsr_score': 0.3,
            'bootstrap_calmar_lb_score': 0.1,
            'bootstrap_profit_factor_lb_score': 0.9,
            'dsr_source': 'dsr',
            'bootstrap_calmar_lb_source': 'calmar',
            'bootstrap_profit_factor_lb_source': 'profit_factor',
            'trades_total': 8,
            'trades_12m': 1,
            'trades_6m': 0,
            'flags': {'eligible': True, 'psr_ok': True, 'dd_ok': True},
            'penalty_scalar': 1.0,
            'last_trigger': None,
        },
    ]


def make_triggers():
    return {
        'SETUP_GOOD': {
            'fired_any': True,
            'fired_all': True,
            'signal_hits': {'SIG_A': True, 'SIG_B': True},
            'trigger_strength_all': 1.0,
            'trigger_strength_soft': 1.0,
            'total_trigger_count': 2,
            'most_recent_all': '2025-09-26',
            'most_recent_any': '2025-09-26',
        },
        'SETUP_SPARSE': {
            'fired_any': False,
            'fired_all': False,
            'signal_hits': {'SIG_C': False, 'SIG_D': False},
            'trigger_strength_all': 0.0,
            'trigger_strength_soft': 0.0,
            'total_trigger_count': 0,
        },
    }


def test_dts_selector_live_and_blockers():
    selector = DailyTradeSelector(
        priors=make_priors(),
        trigger_map=make_triggers(),
        as_of='2025-09-26',
        config=SelectorConfig(max_positions=5, recent_trades_floor=3, min_total_trades=15, mode='soft_and'),
    )

    live_df = selector.run()

    assert not live_df.empty
    assert set(live_df['setup_id']) == {'SETUP_GOOD'}
    summary = selector.summary()
    assert summary['n_checked'] == 2
    assert summary['n_final_selected'] == 1
    assert summary['n_scored_with_bootstrap'] == 1
