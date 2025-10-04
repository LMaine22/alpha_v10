"""Daily Trade Selector (DTS) with today-first scoring and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from collections import Counter

import numpy as np
import pandas as pd

from alpha_discovery.utils.trade_keys import trade_uniq_key


@dataclass
class SelectorWeights:
    """Weights applied to the core scoring components."""

    dsr: float = 1.0
    calmar: float = 0.9
    profit_factor: float = 0.6
    penalty: float = 1.0
    recency: float = 0.4
    trigger: float = 0.5


@dataclass
class SelectorConfig:
    """Configuration bundle for the DailyTradeSelector."""

    weights: SelectorWeights = field(default_factory=SelectorWeights)
    dormancy_half_life_days: int = 180
    recency_floor: float = 0.25
    entry_window_days: int = 5
    recent_trades_floor: int = 3
    min_total_trades: int = 5
    min_prior_score: float = -999.0
    max_positions: int = 10
    kelly_cap: float = 0.4
    penalty_scalar_weight: float = 1.0
    allow_penalized: bool = True
    mode: str = "soft_and"  # "soft_and" or "strict_and"
    soft_and_penalty: float = 0.15
    friction_penalty: float = 0.0
    recent_trigger_window_days: int = 30
    cooldown_days: int = 5


class DailyTradeSelector:
    """Rank setups for today based on priors and live triggers."""

    def __init__(
        self,
        priors: Iterable[Dict[str, Any]] | pd.DataFrame,
        trigger_map: Optional[Dict[str, Dict[str, Any]]] = None,
        as_of: Optional[pd.Timestamp] = None,
        config: Optional[SelectorConfig] = None,
        dup_stats: Optional[Dict[str, int]] = None,
    ) -> None:
        self.priors = self._coerce_priors(priors)
        self.trigger_map = trigger_map or {}
        self.as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.utcnow().normalize()
        self.config = config or SelectorConfig()
        self.last_all_candidates: pd.DataFrame = pd.DataFrame()
        self.last_blocker_counts: Dict[str, int] = {}
        self.summary_counts: Dict[str, int] = {}
        self.near_miss: List[Dict[str, Any]] = []
        self.dup_stats = dup_stats or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Score all setups and return the live slate sorted by TodayScore."""
        if self.priors.empty:
            self.last_all_candidates = pd.DataFrame()
            self.last_blocker_counts = {}
            self.summary_counts = {}
            self.near_miss = []
            return pd.DataFrame()

        rows = [self._score_record(rec.to_dict()) for _, rec in self.priors.iterrows()]
        result_df = pd.DataFrame(rows)
        if result_df.empty:
            self.last_all_candidates = result_df
            self.last_blocker_counts = {}
            self.summary_counts = {}
            self.near_miss = []
            return result_df

        result_df = self._apply_duplicate_suppression(result_df)

        result_df.sort_values(by=["candidate_score"], ascending=False, inplace=True, na_position="last")

        live_mask = result_df["status"] == "live"
        live_df = result_df[live_mask].copy()
        if not live_df.empty and len(live_df) > self.config.max_positions:
            trimmed_idx = live_df.index[self.config.max_positions:]
            result_df.loc[trimmed_idx, "status"] = "trimmed"
            result_df.loc[trimmed_idx, "position_size"] = 0.0
            result_df.loc[trimmed_idx, "why_primary"] = result_df.loc[trimmed_idx, "why_primary"].fillna("max_positions_exceeded")
            result_df.loc[trimmed_idx, "why_secondary"] = result_df.loc[trimmed_idx, "why_secondary"].fillna("max_positions_exceeded")
            live_df = result_df[(result_df["status"] == "live")].copy()

        if not live_df.empty:
            positive_scores = live_df["today_score"].clip(lower=0)
            total_score = positive_scores.sum()
            if total_score > 0:
                sizes = (positive_scores / total_score) * self.config.kelly_cap
            else:
                sizes = pd.Series(self.config.kelly_cap / max(len(live_df), 1), index=live_df.index)
            result_df.loc[live_df.index, "position_size"] = sizes.round(6)
        else:
            result_df["position_size"] = result_df.get("position_size", 0.0)

        self.last_all_candidates = result_df
        self.last_blocker_counts = self._compute_blocker_counts(result_df)
        self.summary_counts, self.near_miss = self._summarize(result_df)

        live_df = result_df[result_df["status"] == "live"].copy()
        live_df.sort_values(by="today_score", ascending=False, inplace=True)
        return live_df

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------
    def why_counts(self) -> Dict[str, int]:
        """Return aggregated blocker counts from the last run."""
        return dict(self.last_blocker_counts)

    def summary(self) -> Dict[str, int]:
        """Return cached run summary counters."""
        return dict(self.summary_counts)

    def near_miss_candidates(self) -> List[Dict[str, Any]]:
        """Return top near-miss candidates from the last run."""
        return list(self.near_miss)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_priors(priors: Iterable[Dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(priors, pd.DataFrame):
            df = priors.copy()
        else:
            df = pd.DataFrame(list(priors))
        if df.empty:
            return df
        if "setup_id" not in df.columns:
            raise ValueError("Priors must include a 'setup_id' column")
        return df

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            val = float(value)
        except (TypeError, ValueError):
            return default
        return float(val) if np.isfinite(val) else default

    def _recency_weight(self, last_trigger: Any) -> tuple[float, Optional[int]]:
        if last_trigger in (None, "", pd.NaT):
            return self.config.recency_floor, None
        try:
            last_ts = pd.to_datetime(last_trigger)
        except (ValueError, TypeError):
            return self.config.recency_floor, None
        delta_days = max(0, int((self.as_of - last_ts.normalize()).days))
        half_life = max(1, int(self.config.dormancy_half_life_days))
        weight = 0.5 ** (delta_days / half_life)
        weight = max(self.config.recency_floor, min(1.0, weight))
        return float(weight), delta_days

    def _evaluate_trigger(self, setup_id: str, signals: List[str]) -> Dict[str, Any]:
        trigger = self.trigger_map.get(setup_id, {}) or {}
        signal_hits = trigger.get('signal_hits') or {sig: False for sig in signals}
        fired_any = bool(trigger.get('fired_any'))
        fired_all = bool(trigger.get('fired_all'))
        mode = self.config.mode if self.config.mode in {"soft_and", "strict_and"} else "soft_and"
        trigger_strength_key = 'trigger_strength_all' if mode == 'strict_and' else 'trigger_strength_soft'
        strength = trigger.get(trigger_strength_key, 0.0)
        try:
            trigger_strength = float(max(0.0, min(1.0, strength))) if np.isfinite(strength) else 0.0
        except Exception:
            trigger_strength = 0.0
        info = dict(trigger)
        info.setdefault('signal_hits', signal_hits)
        info.setdefault('mode', mode)
        info.setdefault('trigger_strength', trigger_strength)
        return {
            'triggered_any': fired_any,
            'triggered_all': fired_all,
            'signal_hits': signal_hits,
            'trigger_strength': trigger_strength,
            'info': info,
        }

    def _score_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        setup_id = record.get("setup_id")
        ticker = record.get("ticker")
        signals_list = record.get('signals_list') or []
        options_structure_keys = record.get('options_structure_keys') or []
        trigger_eval = self._evaluate_trigger(setup_id, signals_list)
        triggered = trigger_eval['triggered_all'] if self.config.mode == 'strict_and' else trigger_eval['triggered_any']
        trigger_strength = trigger_eval['trigger_strength']
        signals_fp = record.get('signals_fingerprint') or trigger_eval['info'].get('signals_fingerprint')

        reasons: List[str] = []
        status = "live"

        trades_12m = int(record.get("trades_12m") or 0)
        trades_total = int(record.get("trades_total") or 0)
        if trades_total <= 0:
            reasons.append("fatal_zero_trades")
            status = "fatal"
        elif trades_total < self.config.min_total_trades:
            reasons.append("trades_total_lt_min")

        flags = record.get("flags", {}) or {}
        if not flags.get("eligible", True):
            reasons.append("discovery_ineligible")
        if flags.get("psr_ok") is False:
            reasons.append("psr_flag")
        if flags.get("dd_ok") is False:
            reasons.append("dd_flag")

        penalty_scalar = max(1.0, self._safe_float(record.get("penalty_scalar"), 1.0))

        dsr_val = self._safe_float(record.get("dsr_score", record.get("dsr")))
        calmar_val = self._safe_float(record.get("bootstrap_calmar_lb_score", record.get("bootstrap_calmar_lb")))
        pf_val = self._safe_float(record.get("bootstrap_profit_factor_lb_score", record.get("bootstrap_profit_factor_lb")))

        prior_score = (
            self.config.weights.dsr * dsr_val
            + self.config.weights.calmar * calmar_val
            + self.config.weights.profit_factor * pf_val
        ) / penalty_scalar

        if prior_score < self.config.min_prior_score:
            reasons.append("priors_too_weak")

        recency_weight, dormancy_days = self._recency_weight(record.get("last_trigger"))
        recency_weight = max(self.config.recency_floor, recency_weight)

        recency_penalty = 1.0
        if trades_12m < self.config.recent_trades_floor:
            reasons.append("insufficient_recent_trades")
            recency_penalty = 0.5

        soft_and_penalty = 0.0
        if self.config.mode == "soft_and" and trigger_eval['triggered_any'] and not trigger_eval['triggered_all']:
            soft_and_penalty = self.config.soft_and_penalty
            reasons.append("soft_and_partial")

        if not triggered:
            reasons.append("no_recent_trigger")
            if status != "fatal":
                status = "idle"

        used_bootstrap = any(
            record.get(f"{metric}_source") == metric
            for metric in ("bootstrap_calmar_lb", "bootstrap_profit_factor_lb")
        )

        trade_depth_penalty = 1.0
        if 0 < trades_total < self.config.min_total_trades:
            trade_depth_penalty = max(0.2, trades_total / max(1, self.config.min_total_trades))

        candidate_score = prior_score
        candidate_score *= recency_weight
        candidate_score *= recency_penalty
        candidate_score *= trade_depth_penalty
        if soft_and_penalty > 0:
            candidate_score *= (1.0 - soft_and_penalty)
        candidate_score = max(candidate_score, 0.0)
        if triggered:
            candidate_score += self.config.weights.trigger * trigger_strength
        candidate_score -= self.config.friction_penalty

        if penalty_scalar > 1.0:
            reasons.append("soft_flags_present")
            if not self.config.allow_penalized:
                status = "blocked"

        def _compress(val: float) -> float:
            if val >= 0:
                return float(np.log1p(val))
            return float(-np.log1p(-val))

        compressed_score = _compress(candidate_score)
        today_score = compressed_score if (triggered and status == "live") else None
        if today_score is not None and not np.isfinite(today_score):
            today_score = None
            status = "blocked"

        score_components = {
            "prior_score": prior_score,
            "recency_weight": recency_weight,
            "trigger_strength": trigger_strength,
            "penalty_scalar": penalty_scalar,
            "recency_penalty": recency_penalty,
            "trade_depth_penalty": trade_depth_penalty,
            "soft_and_penalty": soft_and_penalty,
            "raw_candidate_score": candidate_score,
            "compressed_score": compressed_score,
        }

        reasons = list(dict.fromkeys(reasons))
        primary = reasons[0] if reasons else None
        secondary = reasons[1] if len(reasons) > 1 else None

        return {
            "setup_id": setup_id,
            "ticker": ticker,
            "direction": record.get('direction'),
            "horizon": record.get('horizon'),
            "status": status,
            "today_score": today_score,
            "candidate_score": compressed_score,
            "position_size": 0.0,
            "why_primary": primary,
            "why_secondary": secondary,
            "reasons": reasons,
            "trades_12m": trades_12m,
            "trades_total": trades_total,
            "recency_days": dormancy_days,
            "score_components": score_components,
            "flags": flags,
            "trigger_info": trigger_eval['info'],
            "penalty_scalar": penalty_scalar,
            "triggered_any": trigger_eval['triggered_any'],
            "triggered_all": trigger_eval['triggered_all'],
            "used_bootstrap": used_bootstrap,
            "soft_flags_present": penalty_scalar > 1.0,
            "signals_fingerprint": signals_fp,
            "signals_list": signals_list,
            "options_structure_keys": options_structure_keys,
        }

    @staticmethod
    def _compute_blocker_counts(df: pd.DataFrame) -> Dict[str, int]:
        if df.empty:
            return {}
        reasons = []
        blocked = df[df["status"] != "live"]
        for col in ("why_primary", "why_secondary"):
            reasons.extend([r for r in blocked[col].tolist() if r])
        return dict(Counter(reasons))

    @staticmethod
    def _extract_trigger_strength(info: Dict[str, Any]) -> float:
        strength = 0.0
        if isinstance(info, dict):
            try:
                strength = float(info.get('trigger_strength', 0.0) or 0.0)
            except (TypeError, ValueError):
                strength = 0.0
        return max(0.0, min(1.0, strength))

    @staticmethod
    def _extract_recent_timestamp(info: Dict[str, Any]) -> pd.Timestamp:
        if not isinstance(info, dict):
            return pd.NaT
        for key in ('most_recent_all', 'most_recent_any'):
            value = info.get(key)
            if value:
                try:
                    return pd.to_datetime(value)
                except Exception:
                    continue
        return pd.NaT

    def _candidate_uniq_key(self, row: pd.Series) -> str:
        info = row.get('trigger_info') or {}
        entry_date = None
        if isinstance(info, dict):
            entry_date = info.get('most_recent_all') or info.get('most_recent_any')
        structure_keys = row.get('options_structure_keys') or []
        structure_key = structure_keys[0] if structure_keys else ''
        horizon_tag = f"H{int(row.get('horizon'))}" if row.get('horizon') not in (None, '') else 'H?'
        exit_policy_tag = info.get('exit_policy_id') if isinstance(info, dict) else ''
        return trade_uniq_key(
            setup_id=row.get('setup_id'),
            ticker=row.get('ticker'),
            direction=row.get('direction'),
            entry_date=entry_date,
            signals_fingerprint=row.get('signals_fingerprint') or '',
            horizon_tag=horizon_tag,
            exit_policy_tag=exit_policy_tag or '',
            structure_key=structure_key,
        )

    def _apply_duplicate_suppression(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or {'setup_id', 'ticker'}.isdisjoint(df.columns):
            self.dup_stats = {'dup_groups': 0, 'dup_suppressed': 0, 'dup_suppressed_cooldown': 0}
            return df

        working = df.copy()
        working['dup_group_size'] = 1
        working['dup_suppressed'] = False
        working['dup_reason'] = None
        working['uniq_key'] = working.apply(self._candidate_uniq_key, axis=1)
        working['_dup_strength'] = working['trigger_info'].apply(self._extract_trigger_strength)
        working['_dup_recent'] = working['trigger_info'].apply(self._extract_recent_timestamp)

        suppressed_total = 0
        cooldown_total = 0
        cooldown_days = max(0, int(self.config.cooldown_days))

        dup_group_count = 0
        for uniq_key, group in working.groupby('uniq_key', sort=False):
            if len(group) <= 1:
                continue

            dup_group_count += 1

            group_indices = group.index.tolist()
            working.loc[group_indices, 'dup_group_size'] = len(group)

            sorter = group.sort_values(
                by=['triggered_all', '_dup_strength', '_dup_recent', 'candidate_score'],
                ascending=[False, False, False, False]
            )

            keep_idx = sorter.index[0]
            keep_recent = working.at[keep_idx, '_dup_recent']
            keep_fp = working.at[keep_idx, 'signals_fingerprint']

            for idx in sorter.index[1:]:
                row = working.loc[idx]
                reason = 'duplicate_suppressed'
                within_cooldown = False
                row_recent = row['_dup_recent']
                if cooldown_days > 0 and pd.notna(keep_recent) and pd.notna(row_recent):
                    delta_days = abs((keep_recent.normalize() - row_recent.normalize()).days)
                    within_cooldown = delta_days < cooldown_days
                if within_cooldown and row.get('signals_fingerprint') == keep_fp:
                    reason = 'cooldown_active'
                    cooldown_total += 1
                elif bool(working.at[keep_idx, 'triggered_all']) and not bool(row.get('triggered_all')):
                    reason = 'duplicate_soft_any_suppressed'

                working.at[idx, 'dup_suppressed'] = True
                working.at[idx, 'dup_reason'] = reason
                if working.at[idx, 'status'] == 'live':
                    working.at[idx, 'status'] = 'duplicate_suppressed'
                working.at[idx, 'position_size'] = 0.0
                working.at[idx, 'today_score'] = None
                if not working.at[idx, 'why_primary']:
                    working.at[idx, 'why_primary'] = reason
                elif not working.at[idx, 'why_secondary']:
                    working.at[idx, 'why_secondary'] = reason
                suppressed_total += 1

        working.drop(columns=['_dup_strength', '_dup_recent'], inplace=True)
        self.dup_stats = {
            'dup_groups': int((working['dup_group_size'] > 1).sum()),
            'dup_suppressed': int(suppressed_total),
            'dup_suppressed_cooldown': int(cooldown_total),
            'dup_group_count': int(dup_group_count),
        }
        return working

    def _summarize(self, df: pd.DataFrame) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        if df.empty:
            return {
                'n_checked': 0,
                'n_recent_trigger': 0,
                'n_blocked_fatal': 0,
                'n_soft_flagged': 0,
                'n_scored_with_bootstrap': 0,
                'n_scored_point_estimate': 0,
                'n_final_selected': 0,
            }, []

        triggered_any = df.get('triggered_any', pd.Series(dtype=bool)).astype(bool)
        soft_flags = df.get('soft_flags_present', pd.Series(dtype=bool)).astype(bool)
        used_bootstrap = df.get('used_bootstrap', pd.Series(dtype=bool)).astype(bool)
        status_series = df.get('status', pd.Series(dtype=object))

        summary = {
            'n_checked': int(len(df)),
            'n_recent_trigger': int(triggered_any.sum()),
            'n_blocked_fatal': int((status_series == 'fatal').sum()),
            'n_soft_flagged': int(soft_flags.sum()),
            'n_scored_with_bootstrap': int(used_bootstrap.sum()),
            'n_scored_point_estimate': int(len(df) - used_bootstrap.sum()),
            'n_final_selected': int((status_series == 'live').sum()),
        }

        if hasattr(self, 'dup_stats') and isinstance(self.dup_stats, dict):
            summary['dup_groups'] = int(self.dup_stats.get('dup_groups', 0) or 0)
            summary['dup_suppressed'] = int(self.dup_stats.get('dup_suppressed', 0) or 0)
            summary['dup_suppressed_cooldown'] = int(self.dup_stats.get('dup_suppressed_cooldown', 0) or 0)
            summary['dup_group_count'] = int(self.dup_stats.get('dup_group_count', 0) or 0)

        near_miss: List[Dict[str, Any]] = []
        if summary['n_final_selected'] == 0 and 'candidate_score' in df.columns:
            dup_mask = df.get('dup_suppressed', pd.Series(False, index=df.index)).astype(bool)
            ranked = df.loc[~dup_mask].sort_values(by='candidate_score', ascending=False).head(5)
            for _, row in ranked.iterrows():
                near_miss.append({
                    'setup_id': row.get('setup_id'),
                    'ticker': row.get('ticker'),
                    'candidate_score': float(row.get('candidate_score', 0.0) or 0.0),
                    'reasons': row.get('reasons', []),
                })

        return summary, near_miss
