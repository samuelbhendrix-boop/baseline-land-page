# baseline_v1.py
"""
Baseline v1 — trajectory + attribution engine (manual-friendly, conservative).

Goals:
- Robust to noise
- Works without bank linking (manual L_t + S_user)
- Produces one weekly signal: MoF, ΔMoF, cause, leverage
- Minimal pattern detectors (time-of-day, weekend, small-leaks, big-ticket)

This file is "real" in the sense that you can import it and run it with your own
transaction data. It does NOT include Plaid or persistence (DB) — you plug those in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Iterable
import math
import statistics


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Transaction:
    """
    amount: outflows should be POSITIVE dollars spent (e.g., 12.34 means spent $12.34)
            inflows can be represented as negative amounts, but v1 ignores inflows unless
            you pass them separately.
    ts: timezone-aware datetime strongly recommended.
    merchant: optional normalized merchant string
    description: optional raw description
    """
    ts: datetime
    amount: float
    merchant: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class WeeklySignal:
    week_end: datetime
    mof: float
    delta_mof: float
    week_type: str  # "non_representative" | "liquidity_event" | "behavioral"
    cause: str
    leverage: str
    baseline_monthly_burn: float
    confidence: float  # 0..1 (used to modulate tone, not content in v1)


# -----------------------------
# Helpers
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def robust_median(values: List[float]) -> float:
    if not values:
        raise ValueError("robust_median: empty values")
    return float(statistics.median(values))


def winsorize(values: List[float], p: float = 0.1) -> List[float]:
    """
    Winsorize by clamping to [p, 1-p] quantiles.
    p=0.1 means clamp bottom 10% and top 10%.
    """
    if not values:
        return []
    if not (0 <= p < 0.5):
        raise ValueError("winsorize p must be in [0, 0.5)")
    xs = sorted(values)
    n = len(xs)
    lo_i = int(math.floor(p * (n - 1)))
    hi_i = int(math.ceil((1 - p) * (n - 1)))
    lo = xs[lo_i]
    hi = xs[hi_i]
    return [clamp(v, lo, hi) for v in values]


def start_of_week(dt: datetime, week_start: int = 0) -> datetime:
    """
    week_start: 0=Monday ... 6=Sunday
    Returns dt truncated to the beginning of its week.
    """
    if dt.tzinfo is None:
        raise ValueError("start_of_week requires timezone-aware datetime")
    delta_days = (dt.weekday() - week_start) % 7
    d0 = dt - timedelta(days=delta_days)
    return d0.replace(hour=0, minute=0, second=0, microsecond=0)


def week_window(week_end: datetime, week_start: int = 0) -> Tuple[datetime, datetime]:
    """
    Returns (week_start_dt, week_end_dt) where week_end_dt is inclusive boundary
    for filtering transactions (<= week_end_dt).
    We treat a "week" as [start, start+7days).
    """
    if week_end.tzinfo is None:
        raise ValueError("week_window requires timezone-aware datetime")
    ws = start_of_week(week_end, week_start=week_start)
    we = ws + timedelta(days=7)
    return ws, we


def filter_txns(txns: Iterable[Transaction], start: datetime, end: datetime) -> List[Transaction]:
    """
    Keep txns with ts in [start, end).
    """
    out = []
    for t in txns:
        if t.ts.tzinfo is None:
            raise ValueError("Transaction.ts must be timezone-aware")
        if start <= t.ts < end:
            out.append(t)
    return out


def outflows_only(txns: Iterable[Transaction]) -> List[Transaction]:
    """
    Keep transactions that are outflows: amount > 0.
    """
    return [t for t in txns if t.amount > 0]


def sum_amount(txns: Iterable[Transaction]) -> float:
    return float(sum(t.amount for t in txns))


# -----------------------------
# Estimators
# -----------------------------

def estimate_monthly_burn_from_txns(
    txns: List[Transaction],
    asof: datetime,
    lookback_weeks: int = 8,
    week_start: int = 0,
) -> Optional[float]:
    """
    Estimate monthly burn from transaction history.
    Uses weekly totals → monthly equivalents → winsorized rolling median.

    Returns None if insufficient txns.
    """
    if asof.tzinfo is None:
        raise ValueError("asof must be timezone-aware")
    if lookback_weeks < 3:
        lookback_weeks = 3

    # Collect weekly spends ending at each week boundary
    ws_asof = start_of_week(asof, week_start=week_start)
    weekly_totals: List[float] = []

    for k in range(lookback_weeks):
        w_end = ws_asof - timedelta(days=7 * k)
        w_start = w_end - timedelta(days=7)
        week_txns = outflows_only(filter_txns(txns, w_start, w_end))
        total = sum_amount(week_txns)
        weekly_totals.append(total)

    # Require at least some non-zero weeks
    nonzero = [w for w in weekly_totals if w > 0]
    if len(nonzero) < 2:
        return None

    monthly_equiv = [w * 4.33 for w in weekly_totals]
    monthly_equiv_w = winsorize(monthly_equiv, p=0.1)
    return robust_median(monthly_equiv_w)


# -----------------------------
# Week classification
# -----------------------------

def classify_week(
    txns_week: Optional[List[Transaction]],
    baseline_monthly_burn: float,
    liquid_cash: float,
    prev_liquid_cash: Optional[float],
    anomaly_ratio_hi: float = 1.8,
    anomaly_ratio_lo: float = 0.4,
    big_txn_frac_of_month: float = 0.35,
    liquidity_event_frac_of_month: float = 0.25,
) -> str:
    """
    Priority:
    1) non_representative if txn-based anomaly suggests unusual week
    2) liquidity_event if cash changed a lot
    3) behavioral otherwise
    """
    if baseline_monthly_burn <= 0:
        raise ValueError("baseline_monthly_burn must be > 0")

    # txn-based anomaly
    if txns_week is not None:
        W_t = sum_amount(outflows_only(txns_week))
        W_expected = baseline_monthly_burn / 4.33
        ratio = (W_t / W_expected) if W_expected > 0 else 1.0

        biggest = 0.0
        out_txns = outflows_only(txns_week)
        if out_txns:
            biggest = max(t.amount for t in out_txns)

        if ratio > anomaly_ratio_hi or ratio < anomaly_ratio_lo or biggest > big_txn_frac_of_month * baseline_monthly_burn:
            return "non_representative"

    # liquidity event (works even without txns)
    if prev_liquid_cash is not None:
        cash_delta = liquid_cash - prev_liquid_cash
        if abs(cash_delta) > liquidity_event_frac_of_month * baseline_monthly_burn:
            return "liquidity_event"

    return "behavioral"


# -----------------------------
# Pattern detectors (v1)
# -----------------------------

@dataclass(frozen=True)
class PatternResult:
    name: str
    description: str
    impact_mof: float
    score: float


def dominant_pattern(
    txns_history: List[Transaction],
    week_end: datetime,
    baseline_monthly_burn: float,
    week_start: int = 0,
    min_impact_mof: float = 0.08,
) -> Optional[PatternResult]:
    """
    Find one dominant behavioral pattern this week vs user’s recent history.

    Approach:
    - Build "expected share" for each pattern using prior 6 behavioral-ish weeks (v1: we don't know week_type here,
      so we use last 6 weeks regardless; the anomaly filter earlier should have removed weird cases for attribution).
    - Compute this week's share for that pattern
    - Convert excess spend to impact_mof and score
    """
    if week_end.tzinfo is None:
        raise ValueError("week_end must be timezone-aware")

    # windows
    ws, we = week_window(week_end, week_start=week_start)
    txns_week = outflows_only(filter_txns(txns_history, ws, we))
    W_t = sum_amount(txns_week)
    if W_t <= 0:
        return None

    # Build prior weeks baseline shares (last 6 weeks)
    prior_shares = {k: [] for k in ["late", "weekend", "small_leaks", "big_ticket"]}
    for i in range(1, 7):
        p_end = ws - timedelta(days=7 * (i - 1))
        p_start = p_end - timedelta(days=7)
        tx = outflows_only(filter_txns(txns_history, p_start, p_end))
        total = sum_amount(tx)
        if total <= 0:
            continue
        prior_shares["late"].append(_share_late(tx, total))
        prior_shares["weekend"].append(_share_weekend(tx, total))
        prior_shares["small_leaks"].append(_share_small_leaks(tx, total))
        prior_shares["big_ticket"].append(_share_big_ticket(tx, total))

    # Expected shares default to a conservative prior if insufficient history
    def expected_share(key: str, default: float) -> float:
        xs = prior_shares[key]
        if len(xs) >= 3:
            return float(statistics.median(xs))
        return default

    exp_late = expected_share("late", 0.10)
    exp_weekend = expected_share("weekend", 0.28)
    exp_small = expected_share("small_leaks", 0.20)
    exp_big = expected_share("big_ticket", 0.35)

    # This week's shares
    sh_late = _share_late(txns_week, W_t)
    sh_weekend = _share_weekend(txns_week, W_t)
    sh_small = _share_small_leaks(txns_week, W_t)
    sh_big = _share_big_ticket(txns_week, W_t)

    # Convert excess spend to MoF impact
    # impact_mof = (excess_amount / monthly_burn)
    candidates: List[PatternResult] = []
    candidates.extend(_pattern_candidate("late", "late-evening convenience spending", sh_late, exp_late, W_t, baseline_monthly_burn))
    candidates.extend(_pattern_candidate("weekend", "weekend-driven discretionary spending", sh_weekend, exp_weekend, W_t, baseline_monthly_burn))
    candidates.extend(_pattern_candidate("small_leaks", "many small purchases (small leaks)", sh_small, exp_small, W_t, baseline_monthly_burn))
    candidates.extend(_pattern_candidate("big_ticket", "a few larger purchases dominating the week", sh_big, exp_big, W_t, baseline_monthly_burn))

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c.score)
    if abs(best.impact_mof) < min_impact_mof:
        return None
    return best


def _pattern_candidate(name: str, desc: str, share: float, exp_share: float, week_total: float, monthly_burn: float) -> List[PatternResult]:
    # Excess is (share - expected_share) * week_total
    excess_amount = (share - exp_share) * week_total
    impact_mof = excess_amount / monthly_burn
    # Repeat likelihood proxy: how far it deviates (bigger deviation = more "pattern-like")
    repeat_like = clamp(abs(share - exp_share) / 0.25, 0.0, 1.0)
    score = abs(impact_mof) * (0.6 + 0.4 * repeat_like)
    if score <= 0:
        return []
    return [PatternResult(name=name, description=desc, impact_mof=impact_mof, score=score)]


def _share_late(txns: List[Transaction], total: float) -> float:
    # Late = 8pm–5am local time of transaction timestamp
    late_amt = 0.0
    for t in txns:
        hr = t.ts.hour
        if (hr >= 20) or (hr < 5):
            late_amt += t.amount
    return late_amt / total if total > 0 else 0.0


def _share_weekend(txns: List[Transaction], total: float) -> float:
    wk_amt = 0.0
    for t in txns:
        if t.ts.weekday() >= 5:  # 5=Sat,6=Sun
            wk_amt += t.amount
    return wk_amt / total if total > 0 else 0.0


def _share_small_leaks(txns: List[Transaction], total: float, lo: float = 5.0, hi: float = 30.0) -> float:
    amt = sum(t.amount for t in txns if lo <= t.amount <= hi)
    return amt / total if total > 0 else 0.0


def _share_big_ticket(txns: List[Transaction], total: float, k: int = 2) -> float:
    if not txns:
        return 0.0
    top = sorted((t.amount for t in txns), reverse=True)[:k]
    return sum(top) / total if total > 0 else 0.0


# -----------------------------
# Text generation (tone)
# -----------------------------

def format_weekly_copy(
    mof: float,
    delta_mof: float,
    week_type: str,
    baseline_monthly_burn: float,
    pattern: Optional[PatternResult],
    delta_behavior_mof: Optional[float],
    confidence: float,
) -> Tuple[str, str]:
    """
    Returns (cause_sentence, leverage_sentence).
    Keep it calm. No shame. No hype.
    """

    # Conservative language modulation
    # confidence < 0.5 => softer verbs
    verb = "suggests" if confidence < 0.5 else "shows"
    change_word = "didn’t meaningfully change" if abs(delta_mof) < 1e-9 else "changed"

    if week_type == "non_representative":
        cause = "This week was non-representative (unusual expenses), so Baseline isn’t updating your pattern from it."
        leverage = "Your long-term trajectory is best read from a normal week."
        return cause, leverage

    if week_type == "liquidity_event":
        cause = f"This {change_word} was driven mainly by a one-time cash event, not a change in lifestyle burn."
        leverage = "If your baseline spending stays stable, this cushion will work quietly in the background."
        return cause, leverage

    # Behavioral
    if abs(delta_mof) < 1e-9:
        cause = "This week didn’t meaningfully change your trajectory."
        leverage = "Small variations are normal; Baseline prioritizes stability over noise."
        return cause, leverage

    if pattern is None or delta_behavior_mof is None:
        cause = f"Baseline {verb} your spending ran meaningfully above/below your usual baseline this week, which moved your trajectory."
        leverage = "If this pattern repeats, it will quietly compound over time."
        return cause, leverage

    # Pattern-based copy
    direction = "more" if pattern.impact_mof < 0 else "less"
    cause = f"Most of the change came from {pattern.description} ({direction} than your usual pattern)."

    # Conservative annualization: use delta_behavior_mof rather than raw delta_mof (focus on behavior)
    annualized = clamp(delta_behavior_mof * (52 / 4.33), -6.0, 6.0)
    leverage = f"If this holds, that’s roughly {annualized:+.1f} months of freedom over a year."
    return cause, leverage


# -----------------------------
# Main entry point
# -----------------------------

def compute_weekly_signal(
    *,
    week_end: datetime,
    liquid_cash: float,
    prev_liquid_cash: Optional[float],
    prev_mof: Optional[float],
    txns_history: Optional[List[Transaction]],
    manual_monthly_spend: Optional[float],
    week_start: int = 0,
) -> WeeklySignal:
    """
    Compute Baseline weekly signal.

    You supply:
    - week_end (tz-aware)
    - liquid_cash and optionally prev_liquid_cash
    - prev_mof (from last week), optional for first week
    - txns_history: recommended last 8+ weeks of transactions (outflows positive)
    - manual_monthly_spend: used when txns insufficient

    Returns WeeklySignal with cause/leverage sentences.
    """
    if week_end.tzinfo is None:
        raise ValueError("week_end must be timezone-aware")
    if liquid_cash < 0:
        raise ValueError("liquid_cash must be >= 0")

    txns_history = txns_history or []

    # Estimate baseline monthly burn
    est = estimate_monthly_burn_from_txns(txns_history, asof=week_end, lookback_weeks=8, week_start=week_start)
    if est is None:
        if manual_monthly_spend is None or manual_monthly_spend <= 0:
            raise ValueError("Need either sufficient transactions or a positive manual_monthly_spend.")
        baseline_burn = float(manual_monthly_spend)
        confidence = 0.45  # manual mode starts lower
    else:
        baseline_burn = float(est)
        confidence = 0.70

    # Months of Freedom
    mof = liquid_cash / baseline_burn

    # Delta MoF (noise gate)
    prev_mof_val = prev_mof if (prev_mof is not None and prev_mof >= 0) else mof
    delta_raw = mof - prev_mof_val

    epsilon = 0.3 if prev_mof is None else 0.2
    delta_mof = 0.0 if abs(delta_raw) < epsilon else delta_raw

    # Week txns slice for classification + attribution
    ws, we = week_window(week_end, week_start=week_start)
    txns_week = outflows_only(filter_txns(txns_history, ws, we)) if txns_history else None

    week_type = classify_week(
        txns_week=txns_week,
        baseline_monthly_burn=baseline_burn,
        liquid_cash=liquid_cash,
        prev_liquid_cash=prev_liquid_cash,
    )

    # Attribution
    pattern = None
    delta_behavior_mof = None

    if week_type == "behavioral" and txns_week is not None and abs(delta_mof) > 0:
        W_t = sum_amount(txns_week)
        W_expected = baseline_burn / 4.33
        delta_behavior_mof = (W_expected - W_t) / baseline_burn

        # Dominant pattern (uses history)
        pattern = dominant_pattern(txns_history, week_end=week_end, baseline_monthly_burn=baseline_burn, week_start=week_start)

        # If baseline is transaction-based but week is very sparse, reduce confidence
        if len(txns_week) < 5:
            confidence = min(confidence, 0.55)

    # Copy
    cause, leverage = format_weekly_copy(
        mof=mof,
        delta_mof=delta_mof,
        week_type=week_type,
        baseline_monthly_burn=baseline_burn,
        pattern=pattern,
        delta_behavior_mof=delta_behavior_mof,
        confidence=confidence,
    )

    return WeeklySignal(
        week_end=week_end,
        mof=mof,
        delta_mof=delta_mof,
        week_type=week_type,
        cause=cause,
        leverage=leverage,
        baseline_monthly_burn=baseline_burn,
        confidence=confidence,
    )


# -----------------------------
# Quick demo runner (optional)
# -----------------------------
if __name__ == "__main__":
    # Example: generate a fake week with late-night spending
    tz = timezone.utc
    now = datetime(2025, 12, 29, 12, 0, tzinfo=tz)  # pick a Monday-ish week end reference
    week_end = start_of_week(now) + timedelta(days=7)  # end boundary for the week

    # Fake txns over last 8 weeks
    txns = []
    for w in range(1, 9):
        base = start_of_week(week_end) - timedelta(days=7 * w)
        # Normal spend
        txns.append(Transaction(ts=base + timedelta(days=1, hours=14), amount=120.0, merchant="Groceries"))
        txns.append(Transaction(ts=base + timedelta(days=3, hours=12), amount=80.0, merchant="Gas"))
        txns.append(Transaction(ts=base + timedelta(days=5, hours=19), amount=60.0, merchant="Dining"))
        # Add late-night spike for most recent week
        if w == 1:
            txns.append(Transaction(ts=base + timedelta(days=5, hours=23), amount=95.0, merchant="Delivery"))

    sig = compute_weekly_signal(
        week_end=week_end,
        liquid_cash=18000.0,
        prev_liquid_cash=17500.0,
        prev_mof=4.1,
        txns_history=txns,
        manual_monthly_spend=4200.0,
        week_start=0,
    )

    print("MoF:", round(sig.mof, 2), "ΔMoF:", round(sig.delta_mof, 2), "Type:", sig.week_type)
    print("Cause:", sig.cause)
    print("Leverage:", sig.leverage)
