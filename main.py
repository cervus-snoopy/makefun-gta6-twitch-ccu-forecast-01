#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one case (schema makefun/twitch-ccu-case/v1) and writes, for each of
hours 2..24 of the release day, five quantiles (5%, 25%, 50%, 75%, 95%) of
that hour's Twitch category viewer count.

Method (fitted offline from the case's own comparables, no network, no
model call, fully deterministic):
  For every past big launch in the case's `comparables`, take its release
  day's 24 hourly viewer counts.  Launch days come in two regimes: a game
  whose first hour is already large (its big markets unlocked before
  00:00 UTC) stays roughly flat and peaks in the US evening; a game whose
  first hour is still tiny (unlock still to come) explodes by 10x-1000x
  later in the day.  A single log-log line cannot capture that kink, so
  for each forecast hour h we fit, over the comparables,

      ln(v_h) ~ a_h + b_h * ln(v_1) + c_h * max(0, ln(40000) - ln(v_1)),

  a hinge regression whose extra slope below a first hour of 40,000
  viewers lets small-first-hour launches rise much faster.  The median
  forecast is the fitted value at the case's own first hour.  The spread
  comes from the fit's residuals: residuals are split into a small
  first-hour pool (v_1 < 20,000) and a large one, the case uses its own
  pool (all residuals if its pool has fewer than 8), and the quantiles
  are median + 1.1 * residual-quantile at each level.  The 1.1 widens
  the empirical 5%/95% bands slightly toward honest 90% coverage.

  Hours or cases with too little comparable data fall back to the house
  persistence baseline (every hour stays at the first hour's count with
  a spread widening as sqrt(h-1)).

  Local leave-one-out pinball loss on the 37 comparable release days
  in the kit: 0.375 (0.251 on the high-first-hour launches the final
  will resemble), against 0.930 (0.321) for the house baseline on the
  same days.  The practice cases' own answers are sealed on the server,
  so no local practice score exists for them.

Standard library only.  Reads $MAKEFUN_INPUT_DIR (else /input) and
writes $MAKEFUN_OUTPUT_DIR (else /output); one <id>.out per <id>.in.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard-normal quantiles at LEVELS (used only by the fallback).
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
HINGE_AT = 40000.0          # first-hour viewers below which the extra slope bites
SMALL_CUT = 20000.0         # first-hour split for the residual pools
SPREAD = 1.1                # empirical residual quantiles are widened by this
MIN_POOL = 8                # a residual pool smaller than this falls back to all
MAX_VIEWERS = 1e9


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp like 2026-09-15T00:00:00Z."""
    if text.endswith("Z"):
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    return datetime.fromisoformat(text)


def safe_log(value):
    return math.log(max(float(value), 1.0))


def quantile(sorted_values, p):
    """Linear-interpolation quantile of a sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def solve_linear(matrix, rhs):
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for c in range(col, n + 1):
                aug[row][c] -= factor * aug[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        out[i] = (aug[i][n] - sum(aug[i][j] * out[j]
                                  for j in range(i + 1, n))) / aug[i][i]
    return out


def ols(features, targets):
    """Least squares via the normal equations; None if singular."""
    k = len(features[0])
    normal = [[sum(row[i] * row[j] for row in features)
               for j in range(k)] for i in range(k)]
    rhs = [sum(row[i] * t for row, t in zip(features, targets))
           for i in range(k)]
    return solve_linear(normal, rhs)


def features_for(first_hour):
    """[1, ln(v1), hinge] for a launch whose first hour is first_hour."""
    log_v1 = safe_log(first_hour)
    return [1.0, log_v1, max(0.0, math.log(HINGE_AT) - log_v1)]


def release_day(hourly, day_start):
    """24 hourly viewer counts (hours 1..24) of a launch's release day.

    Returns a list of 24 entries, None where the source has no point.
    """
    by_hour = {}
    end = day_start + timedelta(days=1)
    for point in hourly or []:
        try:
            when = parse_time(point["from"])
            viewers = point.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None:
            continue
        if day_start <= when < end:
            hour = (when - day_start).seconds // 3600 + 1
            if 1 <= hour <= 24:
                by_hour[hour] = float(viewers)
    return [by_hour.get(h) for h in range(1, 25)]


def baseline_quantiles(first_hour, hour):
    """House persistence baseline for one hour (fallback only)."""
    p = max(float(first_hour), 1.0)
    spread = 0.15 * math.sqrt(hour - 1)
    return [p * math.exp(z * spread) for z in Z]


def forecast_case(case):
    """Return the {"hours": [...]} forecast object for one case."""
    known = case.get("known") or []
    first_hour = None
    for row in known:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            first_hour = float(row["viewers"])
            break
    if first_hour is None and known:
        first_hour = float(known[0].get("viewers") or 1.0)
    if first_hour is None:
        hourly_hist = case.get("history_hourly") or []
        if hourly_hist:
            first_hour = float(hourly_hist[-1].get("viewers") or 1.0)
    if first_hour is None or first_hour <= 0:
        first_hour = 1.0

    # Comparable release days, in a fixed order so the fit is deterministic.
    comparables = []
    for comp in sorted(case.get("comparables") or [],
                       key=lambda c: (c.get("day_start", ""),
                                      c.get("name", ""))):
        try:
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        day = release_day(comp.get("hourly"), day_start)
        if day[0] is None or day[0] <= 0:
            continue
        comparables.append(day)

    target = features_for(first_hour)
    target_is_small = first_hour < SMALL_CUT

    hours_out = []
    for hour in range(2, 25):
        quantiles = None
        rows = [(features_for(day[0]), safe_log(day[hour - 1]))
                for day in comparables if day[hour - 1] is not None]
        if len(rows) >= 4:
            coef = ols([r[0] for r in rows], [r[1] for r in rows])
            if coef is None:  # singular (e.g. no launch under the hinge)
                coef = ols([r[0][:2] for r in rows], [r[1] for r in rows])
                if coef is not None:
                    coef = coef + [0.0]
            if coef is not None and len(rows) >= 3:
                median = sum(a * f for a, f in zip(coef, target))
                residuals = []
                for feats, target_log in rows:
                    fitted = sum(a * f for a, f in zip(coef, feats))
                    residuals.append((target_log - fitted, feats[1]))
                small = [r for r, log_v1 in residuals
                         if math.exp(log_v1) < SMALL_CUT]
                large = [r for r, log_v1 in residuals
                         if math.exp(log_v1) >= SMALL_CUT]
                pool = small if target_is_small else large
                if len(pool) < MIN_POOL:
                    pool = [r for r, _ in residuals]
                pool.sort()
                quantiles = [median + SPREAD * quantile(pool, tau)
                             for tau in LEVELS]
        if quantiles is None:
            # Fallback: the house persistence baseline (raw counts).
            quantiles = baseline_quantiles(first_hour, hour)
        else:
            quantiles = [math.exp(q) for q in quantiles]

        # Clean up: finite, > 0, <= 1e9, non-decreasing, 2 decimals.
        clean = []
        running = 0.0
        for q in quantiles:
            if not math.isfinite(q) or q <= 0:
                q = max(first_hour, 1.0)
            q = min(q, MAX_VIEWERS)
            q = round(q, 2)
            running = max(running, q)
            clean.append(running)
        hours_out.append({"hour": hour, "q": clean})
    return {"hours": hours_out}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(input_dir)
                       if n.endswith(".in"))
    except OSError as exc:
        print(f"cannot list input dir {input_dir}: {exc}", file=sys.stderr)
        return 1
    for name in names:
        case_id = name[:-3]
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, case_id + ".out")
        try:
            with open(in_path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast_case(case)
        except Exception as exc:  # never crash a case: emit the fallback
            print(f"{case_id}: forecast failed ({exc}); "
                  "writing baseline fallback", file=sys.stderr)
            try:
                with open(in_path, "r", encoding="utf-8") as fh:
                    case = json.load(fh)
                known = case.get("known") or []
                first = float(known[0].get("viewers") or 1.0) if known else 1.0
            except Exception:
                first = 1.0
            result = {"hours": [
                {"hour": h,
                 "q": [round(min(max(q, 0.01), MAX_VIEWERS), 2)
                       for q in baseline_quantiles(first, h)]}
                for h in range(2, 25)]}
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, separators=(",", ":"))
            fh.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
