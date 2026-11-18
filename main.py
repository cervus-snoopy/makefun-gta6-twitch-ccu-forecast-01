"""GTA VI Twitch launch-day forecast — round-6 entry (stdlib only).

Method (see REPORT.md for the full spec and guard evidence):
  Per hour h = 2..24, forecast log(v_h / v1) from past big launches
  ("donors": the case's comparables with a complete release-day curve).
  * DST alignment: donor curves shift so their US-unlock hour matches
    the target's (US East unlocks 04:00Z in daylight time, 05:00Z in
    standard time; hours 1-4 never shift). The US unlock is midnight
    Eastern for the whole country (05:00Z standard), not 08:00Z.
  * Scale-conditioned filter (round 6): donors with peak/v1 < 1.3 are
    dropped, EXCEPT when the target itself is large (log v1 >= 12.6,
    ~296k viewers) — at that scale the flat big donors are the right
    analogues, and dropping them biases the median up. At practice
    scale the filter never fires, so practice is exactly the round-5
    entry; on the large-donor holdout it improves 0.1334 -> 0.1192.
  * Centre: 80% kernel local-linear (Gaussian kernel, bandwidth 1.75
    in log space, weighted least squares, ridge 1.0 on the slope) +
    20% LAD median fit of the same regression (iteratively reweighted
    least squares, 30 iterations, residual floor 0.05, started at the
    local-linear fit).
  * Clamp: the evaluation point is capped at the largest donor's
    log v1 + 0.05; the excess is added back as spread (k = 1.0).
    No-op on practice; pure insurance for a final v1 beyond 875k.
  * Momentum (minute data only): if hour 1 carries >= 4 minute reads
    with median gap < 10 min, M = mean log of the second half of the
    reads minus mean log of the first half; the median is nudged by
    0.5 * M on hours 2-4 with decay 1 / 0.5 / 0.25, clipped to +-0.4.
    Practice cases carry no minute data, so this never fires there.
  * CAL + RMS: per-hour calibration fitted on the 4 practice cases
    under this config (CAL = 0.88 * 50/50 blend of mean/median
    per-case base log-errors; RMS = RMS of residuals after CAL).
  * Spread: per-hour kernel-weighted SD of donor residuals against
    leave-one-out centres, smoothed over neighbours, floored at 0.05,
    shrunk by 0.35, rescaled per hour by RMS/SD clamped to [0.8, 1.6].
  Quantiles: q_tau = v1 * exp(median_log + z_tau * sd), z the standard
  normal quantiles at 0.05/0.25/0.5/0.75/0.95.

  Only the standard library is used. No randomness: repeated runs are
  byte-identical. Reads $MAKEFUN_INPUT_DIR (else /input/), writes
  $MAKEFUN_OUTPUT_DIR (else /output/): one .out per .in, hours 2-24.
"""

import json
import math
import os
from datetime import datetime, timedelta

# ---------------------------------------------------------------- config

BANDWIDTH = 1.75
RIDGE = 1.0
BLEND_LAD = 0.20
LAD_ITERS = 30
LAD_FLOOR = 0.05
FCUT = 1.3
FCUT_LARGE = 1.0
FCUT_LOGV1 = 12.6
CLAMP_PAD = 0.05
CLAMP_K = 1.0
SPREAD_SHRINK = 0.35
CAL_SHRINK = 0.88
MOM_COEF = 0.5
MOM_DECAY = (1.0, 0.5, 0.25)
MOM_CLIP = 0.4

HOURS = list(range(2, 25))
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)

# CAL / RMS re-fit in-session on the 4 practice cases under this config
# (scale filter never fires on practice, so these equal the round-5 fit).
CAL = [-0.09847482461071629, -0.10713665250499638, -0.10461015345648965,
       0.3424871127570653, 0.3748521980642472, 0.3158160814736069,
       0.19851081810673304, 0.14792428440225858, 0.06308099658939197,
       0.059930321933111944, -0.07777869803383325, -0.08074294189562405,
       -0.015788727997534568, 0.06090757130221535, 0.10224281690011493,
       0.2200289213195049, 0.3287422481496554, 0.2510821554877034,
       0.2589602533233662, 0.21959949582089527, 0.22017880272186194,
       0.2154053220571411, 0.18846459990572617]
RMS = [0.10824122996156606, 0.2740224333571639, 0.20468361329368365,
       0.20974595480685207, 0.145818280641575, 0.13828272121434573,
       0.14519215890481135, 0.15147162199649375, 0.1393479978321563,
       0.2023718579392226, 0.440762985425929, 0.3179009665177845,
       0.24401680579005505, 0.21959849969673087, 0.2081055727665788,
       0.23532544773655165, 0.1909689476959667, 0.11654363708697824,
       0.1056535761782624, 0.07950959569742917, 0.06605654289034589,
       0.05061537467554552, 0.09288472756605771]

# ---------------------------------------------------------------- dates


def _parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def _fmt_ts(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _us_dst(day):
    """True if a date falls in US daylight saving time: from the 2nd
    Sunday of March (inclusive) to the 1st Sunday of November."""
    year = day.year
    d = datetime(year, 3, 1)
    first_sun_mar = 1 + ((6 - d.weekday()) % 7)
    dst_start = datetime(year, 3, first_sun_mar + 7).date()
    d = datetime(year, 11, 1)
    first_sun_nov = 1 + ((6 - d.weekday()) % 7)
    dst_end = datetime(year, 11, first_sun_nov).date()
    return dst_start <= day < dst_end


def _us_unlock_hour(day):
    # US East unlocks at midnight Eastern: 04:00Z in DST, 05:00Z in
    # standard time. The whole US anchors to Eastern midnight.
    return 4 if _us_dst(day) else 5


# ---------------------------------------------------------------- donors


def _extract_donors(case):
    """Comparables with a complete release-day curve (hours 1..24,
    plus hour 25 when present, for DST shifts). Returns a list of
    dicts with name, day, x = log(v1), v1, y = {hour: log(v_h/v1)}
    and peak_ratio = max(v_1..24) / v1."""
    donors = []
    for comp in case.get("comparables") or []:
        hourly = comp.get("hourly") or []
        by_from = {}
        for row in hourly:
            try:
                v = row["viewers"]
            except (KeyError, TypeError):
                continue
            if isinstance(v, (int, float)) and v is not None:
                by_from[row["from"]] = v
        try:
            base = _parse_ts(comp["day_start"][:10] + "T00:00:00Z")
        except (KeyError, TypeError, ValueError):
            continue
        vals = {}
        ok = True
        for h in range(1, 26):
            v = by_from.get(_fmt_ts(base + timedelta(hours=h - 1)))
            if v is None or v <= 0:
                if h <= 24:
                    ok = False
                vals[h] = None
            else:
                vals[h] = v
        if not ok or vals[1] is None:
            continue
        v1 = vals[1]
        peak = max(vals[h] for h in range(1, 25))
        y = {}
        for h in range(1, 26):
            if vals[h] is not None:
                y[h] = 0.0 if h == 1 else math.log(vals[h] / v1)
        donors.append({"name": comp.get("name", "?"), "day": base.date(),
                       "x": math.log(v1), "v1": v1, "y": y,
                       "peak_ratio": peak / v1})
    return donors


def _shift_donors(donors, target_day):
    """Align donor curves to the target's US-unlock hour. Only hours
    >= 5 shift; hours 1-4 are NZ/Australia-driven and stay put. A
    shifted hour reads its source hour, clamped into [1, 25]."""
    t_us = _us_unlock_hour(target_day)
    out = []
    for d in donors:
        s = t_us - _us_unlock_hour(d["day"])
        if s == 0:
            out.append(d)
            continue
        y2 = dict(d["y"])
        for h in range(2, 25):
            if h >= 5:
                src = h - s
                if src < 1:
                    src = 1
                elif src > 25:
                    src = 25
                if src in d["y"]:
                    y2[h] = d["y"][src]
        nd = dict(d)
        nd["y"] = y2
        out.append(nd)
    return out


# ---------------------------------------------------------------- centre


def _kernel(dx):
    return math.exp(-0.5 * (dx / BANDWIDTH) ** 2)


def _wls(xs, ys, ws, ridge):
    """Weighted local-linear fit at one evaluation point. xs/ys/ws
    are parallel lists over donors. Returns (intercept, slope).
    Falls back to slope 0 when the weighted x-spread vanishes
    (single-donor pool: sxx + ridge == 0 with ridge 0)."""
    sw = sum(ws)
    if sw <= 0:
        return (sum(ys) / len(ys) if ys else 0.0), 0.0
    xb = sum(w * x for w, x in zip(ws, xs)) / sw
    yb = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = 0.0
    sxy = 0.0
    for w, x, y in zip(ws, xs, ys):
        dx = x - xb
        sxx += w * dx * dx
        sxy += w * dx * (y - yb)
    denom = sxx + ridge
    if denom == 0.0:
        slope = 0.0
    else:
        slope = sxy / denom
    return yb - slope * xb, slope


def _centre_at(xs, ys_by_hour, x0, exclude=None):
    """Centre log-ratios at eval point x0 for hours 2..24.
    ys_by_hour: list of 23 lists (one per hour) over donors.
    exclude: donor index to drop (leave-one-out) or None.
    Returns (centre, ll, lad), each a list of 23."""
    n = len(xs)
    w0 = []
    for i in range(n):
        if i == exclude:
            w0.append(0.0)
        else:
            w0.append(_kernel(x0 - xs[i]))
    centre = []
    ll_out = []
    lad_out = []
    for h in range(23):
        ys = ys_by_hour[h]
        a, b = _wls(xs, ys, w0, RIDGE)
        pred_ll = a + b * x0
        # LAD median fit by IRLS, started at the local-linear fit.
        aa, bb = a, b
        for _ in range(LAD_ITERS):
            w = []
            for i in range(n):
                r = ys[i] - (aa + bb * xs[i])
                ar = abs(r)
                if ar < LAD_FLOOR:
                    ar = LAD_FLOOR
                w.append(w0[i] / ar)
            aa, bb = _wls(xs, ys, w, 0.0)
        pred_lad = aa + bb * x0
        ll_out.append(pred_ll)
        lad_out.append(pred_lad)
        centre.append((1.0 - BLEND_LAD) * pred_ll + BLEND_LAD * pred_lad)
    return centre, ll_out, lad_out


def _smooth3(v):
    out = []
    n = len(v)
    for i in range(n):
        lo = i - 1 if i > 0 else 0
        hi = i + 2 if i + 2 <= n else n
        out.append(sum(v[lo:hi]) / (hi - lo))
    return out


# ---------------------------------------------------------------- momentum


def _minute_momentum(case):
    """Nudge (log units) for hours 2-4 from hour-1 minute reads.
    Returns a list of 23. Fires only with >= 4 reads inside hour 1
    whose median gap is < 10 minutes; otherwise all zeros."""
    nudge = [0.0] * 23
    try:
        day = _parse_ts(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return nudge
    pts = []
    for p in case.get("history") or []:
        try:
            t = _parse_ts(p["at"])
            v = p["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if day <= t < day + timedelta(hours=1) and v and v > 0:
            pts.append((t, v))
    pts.sort()
    if len(pts) < 4:
        return nudge
    gaps = sorted((pts[i + 1][0] - pts[i][0]).total_seconds() / 60.0
                  for i in range(len(pts) - 1))
    if gaps[len(gaps) // 2] >= 10:
        return nudge
    k = len(pts) // 2
    first = sum(math.log(v) for _, v in pts[:k]) / k
    last = sum(math.log(v) for _, v in pts[k:]) / (len(pts) - k)
    m = last - first
    for j, decay in enumerate(MOM_DECAY):
        val = MOM_COEF * m * decay
        if val > MOM_CLIP:
            val = MOM_CLIP
        elif val < -MOM_CLIP:
            val = -MOM_CLIP
        nudge[j] = val
    return nudge


# ---------------------------------------------------------------- forecast


def _forecast(case):
    """Returns {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    try:
        v1 = case["known"][0]["viewers"]
    except (KeyError, TypeError, IndexError):
        v1 = None
    if not v1 or v1 <= 0:
        v1 = 1
    log_v1 = math.log(v1)
    try:
        tday = _parse_ts(case["day_start"]).date()
    except (KeyError, TypeError, ValueError):
        tday = datetime(2026, 11, 19).date()

    donors = _shift_donors(_extract_donors(case), tday)
    mom = _minute_momentum(case)

    if not donors:
        # No donor pool at all: flat forecast, wide spread.
        out = {}
        for h in HOURS:
            qs = [min(v1 * math.exp(z * 0.5), 1e9) for z in Z]
            out[h] = qs
        return out

    # Scale-conditioned filter: keep flat donors when the target is
    # large; otherwise drop donors with peak/v1 < 1.3.
    fcut = FCUT_LARGE if log_v1 >= FCUT_LOGV1 else FCUT
    kept = [d for d in donors if d["peak_ratio"] >= fcut]
    if not kept:
        kept = donors

    max_x = max(d["x"] for d in donors)
    cap = max_x + CLAMP_PAD
    excess = log_v1 - cap
    if excess < 0.0:
        excess = 0.0
    x_eval = log_v1 if log_v1 < cap else cap

    xs = [d["x"] for d in kept]
    ys_by_hour = [[d["y"][h] for d in kept] for h in HOURS]
    centre, _, _ = _centre_at(xs, ys_by_hour, x_eval)

    # Spread: kernel-weighted SD of donor residuals against
    # leave-one-out centres, smoothed, floored, shrunk, rescaled.
    loo = []
    for i in range(len(kept)):
        c, _, _ = _centre_at(xs, ys_by_hour, xs[i], exclude=i)
        loo.append(c)
    sd_raw = []
    for h in range(23):
        w0 = [_kernel(x_eval - x) for x in xs]
        sw = sum(w0)
        acc = 0.0
        for i in range(len(kept)):
            r = ys_by_hour[h][i] - loo[i][h]
            acc += w0[i] * r * r
        sd_raw.append(math.sqrt(acc / sw) if sw > 0 else 0.0)
    sd_sm = _smooth3(sd_raw)
    sd_fin = []
    for h in range(23):
        s = sd_sm[h]
        if s < 0.05:
            s = 0.05
        s *= SPREAD_SHRINK
        factor = RMS[h] / s if s > 1e-9 else 1.0
        if factor < 0.8:
            factor = 0.8
        elif factor > 1.6:
            factor = 1.6
        s = s * factor + CLAMP_K * excess
        sd_fin.append(s)

    out = {}
    for h in range(23):
        med = centre[h] + CAL[h] + mom[h]
        qs = []
        for z in Z:
            q = v1 * math.exp(med + z * sd_fin[h])
            if q < 1e-6:
                q = 1e-6
            qs.append(q)
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        # The rules cap every quantile at 1,000,000,000 viewers.
        for j in range(5):
            if qs[j] > 1e9:
                qs[j] = 1e9
        out[HOURS[h]] = qs
    return out


# ---------------------------------------------------------------- main


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        cid = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        fc = _forecast(case)
        rows = [{"hour": h, "q": fc[h]} for h in HOURS]
        with open(os.path.join(out_dir, cid + ".out"), "w",
                  encoding="utf-8") as f:
            json.dump({"hours": rows}, f)


if __name__ == "__main__":
    main()
