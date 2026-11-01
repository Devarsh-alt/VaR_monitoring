
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'PROJECT_ROOT').is_file())
RS = ROOT / "data" / "monitoring" / "regime_shifts"
RS05 = RS / "RS05"

MODELS = ["EWMA", "GARCH-t", "HS", "QRF"]
MONITORS = ["S3", "S4", "S6"]
BACKTESTS = [
    "bt_count250", "bt_count60", "bt_count20",
    "kupiec250", "christoffersen250",
]
EVENT_THRESHOLDS = [0.25, 0.50, 0.75]
MATCH_WINDOWS = [10, 20, 30]
CAL_DATES = 252
BUDGET = 0.10
H = 10
ALPHA = 0.05
N_TICKERS = 24
START = pd.Timestamp("2020-01-01")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default=str(
        RS05 / "RS05_oos_predictions_H10_final.csv.gz"))
    p.add_argument("--panel", default=str(
        RS / 'RS04b_causal_reliability_panel.csv.gz'))
    p.add_argument("--quar", default=str(
        RS / 'RS04_quarantined_outcomes.csv.gz'))
    p.add_argument("--out", default=str(
        RS05 / "RS05_causal_system_alert_eval"))
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--block", type=int, default=20)
    p.add_argument("--event-gap", type=int, default=5)
    p.add_argument("--episode-gap", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261025)
    p.add_argument("--budget", type=float, default=BUDGET, help="Target alert fraction, e.g. 0.05, 0.10, 0.15")
    return p.parse_args()


def check_columns(path, required):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cols = set(pd.read_csv(path, nrows=0, compression="infer").columns)
    missing = sorted(set(required) - cols)
    if missing:
        raise KeyError(f"{path} missing columns: {missing}")


def load_predictions(path):
    cols = [
        "forecast_date", "ticker", "model", "target", "H",
        "spec", "algo", "y", "p_cal",
    ]
    check_columns(path, cols)
    d = pd.read_csv(path, usecols=cols, compression="infer", low_memory=False)
    d["forecast_date"] = pd.to_datetime(d["forecast_date"], errors="coerce")

    for c in ["ticker", "model", "target", "spec", "algo"]:
        d[c] = d[c].astype(str).str.strip()
    d["ticker"] = d["ticker"].str.upper()
    d["target"] = d["target"].str.upper()
    d["spec"] = d["spec"].str.upper()
    d["algo"] = d["algo"].str.lower()
    d["H"] = pd.to_numeric(d["H"], errors="coerce")

    for c in ["y", "p_cal"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d[
        d["target"].eq("T1")
        & d["H"].eq(H)
        & d["algo"].eq("logit")
        & d["model"].isin(MODELS)
        & d["forecast_date"].ge(START)
    ].dropna(subset=["forecast_date", "ticker", "model", "spec", "y"])

    bad = d.groupby(
        ["forecast_date", "ticker", "model"]
    )["y"].nunique()
    if (bad > 1).any():
        raise RuntimeError("T1 outcomes differ across model specifications.")

    missing = set(MONITORS) - set(d["spec"].unique())
    if missing:
        raise RuntimeError(f"Missing monitor specifications: {sorted(missing)}")

    return d.copy()


def load_panel(path):
    cols = [
        "forecast_date", "ticker", "model_name",
        "label_start_date_10d", "label_end_date_10d",
    ]
    check_columns(path, cols)
    d = pd.read_csv(path, usecols=cols, low_memory=False)
    d["forecast_date"] = pd.to_datetime(d["forecast_date"], errors="coerce")
    d["label_start_date_10d"] = pd.to_datetime(
        d["label_start_date_10d"], errors="coerce")
    d["label_end_date_10d"] = pd.to_datetime(
        d["label_end_date_10d"], errors="coerce")
    d["ticker"] = d["ticker"].astype(str).str.strip().str.upper()
    d["model"] = d["model_name"].astype(str).str.strip()

    if d.duplicated(["forecast_date", "ticker", "model"]).any():
        raise RuntimeError("Duplicate RS04b model/ticker/date keys.")

    calendar = pd.DatetimeIndex(
        sorted(d["forecast_date"].dropna().unique()))
    return d, calendar


def make_events_daily(pred, panel):
    priority = {"S6": 0, "S4": 1, "S3": 2, "S2": 3, "S5": 4, "S7": 5}
    x = pred.copy()
    x["_priority"] = x["spec"].map(priority).fillna(99)
    x = x.sort_values("_priority").drop_duplicates(
        ["forecast_date", "ticker", "model"])

    x = x[["forecast_date", "ticker", "model", "y"]].merge(
        panel[[
            "forecast_date", "ticker", "model",
            "label_start_date_10d", "label_end_date_10d",
        ]],
        on=["forecast_date", "ticker", "model"],
        how="left",
        validate="one_to_one",
    )

    d = x.groupby(["model", "forecast_date"], as_index=False).agg(
        n_tickers=("ticker", "nunique"),
        n_valid=("y", "count"),
        positive_fraction=("y", "mean"),
        label_start=("label_start_date_10d", "min"),
        label_end=("label_end_date_10d", "max"),
    )
    d["complete"] = d["n_tickers"].eq(N_TICKERS) & d["n_valid"].eq(N_TICKERS)
    return d


def extract_events(daily, threshold, max_gap):
    x = daily[daily["complete"]].sort_values("forecast_date").reset_index(drop=True)
    if x.empty:
        return pd.DataFrame()

    active = np.flatnonzero(x["positive_fraction"].to_numpy() >= threshold)
    if len(active) == 0:
        return pd.DataFrame()

    groups, current = [], [int(active[0])]
    for i in active[1:]:
        i = int(i)
        if i - current[-1] - 1 <= max_gap:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    rows = []
    for eid, ix in enumerate(groups, 1):
        first, last = x.iloc[ix[0]], x.iloc[ix[-1]]
        rows.append({
            "event_id": eid,
            "event_threshold": threshold,
            "label_origin_start": first["forecast_date"],
            "label_origin_end": last["forecast_date"],
            "event_onset": first["label_start"],
            # Confirmation is after the last positive label window ends.
            "event_confirmation": last["label_end"],
            "n_label_dates": len(ix),
            "peak_positive_fraction": float(
                x.iloc[ix]["positive_fraction"].max()),
        })
    return pd.DataFrame(rows)


def xlogy(a, p):
    a = np.asarray(a, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = a * np.log(p)
    return np.where(a == 0, 0.0, z)


def kupiec(x, n):
    x, n = np.asarray(x, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = x / n
    ll0 = xlogy(n - x, 1 - ALPHA) + xlogy(x, ALPHA)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    return -2 * (ll0 - ll1)


def christoffersen_independence(n00, n01, n10, n11):
    n00, n01, n10, n11 = [
        np.asarray(z, float) for z in (n00, n01, n10, n11)]
    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / np.maximum(n0, 1)
    pi11 = n11 / np.maximum(n1, 1)
    pi = (n01 + n11) / np.maximum(n0 + n1, 1)

    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = (
        xlogy(n00, 1 - pi01) + xlogy(n01, pi01)
        + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    )
    return -2 * (ll0 - ll1)


def load_backtest_scores(path):
    cols = [
        "ticker", "forecast_date",
        "outcome_next_day_breach", "model_name",
    ]
    check_columns(path, cols)
    q = pd.read_csv(path, usecols=cols, low_memory=False)
    q["forecast_date"] = pd.to_datetime(q["forecast_date"], errors="coerce")
    q["ticker"] = q["ticker"].astype(str).str.strip().str.upper()
    q["model"] = q["model_name"].astype(str).str.strip()
    q["b"] = pd.to_numeric(q["outcome_next_day_breach"], errors="coerce")
    q = q.dropna(subset=["forecast_date", "ticker", "model", "b"])
    q = q[q["model"].isin(MODELS)].sort_values(
        ["model", "ticker", "forecast_date"])

    parts = []
    for (model, ticker), g in q.groupby(["model", "ticker"], sort=False):
        s = pd.Series(g["b"].to_numpy(float))
        prev = s.shift(1).fillna(0)
        c250 = s.rolling(250, min_periods=250).sum().to_numpy()
        n00 = ((1-prev)*(1-s)).rolling(250, min_periods=250).sum().to_numpy()
        n01 = ((1-prev)*s).rolling(250, min_periods=250).sum().to_numpy()
        n10 = (prev*(1-s)).rolling(250, min_periods=250).sum().to_numpy()
        n11 = (prev*s).rolling(250, min_periods=250).sum().to_numpy()
        uc = kupiec(c250, np.full(len(g), 250.0))

        parts.append(pd.DataFrame({
            "forecast_date": g["forecast_date"].to_numpy(),
            "ticker": ticker,
            "model": model,
            "bt_count250": c250,
            "bt_count60": s.rolling(60, min_periods=60).sum().to_numpy(),
            "bt_count20": s.rolling(20, min_periods=20).sum().to_numpy(),
            "kupiec250": uc,
            "christoffersen250": uc + christoffersen_independence(
                n00, n01, n10, n11),
        }))
    if not parts:
        raise RuntimeError("No one-day breach rows found.")
    return pd.concat(parts, ignore_index=True)


def daily_scores(pred, bt):
    parts = []
    present = sorted(set(pred["spec"].unique()) & {"S2", "S3", "S4", "S5", "S6", "S7"})

    for model in MODELS:
        for spec in present:
            x = pred[(pred["model"] == model) & (pred["spec"] == spec)]
            d = x.groupby("forecast_date", as_index=False).agg(
                n_tickers=("ticker", "nunique"),
                n_scores=("p_cal", "count"),
                score=("p_cal", "mean"),
            )
            d = d[d["n_tickers"].eq(24) & d["n_scores"].eq(24)].copy()
            d["model"], d["signal"] = model, spec
            parts.append(d[["forecast_date", "model", "signal", "score"]])

        for score_name in BACKTESTS:
            x = bt[bt["model"] == model][
                ["forecast_date", "ticker", score_name]]
            d = x.groupby("forecast_date", as_index=False).agg(
                n_tickers=("ticker", "nunique"),
                n_scores=(score_name, "count"),
                score=(score_name, "mean"),
            )
            d = d[d["n_tickers"].eq(24) & d["n_scores"].eq(24)].copy()
            d["model"], d["signal"] = model, "BT_" + score_name
            parts.append(d[["forecast_date", "model", "signal", "score"]])

    out = pd.concat(parts, ignore_index=True)
    out["quarter"] = out["forecast_date"].dt.to_period("Q").astype(str)
    return out.sort_values(["model", "signal", "forecast_date"])


def causal_thresholds(scores, oos_dates, budget=BUDGET):
    if not (0.0 < budget < 1.0):
        raise ValueError("--budget must be between 0 and 1.")
    # Each quarterly threshold depends only on the preceding 252 daily scores.
    qlabels = pd.Series(oos_dates).dt.to_period("Q").astype(str).to_numpy()
    quarters = sorted(set(qlabels))
    threshold_rows = []

    for (model, signal), g in scores.groupby(["model", "signal"]):
        g = g.dropna(subset=["score"]).sort_values("forecast_date")
        for q in quarters:
            test_dates = oos_dates[qlabels == q]
            if not len(test_dates):
                continue
            start, end = test_dates.min(), test_dates.max()
            history = g[g["forecast_date"] < start].tail(CAL_DATES)
            if len(history) < CAL_DATES:
                continue

            cut = float(history["score"].quantile(1 - budget))
            test = g[g["forecast_date"].isin(test_dates)]
            if test.empty:
                continue

            threshold_rows.append({
                "model": model, "signal": signal, "quarter": q,
                "threshold": cut, "n_calibration_dates": len(history),
                "calibration_start": history["forecast_date"].min(),
                "calibration_end": history["forecast_date"].max(),
                "test_start": start, "test_end": end,
                "n_test_dates": len(test),
                "realized_test_alert_fraction": float((test["score"] >= cut).mean()),
            })

    thresholds = pd.DataFrame(threshold_rows)
    if thresholds.empty:
        raise RuntimeError("No quarterly system thresholds could be fitted.")

    scored = scores.merge(
        thresholds[["model", "signal", "quarter", "threshold"]],
        on=["model", "signal", "quarter"],
        how="inner",
        validate="many_to_one",
    )
    scored["alert"] = scored["score"] >= scored["threshold"]
    return thresholds, scored


def common_start(thresholds):
    required = []
    for model in MODELS:
        required.extend((model, s) for s in MONITORS)
        required.extend((model, "BT_" + s) for s in BACKTESTS)

    have = {
        (m, s): set(g["quarter"].astype(str))
        for (m, s), g in thresholds.groupby(["model", "signal"])
    }
    common = None
    for key in required:
        if key not in have:
            raise RuntimeError(f"Missing threshold series for {key}")
        common = have[key] if common is None else common & have[key]

    if not common:
        raise RuntimeError("No common evaluation quarter across required signals.")
    q = min(common, key=lambda z: pd.Period(z, freq="Q"))
    start = thresholds.loc[thresholds["quarter"].eq(q), "test_start"].min()
    return pd.Timestamp(start), q


def alert_episodes(daily, calendar, positions, gap):
    x = daily.sort_values("forecast_date").drop_duplicates("forecast_date").copy()
    x = x[x["alert"].fillna(False).astype(bool)].copy()
    if x.empty:
        return []

    x["pos"] = x["forecast_date"].map(positions)
    x = x.sort_values("pos").reset_index(drop=True)
    groups, cur = [], [0]
    for i in range(1, len(x)):
        if int(x.loc[i, "pos"]) - int(x.loc[i-1, "pos"]) - 1 <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)

    result = []
    for eid, ix in enumerate(groups, 1):
        b = x.iloc[ix]
        dates = [pd.Timestamp(z) for z in b["forecast_date"]]
        result.append({
            "episode_id": eid, "start": min(dates), "end": max(dates),
            "dates": dates, "alert_days": len(dates),
            "max_score": float(b["score"].max()),
        })
    return result


def pos_of(date, positions, calendar):
    if pd.isna(date):
        return -1
    date = pd.Timestamp(date)
    if date in positions:
        return positions[date]
    return min(int(calendar.searchsorted(date, side="left")), len(calendar)-1)


def match_episodes(events, episodes, window, calendar, positions):
    event_list = events.sort_values("event_onset").to_dict("records") if len(events) else []
    used_events, used_eps = set(), set()
    event_rows, ep_rows = [], []

    for ep in episodes:
        candidates = []
        for ev in event_list:
            if ev["event_id"] in used_events or pd.isna(ev["event_onset"]):
                continue
            onset = pos_of(ev["event_onset"], positions, calendar)
            valid = []
            for date in ep["dates"]:
                p = pos_of(date, positions, calendar)
                lead = onset - p
                if 1 <= lead <= window:
                    valid.append((p, date, lead))
            if valid:
                candidates.append((onset, ev, min(valid, key=lambda z: z[0])))

        if candidates:
            _, ev, first = min(candidates, key=lambda z: z[0])
            used_events.add(ev["event_id"])
            used_eps.add(ep["episode_id"])
            confirm = pos_of(ev["event_confirmation"], positions, calendar)
            event_rows.append({
                **ev, "hit": 1, "matched_episode_id": ep["episode_id"],
                "first_alert_date": first[1], "lead_days_to_label_start": first[2],
                "lead_days_to_label_end": confirm-first[0] if confirm >= 0 else np.nan,
            })
            ep_rows.append({
                **{k:v for k,v in ep.items() if k != "dates"},
                "matched_event_id": ev["event_id"], "false_alarm_episode": 0,
            })

    for ev in event_list:
        if ev["event_id"] not in used_events:
            event_rows.append({
                **ev, "hit": 0, "matched_episode_id": np.nan,
                "first_alert_date": pd.NaT,
                "lead_days_to_label_start": np.nan,
                "lead_days_to_label_end": np.nan,
            })

    for ep in episodes:
        if ep["episode_id"] not in used_eps:
            ep_rows.append({
                **{k:v for k,v in ep.items() if k != "dates"},
                "matched_event_id": np.nan, "false_alarm_episode": 1,
            })

    return pd.DataFrame(event_rows), pd.DataFrame(ep_rows)


def bootstrap_delta(m4, m6, dates, reps, block, rng):
    """
    Paired moving-block bootstrap over events ordered by onset.

    The estimand is the S6-minus-S4 event-hit-rate difference,
    conditional on the detected event set and existing alert matches.
    The dates argument is retained for call compatibility.
    """
    empty_result = {
        "delta_hit_rate_S6_minus_S4": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "bootstrap_reps_valid": 0,
        "bootstrap_block_events": np.nan,
    }

    if m4.empty or m6.empty:
        return empty_result

    a = m4[["event_id", "event_onset", "hit"]].rename(
        columns={"hit": "hit4"}
    )
    b = m6[["event_id", "event_onset", "hit"]].rename(
        columns={"hit": "hit6"}
    )

    pair = a.merge(
        b,
        on=["event_id", "event_onset"],
        validate="one_to_one",
    )

    if pair.empty:
        return empty_result

    pair = pair.sort_values(
        "event_onset", kind="stable"
    ).reset_index(drop=True)

    # Each element is the paired hit-rate contribution of one event.
    diff = (
        pair["hit6"].astype(float).to_numpy()
        - pair["hit4"].astype(float).to_numpy()
    )

    point = float(diff.mean())
    n = len(diff)

    # Avoid a block spanning the entire sample when multiple events exist.
    block = max(1, min(int(block), max(1, n - 1)))
    nblocks = int(math.ceil(n / block))

    draws = []
    for _ in range(reps):
        # Circular moving blocks preserve the ordering within each block.
        starts = rng.integers(0, n, size=nblocks)
        sampled = np.concatenate([
            (s + np.arange(block)) % n for s in starts
        ])[:n]

        draws.append(float(diff[sampled].mean()))

    lo, hi = (
        np.percentile(draws, [2.5, 97.5])
        if draws else (np.nan, np.nan)
    )

    return {
        "delta_hit_rate_S6_minus_S4": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_reps_valid": len(draws),
        "bootstrap_block_events": block,
    }


def main():
    a = args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    pred = load_predictions(Path(a.pred))
    panel, calendar = load_panel(Path(a.panel))
    outcomes = make_events_daily(pred, panel)
    bt = load_backtest_scores(Path(a.quar))
    scores = daily_scores(pred, bt)

    oos_dates = pd.DatetimeIndex(sorted(pred["forecast_date"].unique()))
    thresholds, scored = causal_thresholds(scores, oos_dates, a.budget)
    eval_start, eval_q = common_start(thresholds)
    eval_dates = pd.DatetimeIndex([d for d in oos_dates if d >= eval_start])

    print(f"Prediction rows: {len(pred):,}")
    print(f"OOS dates: {len(oos_dates):,}")
    print(f"Common causal evaluation quarter: {eval_q}")
    print(f"Evaluation start: {eval_start.date()}")
    print(f"Warm-up dates excluded: {(oos_dates < eval_start).sum()}")
    print(f"Evaluation dates: {len(eval_dates):,}")
    print(f"Threshold: prior {CAL_DATES}-date score history; q={1-a.budget:.0%} per signal")

    thresholds.to_csv(out / "RS05_causal_system_quarterly_thresholds.csv", index=False)
    scored = scored[scored["forecast_date"] >= eval_start].copy()
    scored.to_csv(out / "RS05_causal_system_daily_scores_alerts.csv", index=False)

    positions = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    summary, matches_out, alerts_out, boot_out = [], [], [], []

    for model in MODELS:
        for eth in EVENT_THRESHOLDS:
            events = extract_events(outcomes[outcomes["model"] == model], eth, a.event_gap)
            if not events.empty:
                events = events[
                    events["event_onset"].notna()
                    & (events["event_onset"] >= eval_start)
                ].copy().reset_index(drop=True)
                events["event_id"] = np.arange(1, len(events)+1)

            model_signals = sorted(scored.loc[scored["model"] == model, "signal"].unique())
            for signal in model_signals:
                daily = scored[
                    (scored["model"] == model) & (scored["signal"] == signal)
                ].copy()
                episodes = alert_episodes(daily, calendar, positions, a.episode_gap)

                for win in MATCH_WINDOWS:
                    me, ma = match_episodes(events, episodes, win, calendar, positions)
                    false_n = int(ma["false_alarm_episode"].sum()) if not ma.empty else 0
                    hits = int(me["hit"].sum()) if not me.empty else 0
                    lead = (
                        pd.to_numeric(me.loc[me["hit"] == 1, "lead_days_to_label_start"], errors="coerce").dropna()
                        if not me.empty else pd.Series(dtype=float)
                    )
                    lead_end = (
                        pd.to_numeric(me.loc[me["hit"] == 1, "lead_days_to_label_end"], errors="coerce").dropna()
                        if not me.empty else pd.Series(dtype=float)
                    )
                    years = len(eval_dates) / 252
                    alert_days = int(daily["alert"].sum())
                    summary.append({
                        "model": model, "signal": signal,
                        "threshold_method": f"quarterly q{1-a.budget:.0%} of previous 252 daily scores",
                        "event_threshold": eth, "match_window": win,
                        "n_events": len(events), "n_events_hit": hits,
                        "event_hit_rate": hits / len(events) if len(events) else np.nan,
                        "median_lead_to_label_start": float(lead.median()) if len(lead) else np.nan,
                        "median_lead_to_label_end": float(lead_end.median()) if len(lead_end) else np.nan,
                        "n_alert_episodes": len(ma),
                        "n_false_alarm_episodes": false_n,
                        "false_alarm_episodes_per_year": false_n / years if years else np.nan,
                        "alert_days_per_year": alert_days / years if years else np.nan,
                        "realized_system_alert_fraction": alert_days / len(eval_dates),
                        "evaluation_start": eval_start,
                        "evaluation_dates": len(eval_dates),
                    })
                    if not me.empty:
                        me = me.copy()
                        me["model"], me["signal"] = model, signal
                        me["event_threshold"], me["match_window"] = eth, win
                        matches_out.append(me)
                    if not ma.empty:
                        ma = ma.copy()
                        ma["model"], ma["signal"] = model, signal
                        ma["event_threshold"], ma["match_window"] = eth, win
                        alerts_out.append(ma)

            for win in MATCH_WINDOWS:
                paired = {}
                for signal in ["S4", "S6"]:
                    daily = scored[(scored["model"] == model) & (scored["signal"] == signal)]
                    eps = alert_episodes(daily, calendar, positions, a.episode_gap)
                    paired[signal], _ = match_episodes(events, eps, win, calendar, positions)
                delta = bootstrap_delta(
                    paired["S4"], paired["S6"], eval_dates,
                    a.boot, a.block, rng,
                )
                boot_out.append({
                    "model": model, "event_threshold": eth, "match_window": win,
                    "n_events": len(events),
                    "S4_hit_rate": float(paired["S4"]["hit"].mean()) if not paired["S4"].empty else np.nan,
                    "S6_hit_rate": float(paired["S6"]["hit"].mean()) if not paired["S6"].empty else np.nan,
                    **delta,
                })

    pd.DataFrame(summary).to_csv(
        out / "RS05_causal_system_alert_episode_summary.csv", index=False)
    pd.concat(matches_out, ignore_index=True).to_csv(
        out / "RS05_causal_system_event_matches.csv", index=False)
    pd.concat(alerts_out, ignore_index=True).to_csv(
        out / "RS05_causal_system_alert_episodes.csv", index=False)
    pd.DataFrame(boot_out).to_csv(
        out / "RS05_causal_system_S6_minus_S4_bootstrap.csv", index=False)

    print("\nSaved causal system-level results to:", out)
    print("Review the threshold table's calibration and test alert fractions.")
    print("The event lead is to label_start_date_10d, not a directly observed daily-loss onset.")


if __name__ == "__main__":
    main()
