
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
BACKTESTS = [
    "bt_count250", "bt_count60", "bt_count20",
    "kupiec250", "christoffersen250",
]
EVENT_THRESHOLDS = [0.25, 0.50, 0.75]
ALERT_SHARES = [0.10, 0.25, 0.50]
MATCH_WINDOWS = [10, 20, 30]
SPECS = ["S1", "S2", "S3", "S4", "S6"]
N_TICKERS = 24
H = 10
ALPHA = 0.05
CAL_LEN = 252
MIN_FIT_DATES = 300
START_DATE = pd.Timestamp("2020-01-01")


def parse_args():
    p = argparse.ArgumentParser(
        description="RS05 T1 event-level alert evaluation and sensitivity analysis."
    )
    p.add_argument(
        "--pred",
        default=str(RS05 / "RS05_oos_predictions_H10_final.csv.gz"),
    )
    p.add_argument(
        "--panel",
        default=str(RS / 'RS04b_causal_reliability_panel.csv.gz'),
    )
    p.add_argument(
        "--quar",
        default=str(RS / 'RS04_quarantined_outcomes.csv.gz'),
    )
    p.add_argument(
        "--out",
        default=str(RS05 / "RS05_alert_episode_eval"),
    )
    p.add_argument("--boot", type=int, default=500)
    p.add_argument("--block", type=int, default=20)
    p.add_argument("--episode-gap", type=int, default=3)
    p.add_argument("--event-gap", type=int, default=5)
    p.add_argument("--seed", type=int, default=20261025)
    return p.parse_args()


def check_columns(path, required):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    cols = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(required) - cols)
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")


def load_predictions(path):
    required = [
        "forecast_date", "ticker", "model", "target", "H",
        "spec", "algo", "y", "p_cal", "thr10",
    ]
    check_columns(path, required)
    df = pd.read_csv(path, usecols=required, compression="infer", low_memory=False)

    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    for c in ["ticker", "model", "target", "spec", "algo"]:
        df[c] = df[c].astype(str).str.strip()
    df["ticker"] = df["ticker"].str.upper()
    df["target"] = df["target"].str.upper()
    df["spec"] = df["spec"].str.upper()
    df["algo"] = df["algo"].str.lower()

    df["H"] = pd.to_numeric(df["H"], errors="coerce")
    for c in ["y", "p_cal", "thr10"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[
        (df["target"] == "T1")
        & (df["H"] == H)
        & (df["algo"] == "logit")
        & (df["forecast_date"] >= START_DATE)
        & df["model"].isin(MODELS)
    ].dropna(subset=["forecast_date", "ticker", "model", "spec", "y"]).copy()

    if df.empty:
        raise RuntimeError("No T1/H10/logit OOS predictions found.")

    # A realized target must not vary across specifications.
    check = df.groupby(
        ["forecast_date", "ticker", "model"]
    )["y"].nunique()
    if (check > 1).any():
        raise RuntimeError("T1 labels differ across specifications.")

    return df.sort_values(
        ["forecast_date", "model", "ticker", "spec"]
    ).reset_index(drop=True)


def load_panel(path):
    required = [
        "forecast_date", "ticker", "model_name",
        "label_start_date_10d", "label_end_date_10d",
    ]
    check_columns(path, required)
    df = pd.read_csv(path, usecols=required, low_memory=False)

    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    df["label_start_date_10d"] = pd.to_datetime(
        df["label_start_date_10d"], errors="coerce"
    )
    df["label_end_date_10d"] = pd.to_datetime(
        df["label_end_date_10d"], errors="coerce"
    )
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["model"] = df["model_name"].astype(str).str.strip()

    key = ["forecast_date", "ticker", "model"]
    if df.duplicated(key).any():
        raise RuntimeError("Duplicate keys in the RS04b panel.")

    date_meta = (
        df.groupby("forecast_date", as_index=False)
        .agg(
            label_start_date_10d=("label_start_date_10d", "min"),
            label_end_date_10d=("label_end_date_10d", "max"),
        )
        .sort_values("forecast_date")
    )
    return df, date_meta


def build_outcome_daily(pred, panel):
    # Use one copy of each realized target, preferring S6 if available.
    priority = {"S6": 0, "S4": 1, "S3": 2, "S2": 3, "S1": 4}
    x = pred.copy()
    x["_priority"] = x["spec"].map(priority).fillna(99)
    x = x.sort_values("_priority").drop_duplicates(
        ["forecast_date", "ticker", "model"]
    )

    x = x[["forecast_date", "ticker", "model", "y"]].merge(
        panel[
            [
                "forecast_date", "ticker", "model",
                "label_start_date_10d", "label_end_date_10d",
            ]
        ],
        on=["forecast_date", "ticker", "model"],
        how="left",
        validate="one_to_one",
    )

    d = (
        x.groupby(["model", "forecast_date"], as_index=False)
        .agg(
            n_tickers=("ticker", "nunique"),
            n_valid=("y", "count"),
            positive_fraction=("y", "mean"),
            label_start_date_10d=("label_start_date_10d", "min"),
            label_end_date_10d=("label_end_date_10d", "max"),
        )
        .sort_values(["model", "forecast_date"])
    )
    d["complete_24"] = (
        (d["n_tickers"] == N_TICKERS) & (d["n_valid"] == N_TICKERS)
    )
    return d


def extract_events(daily, threshold, max_gap):
    x = daily[
        daily["complete_24"] & (daily["forecast_date"] >= START_DATE)
    ].sort_values("forecast_date").reset_index(drop=True)

    if x.empty:
        return pd.DataFrame()

    active = x["positive_fraction"].to_numpy() >= threshold
    pos = np.flatnonzero(active)
    if len(pos) == 0:
        return pd.DataFrame()

    groups, current = [], [int(pos[0])]
    for k in pos[1:]:
        k = int(k)
        # Merge event periods separated by no more than max_gap trading dates.
        if k - current[-1] - 1 <= max_gap:
            current.append(k)
        else:
            groups.append(current)
            current = [k]
    groups.append(current)

    rows = []
    for event_id, ix in enumerate(groups, start=1):
        first, last = x.iloc[ix[0]], x.iloc[ix[-1]]
        rows.append({
            "event_id": event_id,
            "event_threshold": threshold,
            "label_origin_start": first["forecast_date"],
            "label_origin_end": last["forecast_date"],
            "event_onset_date": first["label_start_date_10d"],
            "event_confirmation_date": first["label_end_date_10d"],
            "event_n_label_dates": len(ix),
            "max_positive_fraction": float(
                x.iloc[ix]["positive_fraction"].max()
            ),
            "mean_positive_fraction": float(
                x.iloc[ix]["positive_fraction"].mean()
            ),
        })
    return pd.DataFrame(rows)


def xlogy(a, p):
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = a * np.log(p)
    return np.where(a == 0, 0.0, v)


def kupiec_lr(x, n, p=ALPHA):
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    pi = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    return -2 * (ll0 - ll1)


def independence_lr(n00, n01, n10, n11):
    n00, n01, n10, n11 = [
        np.asarray(v, dtype=float) for v in (n00, n01, n10, n11)
    ]
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


def build_backtest_scores(path):
    required = [
        "ticker", "forecast_date",
        "outcome_next_day_breach", "model_name",
    ]
    check_columns(path, required)
    q = pd.read_csv(path, usecols=required, low_memory=False)
    q["forecast_date"] = pd.to_datetime(q["forecast_date"], errors="coerce")
    q["ticker"] = q["ticker"].astype(str).str.strip().str.upper()
    q["model"] = q["model_name"].astype(str).str.strip()
    q["b"] = pd.to_numeric(q["outcome_next_day_breach"], errors="coerce")
    q = q.dropna(subset=["forecast_date", "ticker", "model", "b"])
    q = q[q["model"].isin(MODELS)].sort_values(
        ["model", "ticker", "forecast_date"]
    )

    output = []
    for (model, ticker), g in q.groupby(["model", "ticker"], sort=False):
        s = pd.Series(g["b"].to_numpy(float))
        prev = s.shift(1).fillna(0.0)

        c250 = s.rolling(250, min_periods=250).sum().to_numpy()
        n00 = ((1 - prev) * (1 - s)).rolling(250, min_periods=250).sum().to_numpy()
        n01 = ((1 - prev) * s).rolling(250, min_periods=250).sum().to_numpy()
        n10 = (prev * (1 - s)).rolling(250, min_periods=250).sum().to_numpy()
        n11 = (prev * s).rolling(250, min_periods=250).sum().to_numpy()
        uc = kupiec_lr(c250, np.full(len(g), 250.0))

        output.append(pd.DataFrame({
            "forecast_date": g["forecast_date"].to_numpy(),
            "ticker": ticker,
            "model": model,
            "bt_count250": c250,
            "bt_count60": s.rolling(60, min_periods=60).sum().to_numpy(),
            "bt_count20": s.rolling(20, min_periods=20).sum().to_numpy(),
            "kupiec250": uc,
            "christoffersen250": uc + independence_lr(n00, n01, n10, n11),
        }))

    if not output:
        raise RuntimeError("No usable quarantined one-day breach rows.")
    return pd.concat(output, ignore_index=True)


def reconstruct_folds(pred_dates, date_meta):
    dates = pd.DatetimeIndex(
        pd.to_datetime(pred_dates).dropna().unique()
    ).sort_values()
    meta = date_meta.dropna(
        subset=["forecast_date", "label_end_date_10d"]
    ).drop_duplicates("forecast_date")

    folds = {}
    quarters = pd.PeriodIndex(dates, freq="Q")
    for quarter in sorted(quarters.unique()):
        test_dates = dates[quarters == quarter]
        if len(test_dates) < 10:
            continue

        first_test = test_dates.min()
        eligible = meta.loc[
            meta["label_end_date_10d"] < first_test, "forecast_date"
        ].drop_duplicates().sort_values()

        if len(eligible) < CAL_LEN + MIN_FIT_DATES:
            continue
        folds[str(quarter)] = {
            "test_dates": set(test_dates),
            "cal_dates": set(eligible.iloc[-CAL_LEN:]),
        }

    if not folds:
        raise RuntimeError("Unable to reconstruct quarterly calibration folds.")
    return folds


def build_backtest_alert_rows(scores, folds):
    pieces = []
    for fold_name, fold in folds.items():
        cal = scores[scores["forecast_date"].isin(fold["cal_dates"])]
        test = scores[scores["forecast_date"].isin(fold["test_dates"])]

        for model in MODELS:
            c = cal[cal["model"] == model]
            t = test[test["model"] == model]
            for score in BACKTESTS:
                values = c[score].dropna()
                if values.empty:
                    continue

                # Threshold is learned only from this fold's prior mature dates.
                threshold = float(values.quantile(0.90))
                z = t[["forecast_date", "ticker", "model", score]].copy()
                z["alert"] = z[score].notna() & (z[score] >= threshold)
                z["score_value"] = z[score]
                z["spec"] = "BT_" + score
                z["fold"] = fold_name
                pieces.append(z[[
                    "forecast_date", "ticker", "model", "spec",
                    "alert", "score_value", "fold",
                ]])

    if not pieces:
        raise RuntimeError("Could not create backtest alert rows.")
    return pd.concat(pieces, ignore_index=True)


def daily_model_signal(pred, model, spec):
    x = pred[(pred["model"] == model) & (pred["spec"] == spec)].copy()
    x = x.dropna(subset=["p_cal", "thr10"])
    if x.empty:
        return pd.DataFrame()

    x["alert"] = x["p_cal"] >= x["thr10"]
    d = x.groupby("forecast_date", as_index=False).agg(
        n_tickers=("ticker", "nunique"),
        alert_fraction=("alert", "mean"),
        mean_score=("p_cal", "mean"),
    )
    d["complete_24"] = d["n_tickers"].eq(N_TICKERS)
    d.loc[~d["complete_24"], "alert_fraction"] = np.nan
    return d.sort_values("forecast_date").reset_index(drop=True)


def daily_backtest_signal(bt_rows, model, spec):
    x = bt_rows[(bt_rows["model"] == model) & (bt_rows["spec"] == spec)].copy()
    x = x.dropna(subset=["score_value"])
    if x.empty:
        return pd.DataFrame()

    d = x.groupby("forecast_date", as_index=False).agg(
        n_tickers=("ticker", "nunique"),
        alert_fraction=("alert", "mean"),
        mean_score=("score_value", "mean"),
    )
    d["complete_24"] = d["n_tickers"].eq(N_TICKERS)
    d.loc[~d["complete_24"], "alert_fraction"] = np.nan
    return d.sort_values("forecast_date").reset_index(drop=True)


def equal_burden_signal(daily, budget=0.10):
    """Retrospective comparator: exactly the top 10% of OOS dates by mean risk."""
    d = daily[
        daily["complete_24"] & daily["mean_score"].notna()
    ].copy()
    if d.empty:
        return d

    k = max(1, int(math.ceil(budget * len(d))))
    chosen = d.sort_values(
        ["mean_score", "forecast_date"], ascending=[False, True]
    ).head(k)

    d["alert_fraction"] = 0.0
    d.loc[chosen.index, "alert_fraction"] = 1.0
    return d.sort_values("forecast_date").reset_index(drop=True)


def make_alert_episodes(daily, cross_section, episode_gap, date_positions):
    x = daily[
        daily["complete_24"] & daily["alert_fraction"].notna()
    ].copy()
    x = x.sort_values("forecast_date").reset_index(drop=True)
    if x.empty:
        return []

    x["is_alert"] = x["alert_fraction"] >= cross_section
    x = x[x["is_alert"]].copy()
    if x.empty:
        return []

    x["position"] = x["forecast_date"].map(date_positions)
    x = x.sort_values("position").reset_index(drop=True)

    groups, current = [], [0]
    for i in range(1, len(x)):
        gap = int(x.loc[i, "position"]) - int(x.loc[i - 1, "position"]) - 1
        if gap <= episode_gap:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    result = []
    for eid, indices in enumerate(groups, start=1):
        b = x.iloc[indices]
        alert_dates = [pd.Timestamp(d) for d in b["forecast_date"]]
        result.append({
            "alert_episode_id": eid,
            "alert_start_date": min(alert_dates),
            "alert_end_date": max(alert_dates),
            "alert_dates": alert_dates,
            "n_alert_days": len(alert_dates),
            "max_alert_fraction": float(b["alert_fraction"].max()),
            "max_mean_score": (
                float(b["mean_score"].max())
                if b["mean_score"].notna().any() else np.nan
            ),
        })
    return result


def position(date, date_positions, calendar):
    if pd.isna(date):
        return -1
    date = pd.Timestamp(date)
    if date in date_positions:
        return date_positions[date]
    # Use the next observed session if source dates differ by one day.
    return min(int(calendar.searchsorted(date, side="left")), len(calendar) - 1)


def match_episodes(events, alert_episodes, match_window, calendar, date_positions):
    """Greedy chronological matching; each event and alert episode can hit once."""
    event_list = (
        events.sort_values("event_onset_date").to_dict("records")
        if not events.empty else []
    )
    used_event_ids, matched_episode_ids = set(), set()
    event_rows, episode_rows = [], []

    for ep in alert_episodes:
        candidates = []
        for ev in event_list:
            if ev["event_id"] in used_event_ids:
                continue
            if pd.isna(ev["event_onset_date"]):
                continue

            onset = position(ev["event_onset_date"], date_positions, calendar)
            valid_alert_dates = []
            for alert_date in ep["alert_dates"]:
                alert_pos = position(alert_date, date_positions, calendar)
                lead = onset - alert_pos
                if 1 <= lead <= match_window:
                    valid_alert_dates.append((alert_pos, alert_date, lead))

            if valid_alert_dates:
                first = min(valid_alert_dates, key=lambda z: z[0])
                candidates.append((onset, ev, first))

        if candidates:
            _, ev, first = min(candidates, key=lambda z: z[0])
            used_event_ids.add(ev["event_id"])
            matched_episode_ids.add(ep["alert_episode_id"])

            confirmation = position(
                ev.get("event_confirmation_date"), date_positions, calendar
            )
            event_rows.append({
                **ev,
                "hit": 1,
                "matched_alert_episode_id": ep["alert_episode_id"],
                "first_alert_date_in_window": first[1],
                "lead_trading_days": int(first[2]),
                "lead_to_confirmation_date": (
                    confirmation - first[0] if confirmation >= 0 else np.nan
                ),
            })
            episode_rows.append({
                **{k: v for k, v in ep.items() if k != "alert_dates"},
                "matched_event_id": ev["event_id"],
                "false_alarm_episode": 0,
            })

    for ev in event_list:
        if ev["event_id"] not in used_event_ids:
            event_rows.append({
                **ev,
                "hit": 0,
                "matched_alert_episode_id": np.nan,
                "first_alert_date_in_window": pd.NaT,
                "lead_trading_days": np.nan,
                "lead_to_confirmation_date": np.nan,
            })

    for ep in alert_episodes:
        if ep["alert_episode_id"] not in matched_episode_ids:
            episode_rows.append({
                **{k: v for k, v in ep.items() if k != "alert_dates"},
                "matched_event_id": np.nan,
                "false_alarm_episode": 1,
            })

    return pd.DataFrame(event_rows), pd.DataFrame(episode_rows)


def summarize(model, signal, method, event_threshold, alert_share, match_window,
              events, event_matches, episode_matches, daily, n_oos_dates,
              event_gap, episode_gap):
    n_events = len(events)
    hits = int(event_matches["hit"].sum()) if not event_matches.empty else 0
    leads = (
        pd.to_numeric(
            event_matches.loc[event_matches["hit"] == 1, "lead_trading_days"],
            errors="coerce",
        ).dropna()
        if not event_matches.empty else pd.Series(dtype=float)
    )
    confirmation_leads = (
        pd.to_numeric(
            event_matches.loc[event_matches["hit"] == 1, "lead_to_confirmation_date"],
            errors="coerce",
        ).dropna()
        if not event_matches.empty else pd.Series(dtype=float)
    )

    false_alarms = (
        int(episode_matches["false_alarm_episode"].sum())
        if not episode_matches.empty else 0
    )
    alert_dates = 0
    if not daily.empty:
        valid = daily[daily["complete_24"] & daily["alert_fraction"].notna()]
        alert_dates = int((valid["alert_fraction"] >= alert_share).sum())

    years = max(n_oos_dates / 252.0, 1e-9)
    return {
        "model": model,
        "signal": signal,
        "signal_method": method,
        "event_threshold": event_threshold,
        "cross_section_alert_rule": alert_share,
        "match_window_trading_days": match_window,
        "event_merge_gap_non_event_dates": event_gap,
        "alert_episode_merge_gap_non_alert_dates": episode_gap,
        "n_events": n_events,
        "n_events_hit": hits,
        "event_hit_rate": hits / n_events if n_events else np.nan,
        "median_lead_trading_days_to_label_start": float(leads.median()) if len(leads) else np.nan,
        "mean_lead_trading_days_to_label_start": float(leads.mean()) if len(leads) else np.nan,
        "median_lead_to_label_end": float(confirmation_leads.median()) if len(confirmation_leads) else np.nan,
        "n_alert_episodes": len(episode_matches),
        "n_false_alarm_episodes": false_alarms,
        "false_alarm_episodes_per_year": false_alarms / years,
        "alert_days_per_year": alert_dates / years,
        "alert_episodes_per_year": len(episode_matches) / years,
        "oos_years_at_252_sessions": years,
    }


def paired_block_bootstrap(event4, event6, oos_dates, reps, block, rng):
    if event4.empty or event6.empty:
        return {"delta_hit_rate_S6_minus_S4": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    a = event4[["event_id", "event_onset_date", "hit"]].rename(
        columns={"hit": "hit_S4"}
    )
    b = event6[["event_id", "event_onset_date", "hit"]].rename(
        columns={"hit": "hit_S6"}
    )
    pair = a.merge(
        b, on=["event_id", "event_onset_date"],
        how="inner", validate="one_to_one",
    )
    if pair.empty:
        return {"delta_hit_rate_S6_minus_S4": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    point = float(pair["hit_S6"].mean() - pair["hit_S4"].mean())
    n_dates = len(oos_dates)
    block = max(1, min(block, n_dates))
    n_blocks = int(math.ceil(n_dates / block))
    event_pos = np.array([
        int(np.clip(oos_dates.searchsorted(pd.Timestamp(d)), 0, n_dates - 1))
        for d in pair["event_onset_date"]
    ])

    draws = []
    for _ in range(reps):
        starts = rng.integers(0, n_dates, size=n_blocks)
        sampled = np.concatenate([
            (s + np.arange(block)) % n_dates for s in starts
        ])[:n_dates]
        weights_by_date = np.bincount(sampled, minlength=n_dates)
        weights = weights_by_date[event_pos].astype(float)
        if weights.sum() == 0:
            continue
        h4 = np.average(pair["hit_S4"].to_numpy(float), weights=weights)
        h6 = np.average(pair["hit_S6"].to_numpy(float), weights=weights)
        draws.append(h6 - h4)

    lo, hi = (
        np.percentile(draws, [2.5, 97.5])
        if draws else [np.nan, np.nan]
    )
    return {
        "delta_hit_rate_S6_minus_S4": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_reps_valid": len(draws),
        "bootstrap_block_trading_dates": block,
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    pred = load_predictions(Path(args.pred))
    panel, date_meta = load_panel(Path(args.panel))
    outcomes = build_outcome_daily(pred, panel)

    available_specs = sorted(pred["spec"].unique())
    print(f"Prediction rows: {len(pred):,}")
    print(f"OOS dates: {pred['forecast_date'].nunique():,}")
    print(f"Specifications found: {available_specs}")

    missing = [s for s in ["S3", "S4", "S6"] if s not in available_specs]
    if missing:
        raise RuntimeError(f"Required headline specifications are missing: {missing}")
    optional_missing = [s for s in ["S1", "S2"] if s not in available_specs]
    if optional_missing:
        print(f"Note: S1/S2 are absent and will not be inferred: {optional_missing}")

    folds = reconstruct_folds(pred["forecast_date"], date_meta)
    bt_scores = build_backtest_scores(Path(args.quar))
    bt_rows = build_backtest_alert_rows(bt_scores, folds)
    print(f"Quarterly folds reconstructed: {len(folds)}")

    calendar = pd.DatetimeIndex(sorted(
        set(date_meta["forecast_date"].dropna())
        | set(date_meta["label_start_date_10d"].dropna())
    ))
    date_positions = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    oos_dates = pd.DatetimeIndex(sorted(pred["forecast_date"].dropna().unique()))

    # Signal cache: all native monitor specs available in the selected file.
    signal_cache = {}
    for model in MODELS:
        for spec in available_specs:
            d = daily_model_signal(pred, model, spec)
            if not d.empty:
                signal_cache[(model, spec, "native_thr10")] = d

        for bt in BACKTESTS:
            spec = "BT_" + bt
            d = daily_backtest_signal(bt_rows, model, spec)
            if not d.empty:
                signal_cache[(model, spec, "calibrated_backtest")] = d

        # Equal realized alert-days: useful as a retrospective ranking check only.
        for spec in ["S3", "S4", "S6"]:
            native = signal_cache.get((model, spec, "native_thr10"))
            if native is not None:
                d = equal_burden_signal(native, budget=0.10)
                if not d.empty:
                    signal_cache[(model, spec, "posthoc_top10pct_dates")] = d

    summary_rows, event_rows, alert_rows, bootstrap_rows = [], [], [], []

    for model in MODELS:
        for event_threshold in EVENT_THRESHOLDS:
            events = extract_events(
                outcomes[outcomes["model"] == model],
                event_threshold,
                args.event_gap,
            )
            if not events.empty:
                events["model"] = model

            for (signal_model, signal, method), daily in signal_cache.items():
                if signal_model != model:
                    continue

                cross_grid = (
                    [0.25] if method == "posthoc_top10pct_dates"
                    else ALERT_SHARES
                )
                for alert_share in cross_grid:
                    episodes = make_alert_episodes(
                        daily, alert_share, args.episode_gap, date_positions
                    )
                    for window in MATCH_WINDOWS:
                        matched_events, matched_alerts = match_episodes(
                            events, episodes, window, calendar, date_positions
                        )
                        summary_rows.append(summarize(
                            model, signal, method, event_threshold, alert_share,
                            window, events, matched_events, matched_alerts, daily,
                            len(oos_dates), args.event_gap, args.episode_gap,
                        ))

                        if not matched_events.empty:
                            z = matched_events.copy()
                            z["model"] = model
                            z["signal"] = signal
                            z["signal_method"] = method
                            z["event_threshold"] = event_threshold
                            z["cross_section_alert_rule"] = alert_share
                            z["match_window_trading_days"] = window
                            event_rows.append(z)

                        if not matched_alerts.empty:
                            z = matched_alerts.copy()
                            z["model"] = model
                            z["signal"] = signal
                            z["signal_method"] = method
                            z["event_threshold"] = event_threshold
                            z["cross_section_alert_rule"] = alert_share
                            z["match_window_trading_days"] = window
                            alert_rows.append(z)

            # Paired S6 - S4 bootstrap for every event/alert/window sensitivity.
            for alert_share in ALERT_SHARES:
                for window in MATCH_WINDOWS:
                    pair = {}
                    for spec in ["S4", "S6"]:
                        daily = signal_cache.get((model, spec, "native_thr10"))
                        if daily is None:
                            pair[spec] = pd.DataFrame()
                            continue
                        eps = make_alert_episodes(
                            daily, alert_share, args.episode_gap, date_positions
                        )
                        pair[spec], _ = match_episodes(
                            events, eps, window, calendar, date_positions
                        )

                    boot = paired_block_bootstrap(
                        pair["S4"], pair["S6"], oos_dates,
                        args.boot, args.block, rng,
                    )
                    bootstrap_rows.append({
                        "model": model,
                        "event_threshold": event_threshold,
                        "cross_section_alert_rule": alert_share,
                        "match_window_trading_days": window,
                        "n_events": len(events),
                        "S4_hit_rate": (
                            float(pair["S4"]["hit"].mean())
                            if not pair["S4"].empty else np.nan
                        ),
                        "S6_hit_rate": (
                            float(pair["S6"]["hit"].mean())
                            if not pair["S6"].empty else np.nan
                        ),
                        **boot,
                    })

    summary = pd.DataFrame(summary_rows)
    event_details = (
        pd.concat(event_rows, ignore_index=True)
        if event_rows else pd.DataFrame()
    )
    alert_details = (
        pd.concat(alert_rows, ignore_index=True)
        if alert_rows else pd.DataFrame()
    )
    boot_df = pd.DataFrame(bootstrap_rows)

    paths = {
        "summary": out / "RS05_alert_episode_summary.csv",
        "events": out / "RS05_alert_episode_event_matches.csv",
        "alerts": out / "RS05_alert_episode_alert_runs.csv",
        "bootstrap": out / "RS05_alert_episode_S6_minus_S4_bootstrap.csv",
    }
    summary.to_csv(paths["summary"], index=False)
    event_details.to_csv(paths["events"], index=False)
    alert_details.to_csv(paths["alerts"], index=False)
    boot_df.to_csv(paths["bootstrap"], index=False)

    print("\nSaved outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("\nFull sensitivity grid:")
    print(f"  Event thresholds: {EVENT_THRESHOLDS}")
    print(f"  Cross-sectional alert rules: {ALERT_SHARES}")
    print(f"  Matching windows (trading dates): {MATCH_WINDOWS}")
    print(f"  Event merge gap: {args.event_gap}; alert-episode gap: {args.episode_gap}")
    print(f"  Bootstrap: {args.boot} replicates, block={args.block} trading dates")

    print("\nInterpretation notes:")
    print("  1. Lead is measured to label_start_date_10d, the first realized")
    print("     observation in a positive T1 window. label_end_date_10d gives")
    print("     the full-window confirmation-date sensitivity.")
    print("  2. An unmatched alert episode is a false alarm; each alert episode")
    print("     and each event can be matched at most once.")
    print("  3. Equal-burden top-10%-of-OOS-dates rows are post-hoc and not deployable.")
    print("  4. S1/S2 are evaluated only if they actually exist in the input predictions.")
    print("  5. These event-level metrics are descriptive unless uncertainty is narrow.")


if __name__ == "__main__":
    main()
