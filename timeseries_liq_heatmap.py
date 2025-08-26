# SOL-PERP liquidation heatmap (merged CSV, total only) + spot price overlay (1-minute default)
# Input CSV: data-sol_snapshots/sol_positions_log.csv
# Columns: liq_price,size_sol,t_bin   (t_bin like "2025-08-22T10:09:00Z")

import os, argparse, math, json, time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from urllib.request import urlopen, Request

# ---------------- IO ----------------

def load_merged_csv(path: str, t_start: datetime, t_end: datetime) -> pd.DataFrame:
    if not os.path.exists(path):
        raise SystemExit(f"Missing merged CSV: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise SystemExit(f"Failed to read {path}: {e}")

    need = {"liq_price", "size_sol", "t_bin"}
    if not need.issubset(df.columns):
        raise SystemExit("CSV must contain columns: liq_price, size_sol, t_bin")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["liq_price", "size_sol", "t_bin"])
    df["t_bin"] = pd.to_datetime(df["t_bin"], utc=True).dt.tz_convert(None)
    df["liq_price"] = pd.to_numeric(df["liq_price"], errors="coerce")
    df["size_sol"] = pd.to_numeric(df["size_sol"], errors="coerce")
    df = df.dropna(subset=["liq_price", "size_sol"])
    df = df[(df["t_bin"] >= t_start) & (df["t_bin"] <= t_end)]
    if df.empty:
        raise SystemExit("No data rows in requested time window.")
    return df

# ------------- External spot helpers -------------

def _binance_interval_for(freq_min: int) -> str:
    if freq_min <= 1:
        return "1m"
    return {3:"3m", 5:"5m", 15:"15m", 30:"30m", 60:"1h"}.get(freq_min, "1m")

def _fetch_binance_page(start_ms: int, end_ms: int, interval: str, limit: int = 1000):
    url = (f"https://api.binance.com/api/v3/klines?symbol=SOLUSDT"
           f"&interval={interval}&startTime={start_ms}&endTime={end_ms}&limit={limit}")
    with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0"}), timeout=10) as r:
        data = json.loads(r.read().decode())
    return data

def fetch_binance_series(t_start: datetime, t_end: datetime, freq_min: int) -> pd.Series:
    interval = _binance_interval_for(freq_min)
    interval_ms = freq_min * 60_000
    start_ms = int(t_start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(t_end.replace(tzinfo=timezone.utc).timestamp() * 1000)

    rows_all = []
    cur = start_ms
    safety = 0
    while cur < end_ms and safety < 100:
        remaining = max(0, end_ms - cur)
        need = int(math.ceil(remaining / interval_ms))
        limit = min(max(need, 1), 1000)
        data = _fetch_binance_page(cur, end_ms, interval, limit=limit)
        if not isinstance(data, list) or len(data) == 0:
            break
        rows_all.extend(data)
        last_open = int(data[-1][0])
        nxt = last_open + interval_ms
        if nxt <= cur:
            break
        cur = nxt
        safety += 1
        if limit >= 1000:
            time.sleep(0.15)

    if len(rows_all) == 0:
        return pd.Series(dtype=float)

    rows = [(pd.to_datetime(int(k[0]), unit="ms"), float(k[4])) for k in rows_all if start_ms <= int(k[0]) <= end_ms]
    s = pd.Series({t: p for t, p in rows}).sort_index()
    s.index = pd.to_datetime(s.index).astype("datetime64[ns]")
    return s

def fetch_coingecko_series(t_start: datetime, t_end: datetime, freq_min: int) -> pd.Series:
    start_s = int(t_start.replace(tzinfo=timezone.utc).timestamp())
    end_s   = int(t_end.replace(tzinfo=timezone.utc).timestamp())
    url = ("https://api.coingecko.com/api/v3/coins/solana/market_chart/range"
           f"?vs_currency=usd&from={start_s}&to={end_s}")
    with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0"}), timeout=10) as r:
        data = json.loads(r.read().decode())
    rows = data.get("prices", [])
    if not isinstance(rows, list) or len(rows) == 0:
        return pd.Series(dtype=float)
    ts = [pd.to_datetime(int(t), unit="ms") for t, _ in rows]
    pv = [float(p) for _, p in rows]
    s = pd.Series(pv, index=pd.to_datetime(ts)).sort_index()
    s.index = s.index.tz_localize("UTC").tz_convert("UTC").tz_localize(None)
    s = s.resample(f"{freq_min}min").last().ffill()
    return s

def fetch_spot_series(t_start: datetime, t_end: datetime, freq_min: int, source: str = "auto") -> pd.Series:
    if source not in ("auto", "binance", "coingecko"):
        source = "auto"
    if source in ("auto", "binance"):
        try:
            s = fetch_binance_series(t_start, t_end, freq_min)
            if len(s) > 0:
                return s
        except Exception as e:
            print(f"[spot] Binance error: {e}; fallback to CoinGecko…")
    try:
        s = fetch_coingecko_series(t_start, t_end, freq_min)
        return s
    except Exception as e:
        print(f"[spot] CoinGecko error: {e}")
    return pd.Series(dtype=float)

# ------------- Binning helpers -------------

def _decimals(x: float) -> int:
    s = f"{x:.12f}".rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0

def make_price_edges(pmin: float, pmax: float, width: float) -> np.ndarray:
    dec = _decimals(width)
    scale = 10 ** dec
    w_i = int(round(width * scale))
    pmin_i = math.floor((pmin * scale) / w_i) * w_i
    pmax_i = math.ceil((pmax * scale) / w_i) * w_i
    if pmax_i <= pmin_i:
        pmax_i = pmin_i + w_i
    edges_i = np.arange(pmin_i, pmax_i + w_i, w_i, dtype=np.int64)
    y_edges = edges_i.astype(np.float64) / scale
    return np.round(y_edges, dec)

# ------------- Matrix builder -------------

def build_matrix(df, spot_raw, bin_width, pmin=None, pmax=None, freq_min=1):
    df = df.copy()
    df["t_bin"] = pd.to_datetime(df["t_bin"]).dt.floor(f"{freq_min}min")
    t_min, t_max = df["t_bin"].min(), df["t_bin"].max()

    t_edges = pd.date_range(start=t_min, end=t_max + pd.Timedelta(minutes=freq_min), freq=f"{freq_min}min")
    grid_index = pd.date_range(start=t_min, end=t_max, freq=f"{freq_min}min")

    if len(spot_raw) > 0:
        spot_aligned = spot_raw.copy()
        spot_aligned.index = pd.to_datetime(spot_aligned.index).floor(f"{freq_min}min")
        spot_aligned = spot_aligned[~spot_aligned.index.duplicated(keep="last")]
        spot_grid = spot_aligned.reindex(grid_index).ffill().bfill()
    else:
        spot_grid = pd.Series(index=grid_index, dtype=float)

    if pmin is None: pmin = float(np.floor(df["liq_price"].min()))
    if pmax is None: pmax = float(np.ceil(df["liq_price"].max()))
    y_edges = make_price_edges(pmin, pmax, bin_width)
    bins_idx = pd.IntervalIndex.from_breaks(y_edges, closed="left")

    df["p_bin"] = pd.cut(df["liq_price"], bins=bins_idx, include_lowest=True)
    grp = df.groupby(["t_bin", "p_bin"], observed=False)["size_sol"].sum().unstack()
    grp = grp.reindex(index=t_edges[:-1]).fillna(0)
    grp = grp.reindex(columns=bins_idx, fill_value=0)

    Z = grp.to_numpy().T
    return Z, t_edges, y_edges, spot_grid, df

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="SOL-PERP liquidation heatmap + spot price overlay (1-min default)")
    ap.add_argument("--folder", default="data-sol_snapshots", help="folder with sol_positions_log.csv")
    ap.add_argument("--out", default="sol_liq_heatmap.png")
    ap.add_argument("--bin", type=float, default=0.5)
    ap.add_argument("--freq", type=int, default=1)
    ap.add_argument("--pmin", type=float)
    ap.add_argument("--pmax", type=float)
    ap.add_argument("--vmin", type=float)
    ap.add_argument("--vmax", type=float)
    ap.add_argument("--log-color", action="store_true")
    ap.add_argument("--source", choices=["auto", "binance", "coingecko"], default="auto")
    ap.add_argument("--npz-out", default="")
    ap.add_argument("--t-start", type=str, default=None)
    ap.add_argument("--t-end", type=str, default=None)
    args = ap.parse_args()

    if args.t_start and args.t_end:
        t_start = datetime.fromisoformat(args.t_start)
        t_end = datetime.fromisoformat(args.t_end)
    else:
        t_end = datetime.utcnow()
        t_start = t_end - timedelta(hours=24)

    print(f"[time] UTC window: {t_start} ~ {t_end} | freq={args.freq}min")

    df = load_merged_csv(os.path.join(args.folder, "sol_positions_log.csv"), t_start, t_end)
    spot_raw = fetch_spot_series(t_start, t_end, args.freq, source=args.source)
    Z, t_edges, y_edges, spot_grid, dfX = build_matrix(df, spot_raw, args.bin, args.pmin, args.pmax, args.freq)

    norm = LogNorm(vmin=args.vmin, vmax=args.vmax) if args.log_color else None

    if args.npz_out:
        np.savez_compressed(args.npz_out, Z=Z, t_edges=t_edges.astype("datetime64[ns]").astype("int64"),
                            y_edges=y_edges, spot_grid=spot_grid.values,
                            spot_index=spot_grid.index.astype("datetime64[ns]").astype("int64"))

    fig, ax = plt.subplots(figsize=(14, 6))
    X = np.array([pd.Timestamp(t).to_pydatetime() for t in t_edges])
    pcm = ax.pcolormesh(X, y_edges, Z, shading="auto", cmap="viridis", norm=norm)
    fig.colorbar(pcm, ax=ax, pad=0.01).set_label("Positions to liquidate (SOL)")

    if len(spot_grid) > 0 and spot_grid.notna().any():
        spot_line = spot_grid.reindex(pd.to_datetime(t_edges[:-1]).astype("datetime64[ns]"), method="nearest")
        ax.plot(t_edges[:-1], spot_line.values, linewidth=1.2, label="Spot (close)")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("USD")
    ax.set_title(f"SOL-PERP Liquidation Heatmap + Spot | bin={args.bin} | freq={args.freq}min")
    ax.set_ylim(y_edges[0], y_edges[-1])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"✅ Saved figure: {args.out}")

if __name__ == "__main__":
    main()
