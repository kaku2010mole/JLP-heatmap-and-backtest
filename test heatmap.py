import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV ===
df = pd.read_csv("/Users/posley3302_15/Desktop/0825 backtest data.csv")
df = df.rename(columns={
    '@timestamp per minute': 'timestamp',
    'TargetSOL_Pct': 'target_pct',
    'SOL_HYPE_perpPrice': 'sol_price',
    'SOLCurrentPos': 'sol_pos',
    'CurrentSOL_Pct': 'current_pct',
    'JLP Amount': 'jlp_amt',
    'JLP Price': 'jlp_price'
})
for col in ["sol_pos", "target_pct", "current_pct", "sol_price", "jlp_amt", "jlp_price"]:
    df[col] = df[col].astype(str).str.replace(",", "").astype(float)

# 将 HKT 转为 UTC（CSV 是 HKT 时间）
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("Asia/Hong_Kong").dt.tz_convert("UTC")

df["price_diff"] = df["sol_price"].diff()

# === Load heatmap NPZ ===
data = np.load("/Users/posley3302_15/Desktop/JLP_heatmap_v2/liq_matrix_72h_1m.npz")
Z = data["Z"]  # shape [price_bins, time_steps]
price_bins = data["y_edges"][:-1]  # [500] price intervals (y-axis)

# === Boost control (tracking actual position) with reversal during cooldown ===
boost_pct_abs = 0.5
boosted_actual_pct = []
boost_applied = False
confirm = 0
release = 0
direction = 0
cooldown_threshold = 2
boost_toggle_points = []

for i in range(len(df)):
    actual = df.loc[i, "current_pct"]
    boosted = actual

    if i >= Z.shape[1]:
        boosted_actual_pct.append(actual)
        continue

    cur_price = df.loc[i, "sol_price"]
    mask_upper = (price_bins > cur_price) & (price_bins <= cur_price + 10)
    mask_lower = (price_bins >= cur_price - 10) & (price_bins < cur_price)

    usd_sum_upper = Z[mask_upper, i].sum()
    usd_sum_lower = Z[mask_lower, i].sum()
    sol_sum_upper = usd_sum_upper / cur_price
    sol_sum_lower = usd_sum_lower / cur_price

    trigger = (sol_sum_upper > 500 or sol_sum_lower > 500)
    new_direction = 1 if sol_sum_lower > sol_sum_upper else -1

    if trigger:
        confirm += 1
        # 冷静期内遇到反方向信号，立即中断冷静并反转
        if boost_applied and release > 0 and new_direction != direction:
            print(f"[{df['timestamp'].iloc[i]}] Reversing during cooldown: {direction} → {new_direction}")
            direction = new_direction
            release = 0
            confirm = 1
            boost_toggle_points.append((df["timestamp"].iloc[i], "reverse"))
        else:
            release = 0
    else:
        confirm = 0
        release += 1

    # 启动 boost
    if not boost_applied and confirm >= 1:
        boost_applied = True
        direction = new_direction
        boost_toggle_points.append((df["timestamp"].iloc[i], "on"))

    # 冷静期结束，取消 boost
    if boost_applied and release >= cooldown_threshold:
        boost_applied = False
        direction = 0
        boost_toggle_points.append((df["timestamp"].iloc[i], "off"))

    # 应用 boost
    if boost_applied:
        boosted += direction * boost_pct_abs

    boosted_actual_pct.append(boosted)

df["boosted_actual_pct"] = boosted_actual_pct

# === Delta Loss Calculation: always vs. raw target ===
df["prev_sol_pos"] = df["sol_pos"].shift(1)
df["prev_pct"] = df["current_pct"].shift(1)
df["prev_target"] = df["target_pct"].shift(1)

# --- No boost: baseline ---
df["theoretical_raw"] = df["prev_sol_pos"] / df["prev_pct"] * df["prev_target"]
df["delta_loss_raw"] = (df["prev_sol_pos"] - df["theoretical_raw"]) * df["price_diff"]
df["delta_loss_raw"] = df["delta_loss_raw"].fillna(0)
df["cumulative_delta_loss_raw"] = df["delta_loss_raw"].cumsum()

# --- With boost: actual follows boosted_actual_pct, but compare to raw target ---
df["prev_boosted_actual_pct"] = df["boosted_actual_pct"].shift(1)
df["theoretical_boosted"] = df["prev_sol_pos"] / df["prev_boosted_actual_pct"] * df["prev_target"]
df["delta_loss_boosted"] = (df["prev_sol_pos"] - df["theoretical_boosted"]) * df["price_diff"]
df["delta_loss_boosted"] = df["delta_loss_boosted"].fillna(0)
df["cumulative_delta_loss_boosted"] = df["delta_loss_boosted"].cumsum()

# === Save CSV for external analysis ===
df.to_csv("boosted_backtest_with_delta_loss.csv", index=False)

# === Plot 1: Position % + Price ===
fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(df["timestamp"], df["target_pct"], label="Raw Target %", color="tab:blue")
ax1.plot(df["timestamp"], df["boosted_actual_pct"], label="Boosted Actual %", color="tab:orange", linestyle="--")
ax1.plot(df["timestamp"], df["current_pct"], label="Actual %", color="tab:green", linestyle="-.")

ax1.set_ylabel("Position Ratio (%)")
ax2 = ax1.twinx()
ax2.plot(df["timestamp"], df["sol_price"], label="SOL Price", color="tab:red", alpha=0.3)
ax2.set_ylabel("SOL Price (USD)")

for ts, status in boost_toggle_points:
    color = {"on": "green", "off": "red", "reverse": "purple"}.get(status, "gray")
    ax1.axvline(x=ts, color=color, linestyle="dotted", alpha=0.6)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
fig.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.title("Target vs Boosted Actual vs Real Position + Price")
plt.grid(True)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# === Plot 2: Delta Loss Comparison ===
plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["cumulative_delta_loss_raw"], label="No Boost", color="tab:gray")
plt.plot(df["timestamp"], df["cumulative_delta_loss_boosted"], label="Directional Boost (w/ reversal)", color="tab:red", linestyle="--")
plt.ylabel("Cumulative Delta Loss (USD)")
plt.xlabel("Time")
plt.title("Cumulative Delta Loss: With vs Without Directional Boost")
plt.grid(True)
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

