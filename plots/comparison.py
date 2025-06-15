import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------------- تنظیمات مسیرها -------------------------
baseline_model = "REINFORCE-b"     # ← با baseline
no_baseline_model = "REINFORCE"    # ← بدون baseline

baseline_path = f"analysis/{baseline_model}/returns_per_episode.npy"
no_baseline_path = f"analysis/{no_baseline_model}/returns_per_episode.npy"

# محل ذخیره نمودار خروجی
output_path = "report/comparison_returns.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ------------------------- تنظیمات متحرک‌سازی -------------------------
window_size = 100

def moving_average(data, window_size):
    if data is None or len(data) < window_size:
        print(f"[WARNING] Skipped MA (len={len(data)})")
        return np.array([])
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ------------------------- بارگذاری داده‌ها -------------------------
baseline_returns = np.load(baseline_path)
no_baseline_returns = np.load(no_baseline_path)

baseline_avg = moving_average(baseline_returns, window_size)
no_baseline_avg = moving_average(no_baseline_returns, window_size)

# اطمینان از تطابق طول‌ها
min_len = min(len(baseline_avg), len(no_baseline_avg))
episodes = np.arange(min_len)
baseline_avg = baseline_avg[:min_len]
no_baseline_avg = no_baseline_avg[:min_len]

# ------------------------- رسم نمودار مقایسه‌ای -------------------------
plt.figure(figsize=(12, 6))
plt.plot(episodes, no_baseline_avg, label="REINFORCE (No Baseline)", linewidth=2, color="red")
plt.plot(episodes, baseline_avg, label="REINFORCE (With Baseline)", linewidth=2, color="blue")

plt.title("Smoothed Returns Comparison (avg over 100 episodes)")
plt.xlabel("Episode")
plt.ylabel("Average Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Comparison plot saved to: {output_path}")
