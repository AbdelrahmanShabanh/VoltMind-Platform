import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import os
from google.colab import drive

# ===============================
# CONFIGURATION
# ===============================
NUM_HOUSES = 100
DAYS = 30
FREQ = "1min"

# Mount Google Drive
drive.mount('/content/drive')

# Set output folder to a directory in your Drive
OUTPUT_FOLDER = "/content/drive/MyDrive/realistic_multi_house_raw"

np.random.seed(42)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Time index
date_range = pd.date_range(start="2024-01-01", periods=60*24*DAYS, freq=FREQ)

for house_id in range(1, NUM_HOUSES + 1):

    df = pd.DataFrame()
    df["DateTime"] = date_range
    df["Date"] = df["DateTime"].dt.date
    df["Time"] = df["DateTime"].dt.time

    hour = df["DateTime"].dt.hour
    dayofweek = df["DateTime"].dt.dayofweek

    # ===============================
    # REALISTIC POWER MODEL
    # ===============================

    # Base night load
    base_load = np.random.uniform(0.2, 0.5)

    # Morning peak (6 9 AM)
    morning_peak = np.where((hour >= 6) & (hour <= 9),
                            np.random.uniform(1.5, 3.5), 0)

    # Evening peak (6 10 PM)
    evening_peak = np.where((hour >= 18) & (hour <= 22),
                            np.random.uniform(2.0, 4.5), 0)

    # Weekend slightly higher usage
    weekend_factor = np.where(dayofweek >= 5, 1.2, 1)

    # Total power
    global_active_power = (
        base_load +
        morning_peak +
        evening_peak +
        np.random.normal(0, 0.2, len(df))
    ) * weekend_factor

    # Clip to realistic range
    global_active_power = np.clip(global_active_power, 0.1, 6)

    # ===============================
    # Voltage (realistic EU)
    # ===============================
    voltage = 230 + np.random.normal(0, 4, len(df))
    voltage = np.clip(voltage, 220, 240)

    # ===============================
    # Intensity (I = P*1000 / V)
    # ===============================
    global_intensity = (global_active_power * 1000) / voltage

    # ===============================
    # Sub-metering (kW portions)
    # ===============================
    sub1 = global_active_power * np.random.uniform(0.2, 0.4)
    sub2 = global_active_power * np.random.uniform(0.1, 0.3)
    sub3 = global_active_power * np.random.uniform(0.1, 0.2)

    df["Global_active_power"] = global_active_power
    df["Voltage"] = voltage
    df["Global_intensity"] = global_intensity
    df["Sub_metering_1"] = sub1
    df["Sub_metering_2"] = sub2
    df["Sub_metering_3"] = sub3

    df = df.drop(columns=["DateTime"])

    # ===============================
    # ADD RAW DATA ISSUES
    # ===============================

    # Missing values (3%)
    for col in df.columns[2:]:
        df.loc[df.sample(frac=0.03).index, col] = np.nan

    # Outliers (1%)   extreme spikes
    spike_idx = df.sample(frac=0.01).index
    df.loc[spike_idx, "Global_active_power"] *= 3

    # Duplicate rows (1%)
    duplicates = df.sample(frac=0.01)
    df = pd.concat([df, duplicates])

    df.to_csv(f"{OUTPUT_FOLDER}/house_{house_id}.csv", index=False)

print("Realistic multi-house RAW dataset created successfully.")
