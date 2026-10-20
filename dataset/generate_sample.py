"""Creates SAMPLE DATA (synthetic) for testing: one year of hourly household consumption in kWh.
A few missing values, duplicate rows and spikes are inserted on purpose to demonstrate preprocessing."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
idx = pd.date_range("2024-01-01 00:00", "2024-12-31 23:00", freq="h")
hour, dow, doy = idx.hour.to_numpy(), idx.dayofweek.to_numpy(), idx.dayofyear.to_numpy()
base = 0.45
morning = 0.55 * np.exp(-((hour - 7) ** 2) / 4)
evening = 1.1 * np.exp(-((hour - 20) ** 2) / 5)
weekend = np.where(dow >= 5, 0.25 * np.exp(-((hour - 13) ** 2) / 12) + 0.1, 0)
season = 0.35 * np.cos(2 * np.pi * (doy - 200) / 365) + 0.25   # summer cooling load
kwh = base + morning + evening + weekend + season + rng.normal(0, 0.12, len(idx))
kwh = np.clip(kwh, 0.1, None)
df = pd.DataFrame({"datetime": idx.strftime("%Y-%m-%d %H:%M:%S"), "consumption_kwh": np.round(kwh, 3)})
df.loc[rng.choice(len(df), 25, replace=False), "consumption_kwh"] = np.nan       # missing values
df.loc[rng.choice(len(df), 4, replace=False), "consumption_kwh"] *= 6            # spikes
df = pd.concat([df, df.sample(6, random_state=1)]).sort_values("datetime", kind="stable")  # duplicates
df.to_csv("dataset/electricity_consumption.csv", index=False)
print("rows:", len(df))
