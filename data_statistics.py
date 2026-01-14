import pandas as pd

df = pd.read_csv("/Users/gavinyu/Desktop/Air Quality Project/meteo_sensor_together_Joe.csv")

y = pd.to_numeric(df["pm25_mean"], errors="coerce")
print(y.describe())          # count, mean, std, min, quartiles, max
print("std:", y.std())
print("mean:", y.mean())

print(df.columns.tolist())
print(df.dtypes)

df["time_dt"] = pd.to_datetime(df["time"], errors="coerce")

print("time_dt dtype:", df["time_dt"].dtype)
print("time_dt non-null %:", df["time_dt"].notna().mean())
print(df.loc[df["time_dt"].notna(), "time_dt"].head(5))

import numpy as np
import pandas as pd

x = pd.to_numeric(df["timelocal"], errors="coerce").dropna()
print("timelocal min/max:", float(x.min()), float(x.max()))
print("example values:", x.head(10).tolist())

# look at typical step size (difference between consecutive times within a sensor)
d = (
    df.sort_values(["sn", "timelocal"])
      .groupby("sn")["timelocal"]
      .apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().diff().median())
      .dropna()
)
print("median per-sensor step (describe):")
print(d.describe())
