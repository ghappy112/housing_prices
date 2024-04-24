import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load datasets
df = pd.read_json("HPI_master.json")
recession_df = pd.read_csv("JHDUSRGDPBR.csv")

# clean housing prices dataset and engineer features
df = df[(df["hpi_type"]=="traditional") & (df["place_name"]=="United States") & (df["frequency"]=="monthly") & (df["hpi_flavor"]=="purchase-only")][["yr", "period", "index_nsa"]].sort_values(by=["yr", "period"])
df = df.groupby("yr")["index_nsa"].mean().to_frame()
df = pd.DataFrame({"Year": df.index.values, "Housing Price Index": df["index_nsa"].values}).sort_values("Year")
df["Housing Price Index"] = (df["Housing Price Index"] / df["Housing Price Index"].values[0]) * 100

# clean recession dataset and engineer features
recession_df.columns = [recession_df.columns[0], "Recession"]
recession_df["Year"] = recession_df["DATE"].astype(str).str.split("-").apply(lambda x: x[0])
recession_df = recession_df.groupby("Year").max("Recession")
recession_df = pd.DataFrame({"Year": recession_df.index.values, "Recession": recession_df["Recession"].values})
recession_df["Year"] = recession_df["Year"].astype(int)

# join housing prices and recession datasets
df = df.merge(recession_df, how="left", on="Year")
del recession_df
df["Recession"] = df["Recession"].fillna(0)
df["Recession"] = df["Recession"].astype(int)

# export dataset with engineered features
df.to_csv("housing_prices.csv")

# create numpy arrays that will be used to create recession bands in time series plot
y, r = [], []
prev_rec = 0
for year, rec, next_rec in zip(df["Year"].values, df["Recession"].values, df["Recession"].values[1:] + [0]):
    if rec == 1 and prev_rec == 0:
        y.append(year - 0.5)
        r.append(1)
    y.append(year)
    r.append(rec)
    if rec == 1 and next_rec == 0:
        y.append(year + 0.5)
        r.append(1)
    prev_rec = rec
y = np.array(y)
r = np.array(r)

# create and export time series plot
k = 2
plt.figure(figsize=(6.4*k,4.8*k))
sns.set_style("darkgrid")
min_year = df["Year"].values[0]
max_year = df["Year"].values[-1]
plt.xlim(min_year, max_year)
plt.ylim(min(df["Housing Price Index"].values), max(df["Housing Price Index"].values))
sns.lineplot(x="Year", y="Housing Price Index", data=df)
max_ = max(df["Housing Price Index"].values)
plt.fill_between(y, [max_ for x in y], where=np.array(r)==1, color="gray", alpha=0.5, label="Binary = 1")
plt.title(f"Housing Prices ({min_year}-{max_year})")
plt.savefig('housing_prices.png', bbox_inches='tight')
plt.show()
