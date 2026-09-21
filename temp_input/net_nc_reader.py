import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

INPUT_FILE = "fair_run.nc"
SCENARIOS = ["VL", "HL"]

ds = xr.open_dataset(INPUT_FILE)

temperatures = ds["temperature"]

# Check dimensions
required_dims = {"timebounds", "scenario", "config"}
if not required_dims.issubset(temperatures.dims):
    raise ValueError(
        f"Expected dimensions {required_dims}, "
        f"but found {temperatures.dims}"
    )

def config_processing(data, mode="median"):
    if mode == "median":
        data = data.median(dim="config")  # Convert to dataframe
        df = data.to_dataframe(name="temperature").reset_index()
        # Keep only requested columns
        df = df[["timebounds", "temperature"]]
    elif mode=="full":
        # Convert to dataframe
        df = data.to_dataframe(name="temperature").reset_index()

        # Arrange configurations as columns
        df = df.pivot(
            index="timebounds",
            columns="config",
            values="temperature"
        ).reset_index()

        # Remove the name from the columns index
        df.columns.name = None
    elif mode=="quantile":
        quantile_levels = np.linspace(0, 1, 10, endpoint=False)
        data = data.quantile(
            quantile_levels,
            dim="config"
        )

        # Convert to dataframe
        df = data.to_dataframe(name="temperature").reset_index()

        # Pivot quantiles into columns
        df = df.pivot(
            index="timebounds",
            columns="quantile",
            values="temperature"
        ).reset_index()

        # Rename quantile columns
        df.columns = [
            "timebounds"
            if col == "timebounds"
            else f"q{int(col * 100):02d}"
            for col in df.columns
        ]
    elif mode=="cluster": # chooses representative trajectory for each quantile
        N_CLUSTERS = 10
        # Focus on the period relevant for tipping
        T = data.sel(timebounds=slice(2000, 2501))

        # config × time
        X = T.transpose("config", "timebounds").values

        # Cluster trajectories
        kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X)

        # Find the actual configuration closest to each cluster centroid
        representatives = []

        for cluster in range(N_CLUSTERS):
            members = np.where(labels == cluster)[0]

            centroid = kmeans.cluster_centers_[cluster]

            distances = np.linalg.norm(
                X[members] - centroid,
                axis=1
            )

            representative = members[np.argmin(distances)]
            representatives.append(representative)

        representatives = np.array(representatives)
        # Actual FAIR configuration IDs
        config_ids = T.config.values[representatives]
        data = data.sel(config=config_ids)
        # Convert to dataframe
        df = data.to_dataframe(name="temperature").reset_index()

        # Arrange configurations as columns
        df = df.pivot(
            index="timebounds",
            columns="config",
            values="temperature"
        ).reset_index()

        # Remove the name from the columns index
        df.columns.name = None
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return df

fig, ax = plt.subplots()

for scenario in SCENARIOS:
    # Select scenario — keep all configurations
    data = temperatures.sel(scenario=scenario)
    df = config_processing(data, "cluster")
    # Save
    output_file = f"trajectories/temperature_{scenario}.csv"
    df.to_csv(output_file, index=False)

    print(
        f"{scenario}: {len(df)} time points, "
        f"{len(df.columns) - 1} configurations "
        f"-> {output_file}"
    )
    ax.plot(df["timebounds"], df.drop(columns="timebounds"), color="tab:blue" if scenario == "VL" else "tab:red", alpha=0.1)
plt.show()