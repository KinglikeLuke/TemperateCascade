import xarray as xr
import matplotlib.pyplot as plt
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

fig, ax = plt.subplots()

for scenario in SCENARIOS:
    # Select scenario — keep all configurations
    data = temperatures.sel(scenario=scenario)

    # # Convert to dataframe
    # df = data.to_dataframe(name="temperature").reset_index()
    #
    # # Arrange configurations as columns
    # df = df.pivot(
    #     index="timebounds",
    #     columns="config",
    #     values="temperature"
    # ).reset_index()
    #
    # # Remove the name from the columns index
    # df.columns.name = None
    # Arithmetic mean across all configurations
    data = data.mean(dim="config") # Convert to dataframe
    df = data.to_dataframe(name="temperature").reset_index()
    # Keep only requested columns
    df = df[["timebounds", "temperature"]]
    # Save
    output_file = f"trajectories/temperature_{scenario}.csv"
    df.to_csv(output_file, index=False)

    print(
        f"{scenario}: {len(df)} time points, "
        f"{len(df.columns) - 1} configurations "
        f"-> {output_file}"
    )
    ax.plot(df["timebounds"], df.drop(columns="timebounds"), color="tab:blue" if scenario == "VL" else "tab:red", alpha=0.01)
plt.show()