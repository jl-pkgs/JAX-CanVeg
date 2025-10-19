# %%
from jax_canveg.subjects import Met
import pandas as pd
import os


def test_Met():
    f = "data/Forcing_US-Whs.csv"
    df = pd.read_csv(f)
    met = Met.from_df(df)
    met

    met.to_csv("temp.csv") # also works
    os.remove("temp.csv")


if __name__ == "__main__":
    test_Met()
