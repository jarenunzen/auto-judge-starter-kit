import pandas as pd
import numpy as np

df = pd.read_json(r"C:\Users\jaren\autojudge\my-judge\correlations-2026-02-25_16-22-46.jsonl", lines=True)

df = df.round(3)

df.to_latex(
    "results_table.csv",
    index=False,
)