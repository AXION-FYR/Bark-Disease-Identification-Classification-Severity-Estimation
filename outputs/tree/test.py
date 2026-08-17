
import pandas as pd

df = pd.read_csv("outputs/tree/tree_severity.csv")

# round for display
df["pct_bark_%"] = (df["pct_bark"] * 100).round(1)
df["qsi"] = df["qsi"].round(4)
df["spread_%"] = (df["spread"] * 100).round(0).astype(int)

# select and rename columns for the table
table = df[[
    "tree", "disease", "pct_bark_%", "qsi",
    "severity_stage", "spread_%", "girdling_risk",
    "action_stage", "n_views", "total_views", "mean_conf"
]].rename(columns={
    "tree":           "Tree",
    "disease":        "Disease",
    "pct_bark_%":     "Area %",
    "qsi":            "QSI",
    "severity_stage": "Severity stage",
    "spread_%":       "Spread %",
    "girdling_risk":  "Girdling",
    "action_stage":   "Action stage",
    "n_views":        "Views (disease)",
    "total_views":    "Views (total)",
    "mean_conf":      "Confidence"
})

print(table.to_string(index=False))
print(f"\nTotal rows: {len(table)}")
print(f"Trees: {df.tree.nunique()}")
print(f"\nAction stage distribution:")
print(df.action_stage.value_counts().to_string())
