import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .config import DATA_PATH

OUTDIR = DATA_PATH.parent / "eda_charts"
OUTDIR.mkdir(exist_ok=True, parents=True)

def main():
    df = pd.read_csv(DATA_PATH)

    # Class balance
    n_total = len(df)
    pos = int(df["fraud"].sum())
    neg = n_total - pos
    fig, ax = plt.subplots(figsize=(5,5))
    ax.pie([neg, pos], labels=["Benign", "Fraud"], autopct="%1.1f%%", startangle=140)
    ax.set_title("Class Balance (Fraud vs Benign)")
    fig.savefig(OUTDIR / "class_balance_pie.png", bbox_inches="tight"); plt.close(fig)

    # Amount distribution (fraud vs benign)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.hist(df[df["fraud"]==0]["amount"], bins=100, alpha=0.5, label="Benign")
    ax.hist(df[df["fraud"]==1]["amount"], bins=100, alpha=0.5, label="Fraud")
    ax.set_xlim(0, df["amount"].quantile(0.99))
    ax.set_title("Transaction Amount Distribution")
    ax.set_xlabel("Amount"); ax.set_ylabel("Density"); ax.legend()
    fig.tight_layout(); fig.savefig(OUTDIR / "amount_distribution.png", bbox_inches="tight"); plt.close(fig)

    # Fraud rate by age (category code)
    fr_age = df.groupby("age")["fraud"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6,4))
    fr_age.plot(kind="bar", ax=ax, color="orange")
    ax.set_title("Fraud Rate by Age Group"); ax.set_ylabel("Fraud rate")
    fig.tight_layout(); fig.savefig(OUTDIR / "fraud_rate_by_age.png", bbox_inches="tight"); plt.close(fig)

    # Fraud rate by category (top by rate, support>=500)
    if "category" in df.columns:
        cat_stats = (df.groupby("category")["fraud"]
                     .agg(["mean","count"])
                     .rename(columns={"mean":"fraud_rate","count":"n"})
                     .sort_values("fraud_rate", ascending=False))
        cat_stats = cat_stats[cat_stats["n"]>=500].head(10)
        fig, ax = plt.subplots(figsize=(7,4))
        cat_stats["fraud_rate"].plot(kind="bar", ax=ax, color="green")
        ax.set_title("Top Categories by Fraud Rate (support ≥ 500)")
        ax.set_ylabel("Fraud rate")
        fig.tight_layout(); fig.savefig(OUTDIR / "fraud_rate_by_category.png", bbox_inches="tight"); plt.close(fig)

    print("EDA charts saved to:", OUTDIR)

if __name__ == "__main__":
    main()
