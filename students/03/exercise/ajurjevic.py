import csv
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "09_reaction_time.csv"
OUT_PLOT = Path(__file__).resolve().parent / "reaction_time_plot.png"


def load_data(path):
    xs = []
    ys = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["sleep_hours"]))
            ys.append(float(row["reaction_time"]))
    return np.array(xs).reshape(-1, 1), np.array(ys)


def fit_and_plot(X, y, out_plot):
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_

    # print summary
    print(f"Intercept: {intercept:.2f}")
    print(f"Slope (reaction_time per sleep_hour): {slope:.2f}")

    # plot
    plt.scatter(X, y, label="data")
    xline = np.linspace(X.min() - 1, X.max() + 1, 100).reshape(-1, 1)
    plt.plot(xline, model.predict(xline), color="C1", label="fit")
    plt.xlabel("sleep_hours")
    plt.ylabel("reaction_time (ms)")
    plt.title("Reaction time vs sleep hours")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_plot)
    print(f"Saved plot to {out_plot}")
    return model


if __name__ == "__main__":
    if not DATA_PATH.exists():
        raise SystemExit(f"Data file not found: {DATA_PATH}")
    X, y = load_data(DATA_PATH)
    fit_and_plot(X, y, OUT_PLOT)
