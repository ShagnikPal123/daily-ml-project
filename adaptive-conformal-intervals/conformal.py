"""Split-conformal prediction intervals on top of any scikit-learn regressor.

Idea: a model that says "42" is less useful than one that says "42, and I am
90% sure it is between 38 and 47". Conformal prediction gives that guarantee
without assuming the noise is Gaussian. We use a *normalized* score so the
interval is wider where the model is less certain (heteroscedastic noise).
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)


def make_data(n=3000):
    x = rng.uniform(-3, 3, size=(n, 1))
    noise_scale = 0.1 + 0.6 * np.abs(x[:, 0])          # noise grows away from 0
    y = np.sin(2 * x[:, 0]) + rng.normal(0, noise_scale)
    return x, y


class ConformalRegressor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.mean = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0)
        self.spread = GradientBoostingRegressor(n_estimators=100, max_depth=2, random_state=0)

    def fit(self, X, y):
        X_tr, X_cal, y_tr, y_cal = train_test_split(X, y, test_size=0.4, random_state=0)
        self.mean.fit(X_tr, y_tr)
        # second model learns how big the error tends to be at each x
        self.spread.fit(X_tr, np.abs(y_tr - self.mean.predict(X_tr)))
        sigma = np.maximum(self.spread.predict(X_cal), 1e-3)
        scores = np.abs(y_cal - self.mean.predict(X_cal)) / sigma
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q = np.quantile(scores, min(q, 1.0), method="higher")
        return self

    def predict(self, X):
        mu = self.mean.predict(X)
        half = self.q * np.maximum(self.spread.predict(X), 1e-3)
        return mu, mu - half, mu + half


if __name__ == "__main__":
    X, y = make_data()
    X_fit, X_test, y_fit, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    for alpha in (0.2, 0.1, 0.05):
        m = ConformalRegressor(alpha).fit(X_fit, y_fit)
        _, lo, hi = m.predict(X_test)
        cover = np.mean((y_test >= lo) & (y_test <= hi))
        print(f"target {1-alpha:.0%}  empirical coverage {cover:.1%}  mean width {np.mean(hi-lo):.2f}")

    m = ConformalRegressor(0.1).fit(X_fit, y_fit)
    order = np.argsort(X_test[:, 0])
    xs = X_test[order, 0]
    mu, lo, hi = m.predict(X_test[order])
    plt.figure(figsize=(8, 4.5))
    plt.scatter(X_test[:, 0], y_test, s=5, alpha=0.3, label="test data")
    plt.plot(xs, mu, color="C1", label="prediction")
    plt.fill_between(xs, lo, hi, color="C1", alpha=0.25, label="90% interval")
    plt.legend(); plt.title("Adaptive conformal intervals"); plt.tight_layout()
    plt.savefig("intervals.png", dpi=120)
