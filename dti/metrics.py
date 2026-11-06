"""Metrics used in the DTA literature: MSE, concordance index, and r_m^2."""
import numpy as np


def mse(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def rmse(y, p):
    return float(np.sqrt(mse(y, p)))


def pearson(y, p):
    return float(np.corrcoef(y, p)[0, 1])


def concordance_index(y, p):
    """Probability that a pair is ordered the same way in truth and prediction.

    This is the CI from Pahikkala et al. 2015 as used by DeepDTA/GraphDTA. O(n^2), so
    for large test sets we sample pairs; DAVIS test is ~5k so the sampling branch rarely
    triggers.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    if n > 6000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=6000, replace=False)
        y, p = y[idx], p[idx]
        n = 6000

    summ = 0.0
    pair = 0
    order = np.argsort(y)
    y, p = y[order], p[order]
    for i in range(1, n):
        for j in range(i):
            if y[i] > y[j]:
                pair += 1
                summ += 1.0 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])
    return float(summ / pair) if pair else 0.0


def r_squared(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot)


def rm2(y, p):
    """r_m^2 index (Roy et al.): penalises models whose regression line misses the diagonal."""
    y, p = np.asarray(y), np.asarray(p)
    r2 = pearson(y, p) ** 2
    k = np.sum(y * p) / np.sum(p ** 2)
    r02 = 1 - np.sum((y - k * p) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return float(r2 * (1 - np.sqrt(abs(r2 - r02))))


def all_metrics(y, p):
    return {
        "mse": round(mse(y, p), 4),
        "rmse": round(rmse(y, p), 4),
        "pearson": round(pearson(y, p), 4),
        "ci": round(concordance_index(y, p), 4),
        "r2": round(r_squared(y, p), 4),
        "rm2": round(rm2(y, p), 4),
    }
