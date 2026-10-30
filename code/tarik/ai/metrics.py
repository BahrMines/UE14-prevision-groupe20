import numpy as np
from data.custom_types import data_series

def convert_to_np(series: data_series) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"tableau 1D attendu, reçu {arr.ndim}D de forme {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("le tableau contient des NaN ou des inf")
    return arr


def prepare(x: data_series, xhat: data_series) -> tuple[np.ndarray, np.ndarray]:
    x, xhat = convert_to_np(x), convert_to_np(xhat)
    if len(x) != len(xhat):
        raise ValueError(f"longueurs différentes : {len(x)} et {len(xhat)}")
    return x, xhat


def nMBE(x: data_series, xhat: data_series) -> float:
    x, xhat = prepare(x, xhat)
    return 100 * np.mean(xhat - x) / np.mean(x)


def nMAE(x: data_series, xhat: data_series) -> float:
    x, xhat = prepare(x, xhat)
    return 100 * np.mean(np.abs(xhat - x)) / np.mean(x)


def nRMSE(x: data_series, xhat: data_series) -> float:
    x, xhat = prepare(x, xhat)
    return 100 * np.sqrt(np.mean((xhat - x) ** 2)) / np.mean(x)


def R2(x: data_series, xhat: data_series) -> float:
    x, xhat = prepare(x, xhat)
    return 1 - np.sum((xhat - x) ** 2) / np.sum((x - np.mean(x)) ** 2)

def _L(e: np.ndarray, k: int) -> float:
    return np.mean(np.abs(e) ** k) ** (1 / k)

def nice_sigma(x: data_series, xhat: data_series, xpers: data_series) -> float:
    x, xhat = prepare(x, xhat)
    x, xpers = prepare(x, xpers)
    return np.mean([_L(xhat - x, k) / _L(xpers - x, k) for k in (1, 2, 3)])

def stats4(z: data_series) -> tuple[float, float, float, float]:
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    mu = z.mean()
    sigma = z.std()
    u = (z - mu) / sigma
    return mu, sigma, np.mean(u ** 3), np.mean(u ** 4)