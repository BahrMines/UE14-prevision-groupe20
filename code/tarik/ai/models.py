from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from data.custom_types import activation, data_series

"""
Commun
"""

def design(x: data_series, p: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Matrice des lignes "retardées" de p indices
    x: série complète
    p: nombre de retards
    """
    x = np.asarray(x, dtype=float)
    X = np.full((len(x), p), np.nan)
    for k in range(1, p + 1):
        X[k:, k - 1] = x[:-k]
    return X, x


def valid_rows(X: np.ndarray, y: np.ndarray, rows: np.ndarray) -> np.ndarray:
    #Garde les heures dont les retards et la cible sont connus
    rows = np.asarray(rows)
    ok = np.isfinite(X[rows]).all(axis=1) & np.isfinite(y[rows])
    return rows[ok]


def difference(x: data_series) -> np.ndarray:
    #Variations z[t] = x[t] - x[t-1] ; même longueur que x, z[0] = NaN.
    return np.concatenate(([np.nan], np.diff(np.asarray(x, dtype=float))))


"""
Modèles
"""


class Modele(ABC):
    nom: str
    plancher: float | None = 0.0

    def __init__(self, p: int, diff: bool = False) -> None:
        """
        p: retard max
        diff:
        """
        self.p = p
        self.diff = diff

    def to_z(self, x: data_series) -> np.ndarray:
        # z = série sur laquelle les calculs sont faits, x si pas modèle diff
        return difference(x) if self.diff else np.asarray(x, dtype=float)

    def to_x(self, zhat: np.ndarray, x: data_series, rows: np.ndarray) -> np.ndarray:
        # "recoloration" de la liste si elle est diff
        if not self.diff:
            return zhat
        return np.asarray(x, dtype=float)[np.asarray(rows) - 1] + zhat

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray, mu: float, sigma: float) -> Self:
        """
        X: retards d'apprentissage
        y: cibles d'apprentissage
        mu, sigma: moyenne et écart-type de l'apprentissage
        """

    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """X: retards des heures à prévoir -> prévisions"""

    @property
    def label(self) -> str:
        return self.nom + (" diff." if self.diff else "")

    def usable_rows(self, x: data_series, rows: np.ndarray) -> np.ndarray:
        return valid_rows(*design(self.to_z(x), self.p), rows)

    def fit(self, x: data_series, i1: int) -> Self:
        z = self.to_z(x)
        mu, sigma = np.nanmean(z[:i1]), np.nanstd(z[:i1])
        X, y = design(z, self.p)
        rows = valid_rows(X, y, np.arange(i1))
        return self._fit(X[rows], y[rows], mu, sigma)

    def predict(self, x: data_series, rows: np.ndarray) -> np.ndarray:
        X, _ = design(self.to_z(x), self.p)
        xhat = self.to_x(self._predict(X[rows]), x, rows)
        return xhat if self.plancher is None else np.maximum(xhat, self.plancher)


class Persistance(Modele):
    nom = "Pers."
    plancher = None

    def __init__(self) -> None:
        super().__init__(p=1)

    def _fit(self, X: np.ndarray, y: np.ndarray, mu: float, sigma: float) -> Self:
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return X[:, 0] # on décale pas plus car p=1 le fait déjà


class AR(Modele):
    nom = "AR"
    mu: float
    w: np.ndarray

    def __init__(self, p: int, diff: bool = False) -> None:
        super().__init__(p, diff)

    def _fit(self, X: np.ndarray, y: np.ndarray, mu: float, sigma: float) -> Self:
        self.mu = mu

        self.w = np.linalg.lstsq(X - mu, y - mu, rcond=None)[0]

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.mu + (X - self.mu) @ self.w


class ELM(Modele):
    nom = "ELM"
    mu: float
    sigma: float
    H: np.ndarray
    beta: np.ndarray

    def __init__(self, p: int, L: int, seed: int, diff: bool = False,
                 alpha: float = 1e-3, g: activation = np.tanh) -> None:
        """
        p: nb de retards
        L: nb de neurones de la couche cachée
        seed: seed du random de W et b
        diff:
        alpha: force du ridge
        g: fonction d'activation
        """
        super().__init__(p, diff)
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-1, 1, (p, L)) / np.sqrt(p) # /racine(p) : somme de p termes normalisée
        self.b = rng.uniform(-1, 1, L)
        self.alpha, self.g = alpha, g

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        return self.g(((X - self.mu) / self.sigma) @ self.W + self.b)

    def _fit(self, X: np.ndarray, y: np.ndarray, mu: float, sigma: float) -> Self:
        self.mu, self.sigma = mu, sigma
        H = self._hidden(X)
        A = H.T @ H + self.alpha * np.eye(H.shape[1])
        self.beta = np.linalg.lstsq(A, H.T @ (y - mu), rcond=None)[0]
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.mu + self._hidden(X) @ self.beta