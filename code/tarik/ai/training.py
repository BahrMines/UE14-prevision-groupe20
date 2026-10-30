from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from data.custom_types import data_series
from ai.metrics import R2, nMAE, nMBE, nRMSE, nice_sigma
from ai.models import AR, ELM, Modele, Persistance

P_MAX = 24
L_GRID = [5, 10, 20, 50, 100, 200, 500]
N_DRAWS = 20
TOL = 0.005
SEED = 2026
VARIANTS = {"raw": False, "diff": True}


def common_rows(x: data_series, rows: np.ndarray, models: list[Modele]) -> np.ndarray:
    for model in models:
        rows = model.usable_rows(x, rows)
    return rows


def choose_simplest(errors: dict) -> int:
    #Plus petite clé (p ou L) dont l'erreur est à moins de TOL du minimum.
    threshold = min(errors.values()) * (1 + TOL)
    return next(key for key, error in errors.items() if error <= threshold)


@dataclass
class Tuned:
    p: int
    L: int
    seed: int
    curve_p: dict[int, float]
    curve_L: dict[int, float]


def tune(err: Callable[[Modele], float], diff: bool) -> Tuned:
    curve_p = {p: err(AR(p, diff)) for p in range(1, P_MAX + 1)}
    p = choose_simplest(curve_p)

    best = {}                                            # pour chaque L : (erreur, graine) du meilleur tirage
    for L in L_GRID:
        seeds = [SEED + 1000 * L + k for k in range(N_DRAWS)]
        best[L] = min((err(ELM(p, L, seed, diff)), seed) for seed in seeds)
    curve_L = {L: error for L, (error, _) in best.items()}
    L = choose_simplest(curve_L)

    return Tuned(p, L, best[L][1], curve_p, curve_L)


def select_hyperparameters(x: data_series, i1: int, i2: int) -> tuple[dict[str, Tuned], float]:
    #Choisit p, L et la seed sur la validation
    rows = common_rows(x, np.arange(i1, i2), [AR(P_MAX, diff) for diff in VARIANTS.values()])
    xv = np.asarray(x, dtype=float)[rows]

    def err(model: Modele) -> float:      # erreur de validation d'un modèle
        return nRMSE(xv, model.fit(x, i1).predict(x, rows))

    pers = err(Persistance())
    print(f"Validation : {len(rows)} heures ; nRMSE persistance = {pers:.3f} %")

    hp = {}
    for name, diff in VARIANTS.items():
        tuned = tune(err, diff)
        hp[name] = tuned
        print(f"[{name}] AR : p = {tuned.p} (nRMSE {tuned.curve_p[tuned.p]:.3f} %) | "
              f"ELM : L = {tuned.L}, graine {tuned.seed} (nRMSE {tuned.curve_L[tuned.L]:.3f} %)")
    return hp, pers


def predict_test(x: data_series, i1: int, i2: int, hp: dict[str, Tuned]) -> tuple[np.ndarray, dict]:
    #Prévisions des 5 méthodes sur le test avec les réglages retenus
    raw, diff = hp["raw"], hp["diff"]
    models = [
        Persistance(),
        AR(raw.p),
        ELM(raw.p, raw.L, raw.seed),
        AR(diff.p, True),
        ELM(diff.p, diff.L, diff.seed, True),
    ]
    rows = common_rows(x, np.arange(i2, len(x)), models)
    return rows, {model.label: model.fit(x, i1).predict(x, rows) for model in models}


def compute_metrics(xobs: np.ndarray, pred: dict) -> dict:
    functions = {
        "nmbe": nMBE,
        "nmae": nMAE,
        "nrmse": nRMSE,
        "r2": R2,
        "nice": lambda x, xhat: nice_sigma(x, xhat, pred["Pers."]),
    }
    res = {name: {label: float(f(xobs, xhat)) for label, xhat in pred.items()} for name, f in functions.items()}
    assert abs(res["nice"]["Pers."] - 1) < 1e-12, "le NICE de la persistance doit valoir 1"
    return res