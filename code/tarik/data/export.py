from pathlib import Path

import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from ai.training import Tuned

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "tarik/stats.csv"
FIGURES = ROOT / "../Resultats" / "figures" / "Tarik"

KEY = "Série (élève)"
DIGITS = {"nmbe": 3, "nmae": 3, "nrmse": 3, "r2": 4, "nice": 4}
METRIC_NAMES = {"nmbe": "nMBE (%)", "nmae": "nMAE (%)", "nrmse": "nRMSE (%)", "r2": "R²", "nice": "NICEΣ"}
STAT_NAMES = ["moyenne", "ecart_type", "asymetrie", "kurtosis"]

INK = "#2b2b29"
MUTED = "#8a8a85"
GRID = "#e6e5e1"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

STYLES = {"Pers.": (MUTED, "-"), "AR": (BLUE, "-"), "AR diff.": (BLUE, "--"),
          "ELM": (ORANGE, "-"), "ELM diff.": (ORANGE, "--")}
NAMES = {"Pers.": "Persistance", "AR": "AR brut", "AR diff.": "AR différencié",
         "ELM": "ELM brut", "ELM diff.": "ELM différencié"}
PARAMETER = {"AR": ("p", "Nombre de retards p"), "ELM": ("L", "Nombre de neurones cachés L")}
VARIANT_NAMES = {"raw": "brute", "diff": "différenciée"}
HOURS = 72
UNIT = "NO₂ (µg/m³)"

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "text.color": INK,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlelocation": "left", "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK, "ytick.color": INK, "legend.frameon": False,
})

class Exports:
    def __init__(self, series: str, domain: str) -> None:
        self.series = series
        self.domain = domain

    def save(self, raw: tuple, diff: tuple, n_train: int, n_test: int, res: dict) -> None:
        row = {KEY: self.series, "Domaine, pas": self.domain, "N appr. / N test": f"{n_train} / {n_test}"}

        for suffix, values in (("brute", raw), ("diff", diff)):
            for name, value in zip(STAT_NAMES, values):
                row[f"{name} {suffix}"] = round(float(value), 4)

        for metric, values in res.items():
            for model, value in values.items():
                row[f"{METRIC_NAMES[metric]} {model}"] = round(value, DIGITS[metric])

        pd.DataFrame([row]).to_csv(TABLE, sep=";", index=False)


def legend_below(ax: plt.Axes, handles: list | None = None) -> None:
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=6)

class Plots:
    def __init__(self, show: bool = False) -> None:
        self.show = show

    def _save(self, fig: plt.Figure, name: str) -> None:
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES / f"{name}.png")

        if not self.show:
            plt.close(fig)

    def display(self) -> None:
        if self.show:
            plt.show()

    def series(self, s: pd.Series, i1: int, i2: int) -> None:
        fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
        periods = {"Apprentissage": (0, i1, BLUE), "Validation": (i1, i2, ORANGE), "Test": (i2, len(s), GREEN)}

        for name, (start, end, color) in periods.items():
            ax.plot(s.index[start:end], s.to_numpy()[start:end], lw=0.5, color=color, label=name)

        ax.set(ylabel=UNIT, title="Série horaire de NO₂ et découpage chronologique")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
        legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

        for handle in legend.legend_handles:
            handle.set_linewidth(2)

        self._save(fig, "series")

    def validation(self, model: str, hp: dict[str, Tuned], pers: float) -> None:
        parameter, xlabel = PARAMETER[model]
        fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)

        for variant, tuned in hp.items():
            curve = getattr(tuned, f"curve_{parameter}")
            chosen = getattr(tuned, parameter)
            color, linestyle = STYLES[model if variant == "raw" else f"{model} diff."]

            ax.plot(list(curve), list(curve.values()), color=color, ls=linestyle, lw=1.5, marker="o", ms=3,
                    label=f"{model} {VARIANT_NAMES[variant]} ({parameter} = {chosen})")

            ax.plot(chosen, curve[chosen], "o", ms=11, mfc="none", mec=color, mew=1.8)

        ax.axhline(pers, color=MUTED, lw=1.2, label="Persistance")

        if parameter == "L":
            ax.set_xscale("log")
            ax.set_xticks(list(curve), [str(L) for L in curve], minor=False)
            ax.minorticks_off()

        ax.set(xlabel=xlabel, ylabel="nRMSE de validation (%)", title=f"Choix de {parameter} pour l'{model}")
        selected = Line2D([], [], ls="", marker="o", ms=11, mfc="none", mec=INK, mew=1.8, label="Valeur retenue")
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [selected], loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)

        self._save(fig, f"validation_{model}")

    def forecast(self, name: str, index: pd.DatetimeIndex, xobs: np.ndarray, pred: dict, hours: int = HOURS) -> None:
        fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
        ax.plot(index[:hours], xobs[:hours], color=INK, lw=1.8, label="Mesure")

        for label, xhat in pred.items():
            color, linestyle = STYLES[label]
            ax.plot(index[:hours], xhat[:hours], color=color, ls=linestyle, lw=1.5, label=NAMES[label])

        ax.set(ylabel=UNIT, title=f"Prévision à 1 h sur 3 jours de test : {name}")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=(0, 12)))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m\n%Hh"))

        legend_below(ax)

        self._save(fig, f"forecast_{name}")

    def nrmse(self, nrmse: dict) -> None:
        fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
        labels = [NAMES[label] for label in nrmse]
        colors = [STYLES[label][0] for label in nrmse]
        bars = ax.bar(labels, list(nrmse.values()), color=colors, width=0.6)

        for bar, label in zip(bars, nrmse):
            if label.endswith("diff."):
                bar.set(hatch="///", edgecolor="white")

        ax.bar_label(bars, fmt="%.1f %%")
        ax.set(ylabel="nRMSE de test (%)", title="Erreur de test des cinq méthodes")
        ax.set_ylim(0, max(nrmse.values()) * 1.12)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=9)
        
        self._save(fig, "nrmse")